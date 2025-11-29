# gesture_ws_server.py
# FastAPI WebSocket server that streams gesture control signals with calibration.
# Run:
#   pip install fastapi "uvicorn[standard]" opencv-python mediapipe numpy
#   uvicorn gesture_ws_server:app --host 127.0.0.1 --port 8765

import asyncio
import json
import time
from typing import Set

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="NeuroSync Gesture WS Server")

# ----------------- TUNABLE CONSTANTS (fallback defaults) -----------------
CAM_INDEX = 0
FPS = 20
SMOOTHING_ALPHA = 0.6

DEFAULT_PALM_OPEN = 0.45
DEFAULT_FIST = 0.23
DEFAULT_THUMBUP = 30
DEFAULT_STEER_SCALE = 35
# -------------------------------------------------------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ----------------- CALIBRATION VARIABLES -----------------
CALIBRATING = False
CALIBRATION_FRAMES = []
CALIBRATION_START_TIME = 0
CALIBRATION_DURATION = 2.0  # seconds

BASE_OPEN = None
BASE_STEER = None
BASE_HAND_Z = None

PALM_OPEN_THRESHOLD = DEFAULT_PALM_OPEN
FIST_THRESHOLD = DEFAULT_FIST
STEER_SCALE = DEFAULT_STEER_SCALE
# ----------------------------------------------------------


# ----------------- CONNECTION HANDLER -----------------
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active.discard(websocket)

    async def broadcast(self, message: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()
# ---------------------------------------------------------


# ----------------- HELPERS -----------------
def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def normalized_openness(lm, w, h):
    """Palm-open metric: distance between index_mcp & pinky_mcp normalized."""
    left_pt = lm[5]
    pinky_pt = lm[17]
    width = abs((left_pt.x - pinky_pt.x) * w)

    palm = euclid((lm[0].x*w, lm[0].y*h), (lm[9].x*w, lm[9].y*h)) + 1e-6
    return float(np.clip(width / palm, 0, 1))

def thumb_angle(lm, w, h):
    wrist = np.array([lm[0].x*w, lm[0].y*h])
    index = np.array([lm[5].x*w, lm[5].y*h])
    thumb = np.array([lm[4].x*w, lm[4].y*h])
    v1 = index - wrist
    v2 = thumb - wrist
    dot = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    cosang = np.clip(dot / denom, -1, 1)
    return float(np.degrees(np.arccos(cosang)))

def steering_angle(lm, w, h):
    wrist = np.array([lm[0].x*w, lm[0].y*h])
    mid = np.array([lm[9].x*w, lm[9].y*h])
    v = mid - wrist
    angle = np.degrees(np.arctan2(v[1], v[0]))  # -180 to 180
    tilt = -(angle + 90)
    return float(tilt)
# ---------------------------------------------------------


# ----------------- CALIBRATION LOGIC -----------------
def finalize_calibration():
    global BASE_OPEN, BASE_STEER, BASE_HAND_Z
    global PALM_OPEN_THRESHOLD, FIST_THRESHOLD, STEER_SCALE

    if not CALIBRATION_FRAMES:
        print("Calibration failed: No frames collected.")
        return

    opens = [f["open"] for f in CALIBRATION_FRAMES]
    steers = [f["steer"] for f in CALIBRATION_FRAMES]
    dists = [f["hand_z"] for f in CALIBRATION_FRAMES]

    BASE_OPEN = float(np.mean(opens))
    BASE_STEER = float(np.mean(steers))
    BASE_HAND_Z = float(np.mean(dists))

    PALM_OPEN_THRESHOLD = BASE_OPEN + 0.12
    FIST_THRESHOLD = BASE_OPEN - 0.06
    STEER_SCALE = max(15, 25 + abs(BASE_STEER) * 40)

    print("\n======= CALIBRATION COMPLETE =======")
    print("Baseline openness:", BASE_OPEN)
    print("Baseline steering:", BASE_STEER)
    print("Baseline hand_z:", BASE_HAND_Z)
    print("New PALM_OPEN_THRESHOLD:", PALM_OPEN_THRESHOLD)
    print("New FIST_THRESHOLD:", FIST_THRESHOLD)
    print("New STEER_SCALE:", STEER_SCALE)
    print("====================================\n")


@app.get("/calibrate")
async def begin_calibration():
    global CALIBRATING, CALIBRATION_FRAMES, CALIBRATION_START_TIME
    CALIBRATING = True
    CALIBRATION_FRAMES = []
    CALIBRATION_START_TIME = time.time()
    return {"status": "calibration_started", "duration": CALIBRATION_DURATION}
# ---------------------------------------------------------


# ----------------- MAIN CAMERA LOOP -----------------
_smoothed_accel = 0.0
_smoothed_steer = 0.0

async def camera_loop():
    global _smoothed_accel, _smoothed_steer, CALIBRATING

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open camera!")
        return

    print("Camera loop running...")

    target_period = 1.0 / FPS

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        ts = time.time()

        out = {
            "timestamp": ts,
            "accelerate": 0,
            "brake": 0,
            "steer": 0,
            "boost": 0,
            "hand_z": 0,
            "confidence": 0
        }

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark

            # metrics
            open_val = normalized_openness(lm, w, h)
            thumb_val = thumb_angle(lm, w, h)
            steer_raw = steering_angle(lm, w, h)

            # hand depth
            wrist = np.array([lm[0].x*w, lm[0].y*h])
            mid = np.array([lm[9].x*w, lm[9].y*h])
            diag = np.hypot(w, h)
            hand_z = float(np.clip(euclid(wrist, mid) / diag, 0, 1))

            # ---- DURING CALIBRATION ----
            if CALIBRATING:
                CALIBRATION_FRAMES.append({"open": open_val, "steer": steer_raw, "hand_z": hand_z})
                await manager.broadcast(json.dumps({"status": "calibrating"}))

                if time.time() - CALIBRATION_START_TIME >= CALIBRATION_DURATION:
                    finalize_calibration()
                    CALIBRATING = False

                await asyncio.sleep(target_period)
                continue

            # ---- NORMAL MODE ----
            # apply baseline steering recentering
            if BASE_STEER is not None:
                steer_raw -= BASE_STEER

            # gestures
            is_open = open_val > PALM_OPEN_THRESHOLD
            is_fist = open_val < FIST_THRESHOLD
            is_thumb_up = thumb_val < DEFAULT_THUMBUP

            # smoothing
            accel_raw = 1.0 if is_open else 0.0
            _smoothed_accel = SMOOTHING_ALPHA * _smoothed_accel + (1 - SMOOTHING_ALPHA) * accel_raw
            _smoothed_steer = SMOOTHING_ALPHA * _smoothed_steer + (1 - SMOOTHING_ALPHA) * (steer_raw / STEER_SCALE)

            out["accelerate"] = float(np.clip(_smoothed_accel, 0, 1))
            out["steer"] = float(np.clip(_smoothed_steer, -1, 1))
            out["brake"] = 1 if is_fist else 0
            out["boost"] = 1 if is_thumb_up and not is_fist else 0
            out["hand_z"] = hand_z

        # send to WebSocket clients
        await manager.broadcast(json.dumps(out))

        # keep FPS stable
        dt = time.time() - t0
        await asyncio.sleep(max(0, target_period - dt))


# ----------------- ROUTES -----------------
@app.get("/")
async def index_page():
    html = """
    <html>
      <body>
        <h2>NeuroSync Gesture WS Server</h2>
        <p>Connect your game to: <b>ws://127.0.0.1:8765/ws</b></p>
        <p>Start calibration: <b>GET /calibrate</b></p>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await manager.connect(ws)
    await ws.send_text(json.dumps({"status": "connected"}))

    try:
        while True:
            try:
                await ws.receive_text()
            except:
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.on_event("startup")
async def on_start():
    asyncio.create_task(camera_loop())
    print("Server ready. Gesture tracking online.")


# ----------------- RUN -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gesture_ws_server:app", host="127.0.0.1", port=8765, reload=True)
