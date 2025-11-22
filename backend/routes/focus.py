from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
import csv
import threading

router = APIRouter()

CALIBRATION_SECONDS = 3.0
SCORE_HISTORY_LENGTH = 100
METRICS_BUFFER_LENGTH = 500
SMOOTHING_ALPHA = 0.6
MICROSLEEP_DURATION = 0.30
BLINK_MIN_INTERVAL = 0.15
GAZE_X_THRESHOLD = 0.35
GAZE_Y_THRESHOLD = 0.5
HEAD_YAW_LIMIT = 35.0
HEAD_PITCH_LIMIT = 30.0
LOG_CSV = "focus_log.csv"

WEIGHT_EYE = 0.55
WEIGHT_HEAD = 0.30
WEIGHT_GAZE = 0.15

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
IRIS_LEFT_IDX = [468, 469, 470, 471]
IRIS_RIGHT_IDX = [473, 474, 475, 476]
PNP_IDX = [1, 152, 33, 263, 61, 291]

latest_frame = None
metrics_buffer = deque(maxlen=METRICS_BUFFER_LENGTH)
latest_metrics = {
    "timestamp": None,
    "focus_percent": 0.0,
    "blink_per_min": 0,
    "gaze_x": 0.0,
    "gaze_y": 0.0,
    "yaw": 0.0,
    "pitch": 0.0,
    "status": "Initializing"
}

def euclid(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)

def eye_aspect_ratio(landmarks, w, h, left=True):
    idx = LEFT_EYE_IDX if left else RIGHT_EYE_IDX
    pts = [landmarks[i] for i in idx]
    vert = (euclid(pts[1], pts[5], w, h) + euclid(pts[2], pts[4], w, h)) / 2.0
    horiz = euclid(pts[0], pts[3], w, h)
    return vert / horiz if horiz != 0 else 0.0

def iris_center(landmarks, idxs, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idxs])
    return np.mean(pts, axis=0)

def gaze_normalized(landmarks, w, h):
    l_iris = iris_center(landmarks, IRIS_LEFT_IDX, w, h)
    r_iris = iris_center(landmarks, IRIS_RIGHT_IDX, w, h)

    l_left, l_right = landmarks[33].x * w, landmarks[133].x * w
    r_left, r_right = landmarks[362].x * w, landmarks[263].x * w

    l_norm = (l_iris[0] - l_left) / (l_right - l_left + 1e-6)
    r_norm = (r_iris[0] - r_left) / (r_right - r_left + 1e-6)
    x = ((l_norm + r_norm) / 2.0 - 0.5) * 2.0

    l_top, l_bottom = min([landmarks[i].y for i in LEFT_EYE_IDX]) * h, max([landmarks[i].y for i in LEFT_EYE_IDX]) * h
    r_top, r_bottom = min([landmarks[i].y for i in RIGHT_EYE_IDX]) * h, max([landmarks[i].y for i in RIGHT_EYE_IDX]) * h

    l_v = (l_iris[1] - l_top) / (l_bottom - l_top + 1e-6)
    r_v = (r_iris[1] - r_top) / (r_bottom - r_top + 1e-6)
    y = ((l_v + r_v) / 2.0 - 0.5) * 2.0

    return float(x), float(y)

def head_pose_angles(landmarks, w, h):
    image_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in PNP_IDX], dtype=np.float64)
    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1]
    ], dtype=np.float64)

    focal_length, center = w, (w / 2, h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return 0, 0, 0

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0

    return math.degrees(y), math.degrees(x), math.degrees(z)

def emoji_and_text(score, blink_rate, away, phone, sleep):
    if sleep: return "😴", "Micro-sleep!"
    if phone: return "📱", "Head-down (phone)"
    if away: return "🙄", "Looking away"
    if blink_rate > 40: return "😵", "Very high blink rate"
    if score >= 80: return "🧠", "Focus: Excellent"
    if score >= 60: return "😌", "Focus: Good"
    if score >= 40: return "😐", "Focus: Fair"
    if score >= 20: return "🥱", "Focus: Poor"
    return "😴", "Focus: Very low"

def generate_focus_feed():
    global latest_frame, latest_metrics

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield b""
        return

    score_history = deque(maxlen=SCORE_HISTORY_LENGTH)
    blink_timestamps = deque()
    last_blink_time, eye_closed_since = 0, None
    calib_open_ears, calibrated_open_ear = [], None
    start_time, calibrating = time.time(), True
    smoothed_score = 50.0

    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "focus_percent", "blink_per_min", "gaze_x", "gaze_y", "yaw", "pitch"])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            now = time.time()

            if calibrating and now - start_time > CALIBRATION_SECONDS:
                calibrated_open_ear = np.percentile(calib_open_ears,75) if calib_open_ears else 0.28
                calibrating = False

            avg_score, blink_rate, gaze_x, gaze_y, yaw, pitch = 0, 0, 0, 0, 0, 0
            micro_sleep, phone_flag, away = False, False, True

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                ear_l, ear_r = eye_aspect_ratio(landmarks, w, h, True), eye_aspect_ratio(landmarks, w, h, False)
                ear = (ear_l + ear_r) / 2
                if calibrating: calib_open_ears.append(ear)

                closed_thresh = calibrated_open_ear * 0.55 if calibrated_open_ear else 0.20
                closed = ear < closed_thresh

                if closed and eye_closed_since is None:
                    eye_closed_since = now
                elif not closed and eye_closed_since:
                    if now - eye_closed_since > 0.03 and (now - last_blink_time) > BLINK_MIN_INTERVAL:
                        blink_timestamps.append(now)
                        last_blink_time = now
                    eye_closed_since = None

                while blink_timestamps and now - blink_timestamps[0] > 60:
                    blink_timestamps.popleft()

                blink_rate = len(blink_timestamps)
                if eye_closed_since and (now - eye_closed_since) >= MICROSLEEP_DURATION:
                    micro_sleep = True

                gaze_x, gaze_y = gaze_normalized(landmarks, w, h)
                away = abs(gaze_x) > GAZE_X_THRESHOLD or abs(gaze_y) > GAZE_Y_THRESHOLD
                yaw, pitch, _ = head_pose_angles(landmarks, w, h)
                phone_flag = (pitch > HEAD_PITCH_LIMIT + 8) and gaze_y > 0.6 or abs(yaw) > 35

                eye_score = np.clip(ear / calibrated_open_ear, 0, 1) if calibrated_open_ear else 0
                yaw_score = max(0, 1 - abs(yaw) / HEAD_YAW_LIMIT)
                pitch_score = max(0, 1 - abs(pitch) / (HEAD_PITCH_LIMIT * 1.5))
                head_score = min(1, yaw_score * 0.7 + pitch_score * 0.3)
                gaze_score = np.clip(1 - abs(gaze_x) / (GAZE_X_THRESHOLD * 2), 0, 1)

                raw_focus = WEIGHT_EYE * eye_score + WEIGHT_HEAD * head_score + WEIGHT_GAZE * gaze_score
                raw_percent = np.clip(raw_focus * 100, 0, 100)

                if micro_sleep: raw_percent = min(raw_percent, 5)
                if phone_flag: raw_percent = min(raw_percent, 35)
                if away: raw_percent *= 0.8

                smoothed_score = SMOOTHING_ALPHA * smoothed_score + (1 - SMOOTHING_ALPHA) * raw_percent
                score_history.append(smoothed_score)
                avg_score = np.mean(score_history)

                metrics = {
                    "timestamp": now,
                    "focus_percent": float(avg_score),
                    "blink_per_min": blink_rate,
                    "gaze_x": float(gaze_x),
                    "gaze_y": float(gaze_y),
                    "yaw": float(yaw),
                    "pitch": float(pitch),
                    "status": "Running"
                }

                metrics_buffer.append(metrics)
                latest_metrics = metrics

                with open(LOG_CSV, "a", newline="") as f:
                    csv.writer(f).writerow(list(metrics.values()))

            emoji, text = emoji_and_text(avg_score, blink_rate, away, phone_flag, micro_sleep)
            cv2.putText(frame, f"{emoji} {text}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Focus: {avg_score:5.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            latest_frame = buffer.tobytes()

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_frame + b"\r\n")

    except GeneratorExit:
        pass
    finally:
        cap.release()

@router.get("/live")
async def live_focus_feed():
    return StreamingResponse(generate_focus_feed(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/snapshot")
async def get_snapshot():
    if latest_frame:
        return Response(content=latest_frame, media_type="image/jpeg")
    return Response(content=b"", media_type="image/jpeg")

@router.get("/metrics")
async def get_metrics():
    return JSONResponse(content=list(metrics_buffer))

@router.get("/metrics/latest")
async def get_latest_metrics():
    return JSONResponse(content=latest_metrics)
