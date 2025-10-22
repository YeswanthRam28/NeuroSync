<<<<<<< HEAD
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import csv
import math

# -------------------- CONFIG --------------------
CALIBRATION_SECONDS = 3.0
SCORE_HISTORY_LENGTH = 100
SMOOTHING_ALPHA = 0.6        # higher = smoother / slower
MICROSLEEP_DURATION = 0.30   # seconds eyes closed -> micro-sleep
BLINK_MIN_INTERVAL = 0.15    # ignore repeats faster than this
GAZE_X_THRESHOLD = 0.35      # relative units from center -> looking away
GAZE_Y_THRESHOLD = 0.35
HEAD_YAW_LIMIT = 25.0        # degrees at which head score drops
HEAD_PITCH_LIMIT = 20.0      # degrees (downwards) considered phone/head-down
LOG_CSV = "focus_log.csv"

# weights for final focus score
WEIGHT_EYE = 0.55
WEIGHT_HEAD = 0.30
WEIGHT_GAZE = 0.15

# -------------------- mediapipe --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

# landmark groups
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
IRIS_LEFT_IDX = [468, 469, 470, 471]
IRIS_RIGHT_IDX = [473, 474, 475, 476]
# 2D points for solvePnP (mediapipe indexes)
PNP_IDX = [1, 152, 33, 263, 61, 291]  # nose tip, chin, left eye outer, right eye outer, left mouth, right mouth

# -------------------- helpers --------------------
def euclid(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)

def eye_aspect_ratio(landmarks, w, h, left=True):
    idx = LEFT_EYE_IDX if left else RIGHT_EYE_IDX
    p = [landmarks[i] for i in idx]
    vert = (euclid(p[1], p[5], w, h) + euclid(p[2], p[4], w, h)) / 2.0
    horiz = euclid(p[0], p[3], w, h)
    return (vert / horiz) if horiz != 0 else 0.0

def iris_center(landmarks, idxs, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idxs])
    return np.mean(pts, axis=0)

def eye_box(landmarks, outer_idx, inner_idx, w, h):
    # approximate eye bounding box using outer and inner corners
    left = landmarks[outer_idx].x * w
    right = landmarks[inner_idx].x * w
    top = min([landmarks[i].y for i in LEFT_EYE_IDX]) * h
    bottom = max([landmarks[i].y for i in LEFT_EYE_IDX]) * h
    return left, right, top, bottom

def gaze_normalized(landmarks, w, h):
    # compute iris center relative to eye width normalized ~[-1,1]
    # left eye
    l_iris = iris_center(landmarks, IRIS_LEFT_IDX, w, h)
    r_iris = iris_center(landmarks, IRIS_RIGHT_IDX, w, h)
    # left eye box using landmarks 33 (outer) and 133 (inner)
    l_left = landmarks[33].x * w
    l_right = landmarks[133].x * w
    r_left = landmarks[362].x * w
    r_right = landmarks[263].x * w
    # normalize positions to [0..1]
    l_norm = (l_iris[0] - l_left) / (l_right - l_left + 1e-6)
    r_norm = (r_iris[0] - r_left) / (r_right - r_left + 1e-6)
    # average, shift to center -> [-1..1]
    x = ((l_norm + r_norm) / 2.0 - 0.5) * 2.0
    # vertical: use iris y relative to eye box height
    l_top = min([landmarks[i].y for i in LEFT_EYE_IDX]) * h
    l_bottom = max([landmarks[i].y for i in LEFT_EYE_IDX]) * h
    r_top = min([landmarks[i].y for i in RIGHT_EYE_IDX]) * h
    r_bottom = max([landmarks[i].y for i in RIGHT_EYE_IDX]) * h
    l_v = (l_iris[1] - l_top) / (l_bottom - l_top + 1e-6)
    r_v = (r_iris[1] - r_top) / (r_bottom - r_top + 1e-6)
    y = ((l_v + r_v) / 2.0 - 0.5) * 2.0
    return float(x), float(y)

def head_pose_angles(landmarks, w, h, camera_matrix=None, dist_coeffs=None):
    # Build 2D-3D correspondences
    image_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in PNP_IDX], dtype=np.float64)

    # Model 3D points (approximate face model in mm)
    model_points = np.array([
        [0.0, 0.0, 0.0],        # nose tip
        [0.0, -63.6, -12.5],    # chin
        [-43.3, 32.7, -26.0],   # left eye outer
        [43.3, 32.7, -26.0],    # right eye outer
        [-28.9, -28.9, -24.1],  # left mouth corner
        [28.9, -28.9, -24.1],   # right mouth corner
    ], dtype=np.float64)

    if camera_matrix is None:
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0
    # convert rotation vector to euler angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0
    # convert to degrees: x = pitch, y = yaw, z = roll
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)
    return yaw, pitch, roll

def emoji_and_text(avg_score, blink_per_min, looking_away, phone_flag, micro_sleep):
    if micro_sleep:
        return "😴", "Micro-sleep!"
    if phone_flag:
        return "📱", "Head-down (phone)"
    if looking_away:
        return "🙄", "Looking away"
    if blink_per_min > 40:
        return "😵", "Very high blink rate"
    if avg_score >= 80:
        return "🧠", "Focus: Excellent"
    if avg_score >= 60:
        return "😌", "Focus: Good"
    if avg_score >= 40:
        return "😐", "Focus: Fair"
    if avg_score >= 20:
        return "🥱", "Focus: Poor"
    return "😴", "Focus: Very low"

# -------------------- logging init --------------------
with open(LOG_CSV, "w", newline="") as f:
    csv.writer(f).writerow(["timestamp", "focus_percent", "blink_per_min", "gaze_x", "gaze_y", "yaw", "pitch"])

# -------------------- MAIN --------------------
cap = cv2.VideoCapture(0)
score_history = deque(maxlen=SCORE_HISTORY_LENGTH)
blink_timestamps = deque()
last_blink_time = 0
eye_closed_since = None
calib_open_ears = []
calibrated_open_ear = None

start_time = time.time()
calibrating = True
print(f"Auto-calibrating for {CALIBRATION_SECONDS} seconds — please look directly at camera with neutral expression...")

# run loop
smoothed_score = 50.0  # percent
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    now = time.time()
    # calibration window
    if calibrating and now - start_time > CALIBRATION_SECONDS:
        # finalize calibration
        if calib_open_ears:
            calibrated_open_ear = max(0.0001, float(np.median(calib_open_ears)))
        else:
            calibrated_open_ear = 0.28  # fallback
        print(f"Calibration done. open EAR baseline = {calibrated_open_ear:.3f}")
        calibrating = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # EARs
        ear_l = eye_aspect_ratio(landmarks, w, h, left=True)
        ear_r = eye_aspect_ratio(landmarks, w, h, left=False)
        ear = (ear_l + ear_r) / 2.0

        # During calibration record open EARs
        if calibrating:
            calib_open_ears.append(ear)

        # define closed EAR threshold dynamically relative to baseline
        if calibrated_open_ear:
            closed_ear_thresh = calibrated_open_ear * 0.55  # tweakable
        else:
            closed_ear_thresh = 0.20

        # blink detection (transition open->closed->open)
        is_closed = ear < closed_ear_thresh
        if is_closed:
            if eye_closed_since is None:
                eye_closed_since = now
        else:
            if eye_closed_since is not None:
                duration = now - eye_closed_since
                # register blink if short
                if duration >= 0.03 and (now - last_blink_time) > BLINK_MIN_INTERVAL:
                    blink_timestamps.append(now)
                    last_blink_time = now
                eye_closed_since = None

        # clean blink history older than 60 seconds
        while blink_timestamps and now - blink_timestamps[0] > 60.0:
            blink_timestamps.popleft()
        blink_per_min = len(blink_timestamps)

        # micro-sleep detection
        micro_sleep = False
        if eye_closed_since is not None and (now - eye_closed_since) >= MICROSLEEP_DURATION:
            micro_sleep = True

        # gaze normalized (-1..1)
        gaze_x, gaze_y = gaze_normalized(landmarks, w, h)
        looking_away = abs(gaze_x) > GAZE_X_THRESHOLD or abs(gaze_y) > GAZE_Y_THRESHOLD

        # head pose
        yaw, pitch, roll = head_pose_angles(landmarks, w, h)
        phone_flag = pitch > HEAD_PITCH_LIMIT or abs(roll) > 30.0

        # scoring components, each 0..1
        # eye_score: normalized from closed->open relative to calibrated open ear
        if calibrated_open_ear:
            eye_score = np.clip((ear / calibrated_open_ear), 0.0, 1.0)
        else:
            eye_score = np.clip((ear - 0.12) / 0.18, 0.0, 1.0)

        # head_score based on yaw & pitch (punish large yaw/pitch)
        yaw_score = max(0.0, 1.0 - (abs(yaw) / HEAD_YAW_LIMIT))
        pitch_score = max(0.0, 1.0 - (abs(pitch) / (HEAD_PITCH_LIMIT * 1.5)))
        head_score = min(1.0, (yaw_score * 0.7 + pitch_score * 0.3))

        # gaze_score penalizes looking away
        gaze_score = max(0.0, 1.0 - (abs(gaze_x) / (GAZE_X_THRESHOLD * 2.0)))
        gaze_score = np.clip(gaze_score, 0.0, 1.0)

        # combine
        raw_focus = (WEIGHT_EYE * eye_score) + (WEIGHT_HEAD * head_score) + (WEIGHT_GAZE * gaze_score)
        raw_percent = float(np.clip(raw_focus * 100.0, 0.0, 100.0))

        # if micro-sleep or phone_flag or looking_away massively penalize
        if micro_sleep:
            raw_percent = min(raw_percent, 5.0)
        if phone_flag:
            raw_percent = min(raw_percent, 35.0)
        if looking_away:
            raw_percent = min(raw_percent, raw_percent * 0.8)

        # smoothing
        smoothed_score = SMOOTHING_ALPHA * smoothed_score + (1.0 - SMOOTHING_ALPHA) * raw_percent
        score_history.append(smoothed_score)
        avg_score = float(np.mean(score_history))

        # draw debug points
        for idx in [33, 133, 362, 263, 1] + IRIS_LEFT_IDX + IRIS_RIGHT_IDX:
            pt = landmarks[idx]
            cx, cy = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 200, 200), -1)

        # logging
        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([time.time(), f"{avg_score:.2f}", blink_per_min, f"{gaze_x:.3f}", f"{gaze_y:.3f}", f"{yaw:.2f}", f"{pitch:.2f}"])

    else:
        avg_score = 0.0
        blink_per_min = 0
        gaze_x = gaze_y = 0.0
        yaw = pitch = roll = 0.0
        micro_sleep = False
        phone_flag = False
        looking_away = True

    # overlay text and emoji
    emoji, text = emoji_and_text(avg_score, blink_per_min, looking_away, phone_flag, micro_sleep)
    cv2.putText(frame, f"{emoji} {text}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Focus: {avg_score:5.1f}%", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
    cv2.putText(frame, f"Blinks/min: {blink_per_min}", (18, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 1)
    cv2.putText(frame, f"Gaze(x,y): {gaze_x:+.2f},{gaze_y:+.2f}", (18, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Yaw:{yaw:.1f} Pitch:{pitch:.1f}", (18, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # calibration overlay
    if calibrating:
        sec_left = max(0.0, CALIBRATION_SECONDS - (now - start_time))
        cv2.putText(frame, f"Calibrating... hold steady ({sec_left:.1f}s)", (18, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

    cv2.imshow("NeuroSync Accurate Focus Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
=======
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import csv
import math

# -------------------- CONFIG --------------------
CALIBRATION_SECONDS = 3.0
SCORE_HISTORY_LENGTH = 100
SMOOTHING_ALPHA = 0.6        # higher = smoother / slower
MICROSLEEP_DURATION = 0.30   # seconds eyes closed -> micro-sleep
BLINK_MIN_INTERVAL = 0.15    # ignore repeats faster than this
GAZE_X_THRESHOLD = 0.35      # relative units from center -> looking away
GAZE_Y_THRESHOLD = 0.35
HEAD_YAW_LIMIT = 25.0        # degrees at which head score drops
HEAD_PITCH_LIMIT = 20.0      # degrees (downwards) considered phone/head-down
LOG_CSV = "focus_log.csv"

# weights for final focus score
WEIGHT_EYE = 0.55
WEIGHT_HEAD = 0.30
WEIGHT_GAZE = 0.15

# -------------------- mediapipe --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

# landmark groups
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
IRIS_LEFT_IDX = [468, 469, 470, 471]
IRIS_RIGHT_IDX = [473, 474, 475, 476]
# 2D points for solvePnP (mediapipe indexes)
PNP_IDX = [1, 152, 33, 263, 61, 291]  # nose tip, chin, left eye outer, right eye outer, left mouth, right mouth

# -------------------- helpers --------------------
def euclid(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)

def eye_aspect_ratio(landmarks, w, h, left=True):
    idx = LEFT_EYE_IDX if left else RIGHT_EYE_IDX
    p = [landmarks[i] for i in idx]
    vert = (euclid(p[1], p[5], w, h) + euclid(p[2], p[4], w, h)) / 2.0
    horiz = euclid(p[0], p[3], w, h)
    return (vert / horiz) if horiz != 0 else 0.0

def iris_center(landmarks, idxs, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idxs])
    return np.mean(pts, axis=0)

def eye_box(landmarks, outer_idx, inner_idx, w, h):
    # approximate eye bounding box using outer and inner corners
    left = landmarks[outer_idx].x * w
    right = landmarks[inner_idx].x * w
    top = min([landmarks[i].y for i in LEFT_EYE_IDX]) * h
    bottom = max([landmarks[i].y for i in LEFT_EYE_IDX]) * h
    return left, right, top, bottom

def gaze_normalized(landmarks, w, h):
    # compute iris center relative to eye width normalized ~[-1,1]
    # left eye
    l_iris = iris_center(landmarks, IRIS_LEFT_IDX, w, h)
    r_iris = iris_center(landmarks, IRIS_RIGHT_IDX, w, h)
    # left eye box using landmarks 33 (outer) and 133 (inner)
    l_left = landmarks[33].x * w
    l_right = landmarks[133].x * w
    r_left = landmarks[362].x * w
    r_right = landmarks[263].x * w
    # normalize positions to [0..1]
    l_norm = (l_iris[0] - l_left) / (l_right - l_left + 1e-6)
    r_norm = (r_iris[0] - r_left) / (r_right - r_left + 1e-6)
    # average, shift to center -> [-1..1]
    x = ((l_norm + r_norm) / 2.0 - 0.5) * 2.0
    # vertical: use iris y relative to eye box height
    l_top = min([landmarks[i].y for i in LEFT_EYE_IDX]) * h
    l_bottom = max([landmarks[i].y for i in LEFT_EYE_IDX]) * h
    r_top = min([landmarks[i].y for i in RIGHT_EYE_IDX]) * h
    r_bottom = max([landmarks[i].y for i in RIGHT_EYE_IDX]) * h
    l_v = (l_iris[1] - l_top) / (l_bottom - l_top + 1e-6)
    r_v = (r_iris[1] - r_top) / (r_bottom - r_top + 1e-6)
    y = ((l_v + r_v) / 2.0 - 0.5) * 2.0
    return float(x), float(y)

def head_pose_angles(landmarks, w, h, camera_matrix=None, dist_coeffs=None):
    # Build 2D-3D correspondences
    image_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in PNP_IDX], dtype=np.float64)

    # Model 3D points (approximate face model in mm)
    model_points = np.array([
        [0.0, 0.0, 0.0],        # nose tip
        [0.0, -63.6, -12.5],    # chin
        [-43.3, 32.7, -26.0],   # left eye outer
        [43.3, 32.7, -26.0],    # right eye outer
        [-28.9, -28.9, -24.1],  # left mouth corner
        [28.9, -28.9, -24.1],   # right mouth corner
    ], dtype=np.float64)

    if camera_matrix is None:
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0
    # convert rotation vector to euler angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0
    # convert to degrees: x = pitch, y = yaw, z = roll
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)
    return yaw, pitch, roll

def emoji_and_text(avg_score, blink_per_min, looking_away, phone_flag, micro_sleep):
    if micro_sleep:
        return "😴", "Micro-sleep!"
    if phone_flag:
        return "📱", "Head-down (phone)"
    if looking_away:
        return "🙄", "Looking away"
    if blink_per_min > 40:
        return "😵", "Very high blink rate"
    if avg_score >= 80:
        return "🧠", "Focus: Excellent"
    if avg_score >= 60:
        return "😌", "Focus: Good"
    if avg_score >= 40:
        return "😐", "Focus: Fair"
    if avg_score >= 20:
        return "🥱", "Focus: Poor"
    return "😴", "Focus: Very low"

# -------------------- logging init --------------------
with open(LOG_CSV, "w", newline="") as f:
    csv.writer(f).writerow(["timestamp", "focus_percent", "blink_per_min", "gaze_x", "gaze_y", "yaw", "pitch"])

# -------------------- MAIN --------------------
cap = cv2.VideoCapture(0)
score_history = deque(maxlen=SCORE_HISTORY_LENGTH)
blink_timestamps = deque()
last_blink_time = 0
eye_closed_since = None
calib_open_ears = []
calibrated_open_ear = None

start_time = time.time()
calibrating = True
print(f"Auto-calibrating for {CALIBRATION_SECONDS} seconds — please look directly at camera with neutral expression...")

# run loop
smoothed_score = 50.0  # percent
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    now = time.time()
    # calibration window
    if calibrating and now - start_time > CALIBRATION_SECONDS:
        # finalize calibration
        if calib_open_ears:
            calibrated_open_ear = max(0.0001, float(np.median(calib_open_ears)))
        else:
            calibrated_open_ear = 0.28  # fallback
        print(f"Calibration done. open EAR baseline = {calibrated_open_ear:.3f}")
        calibrating = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # EARs
        ear_l = eye_aspect_ratio(landmarks, w, h, left=True)
        ear_r = eye_aspect_ratio(landmarks, w, h, left=False)
        ear = (ear_l + ear_r) / 2.0

        # During calibration record open EARs
        if calibrating:
            calib_open_ears.append(ear)

        # define closed EAR threshold dynamically relative to baseline
        if calibrated_open_ear:
            closed_ear_thresh = calibrated_open_ear * 0.55  # tweakable
        else:
            closed_ear_thresh = 0.20

        # blink detection (transition open->closed->open)
        is_closed = ear < closed_ear_thresh
        if is_closed:
            if eye_closed_since is None:
                eye_closed_since = now
        else:
            if eye_closed_since is not None:
                duration = now - eye_closed_since
                # register blink if short
                if duration >= 0.03 and (now - last_blink_time) > BLINK_MIN_INTERVAL:
                    blink_timestamps.append(now)
                    last_blink_time = now
                eye_closed_since = None

        # clean blink history older than 60 seconds
        while blink_timestamps and now - blink_timestamps[0] > 60.0:
            blink_timestamps.popleft()
        blink_per_min = len(blink_timestamps)

        # micro-sleep detection
        micro_sleep = False
        if eye_closed_since is not None and (now - eye_closed_since) >= MICROSLEEP_DURATION:
            micro_sleep = True

        # gaze normalized (-1..1)
        gaze_x, gaze_y = gaze_normalized(landmarks, w, h)
        looking_away = abs(gaze_x) > GAZE_X_THRESHOLD or abs(gaze_y) > GAZE_Y_THRESHOLD

        # head pose
        yaw, pitch, roll = head_pose_angles(landmarks, w, h)
        phone_flag = pitch > HEAD_PITCH_LIMIT or abs(roll) > 30.0

        # scoring components, each 0..1
        # eye_score: normalized from closed->open relative to calibrated open ear
        if calibrated_open_ear:
            eye_score = np.clip((ear / calibrated_open_ear), 0.0, 1.0)
        else:
            eye_score = np.clip((ear - 0.12) / 0.18, 0.0, 1.0)

        # head_score based on yaw & pitch (punish large yaw/pitch)
        yaw_score = max(0.0, 1.0 - (abs(yaw) / HEAD_YAW_LIMIT))
        pitch_score = max(0.0, 1.0 - (abs(pitch) / (HEAD_PITCH_LIMIT * 1.5)))
        head_score = min(1.0, (yaw_score * 0.7 + pitch_score * 0.3))

        # gaze_score penalizes looking away
        gaze_score = max(0.0, 1.0 - (abs(gaze_x) / (GAZE_X_THRESHOLD * 2.0)))
        gaze_score = np.clip(gaze_score, 0.0, 1.0)

        # combine
        raw_focus = (WEIGHT_EYE * eye_score) + (WEIGHT_HEAD * head_score) + (WEIGHT_GAZE * gaze_score)
        raw_percent = float(np.clip(raw_focus * 100.0, 0.0, 100.0))

        # if micro-sleep or phone_flag or looking_away massively penalize
        if micro_sleep:
            raw_percent = min(raw_percent, 5.0)
        if phone_flag:
            raw_percent = min(raw_percent, 35.0)
        if looking_away:
            raw_percent = min(raw_percent, raw_percent * 0.8)

        # smoothing
        smoothed_score = SMOOTHING_ALPHA * smoothed_score + (1.0 - SMOOTHING_ALPHA) * raw_percent
        score_history.append(smoothed_score)
        avg_score = float(np.mean(score_history))

        # draw debug points
        for idx in [33, 133, 362, 263, 1] + IRIS_LEFT_IDX + IRIS_RIGHT_IDX:
            pt = landmarks[idx]
            cx, cy = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 200, 200), -1)

        # logging
        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([time.time(), f"{avg_score:.2f}", blink_per_min, f"{gaze_x:.3f}", f"{gaze_y:.3f}", f"{yaw:.2f}", f"{pitch:.2f}"])

    else:
        avg_score = 0.0
        blink_per_min = 0
        gaze_x = gaze_y = 0.0
        yaw = pitch = roll = 0.0
        micro_sleep = False
        phone_flag = False
        looking_away = True

    # overlay text and emoji
    emoji, text = emoji_and_text(avg_score, blink_per_min, looking_away, phone_flag, micro_sleep)
    cv2.putText(frame, f"{emoji} {text}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Focus: {avg_score:5.1f}%", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
    cv2.putText(frame, f"Blinks/min: {blink_per_min}", (18, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 1)
    cv2.putText(frame, f"Gaze(x,y): {gaze_x:+.2f},{gaze_y:+.2f}", (18, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Yaw:{yaw:.1f} Pitch:{pitch:.1f}", (18, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # calibration overlay
    if calibrating:
        sec_left = max(0.0, CALIBRATION_SECONDS - (now - start_time))
        cv2.putText(frame, f"Calibrating... hold steady ({sec_left:.1f}s)", (18, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

    cv2.imshow("NeuroSync Accurate Focus Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
>>>>>>> ccdce86647545da532a9c5a3730488c2e88c7dd7
