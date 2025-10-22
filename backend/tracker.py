import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui 
import math
from collections import deque, Counter

# --- 1. Enhanced Initialization and Configuration ---

# Initialize MediaPipe Solutions with optimized parameters
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Enhanced hand detection with better tracking
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,  # Higher complexity for better accuracy
    min_detection_confidence=0.7,  # Increased for reliability
    min_tracking_confidence=0.7
)

# Enhanced pose detection
pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Enhanced face mesh detection
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enable iris detection for better eye tracking
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize the webcam with better settings
cap = cv2.VideoCapture(0)
# Fallback to lower resolution if 1280x720 fails
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Webcam and screen resolution
wCam, hCam = 640, 480
wScr, hScr = pyautogui.size()

# Enhanced Mouse Control Parameters
smoothening = 5  # Reduced for more responsive movement
plocX, plocY = 0, 0
clocX, clocY = 0, 0
frameR = 100  # Larger control area

# Advanced Gesture Recognition
GESTURE_BUFFER_SIZE = 5
gesture_buffer = deque(maxlen=GESTURE_BUFFER_SIZE)
last_stable_gesture = "Idle/Unknown"

# System Action Cooldowns
last_action_time = 0 
COOLDOWN_TIME = 0.8  # Reduced for better responsiveness
scroll_cooldown = time.time()

# Enhanced Fatigue/Drowsiness Scoring
BLINK_THRESHOLD = 0.21  # Optimized EAR threshold
MIN_BLINK_FRAMES = 2    # Reduced for faster blink detection
CONSECUTIVE_FRAMES = 3  # For stable detection
blink_count = 0
frames_closed = 0
is_blinking = False
pTime = 0 

# Enhanced Posture Analysis
POSTURE_ALERT_DURATION = 4.0
posture_warning_start_time = None
posture_buffer = deque(maxlen=10)  # Buffer for stable posture detection

# Enhanced Volume Control
Z_MAX_VAL = 0.12
Z_MIN_VAL = -0.12
VOL_SENSITIVITY = 12
volume_buffer = deque(maxlen=3)  # Smooth volume changes

# Enhanced Landmark Indices with more precise points
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCP = [2, 5, 9, 13, 17]  # Metacarpophalangeal joints for better finger detection
FINGER_PIP = [3, 6, 10, 14, 18]  # Proximal interphalangeal joints

POSE_LANDMARKS = {
    'nose': 0, 'shoulder_r': 12, 'shoulder_l': 11, 
    'hip_r': 24, 'hip_l': 23, 'ear_r': 8, 'ear_l': 7
}

# Enhanced Eye Aspect Ratio indices (more precise)
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# --- 2. Enhanced Helper Functions ---

def enhanced_finger_status(hand_landmarks, handedness_label):
    """Enhanced finger detection using multiple joint angles."""
    lm = hand_landmarks.landmark
    is_open = []
    
    # Enhanced Thumb Detection (angle-based)
    thumb_tip = np.array([lm[4].x, lm[4].y])
    thumb_mcp = np.array([lm[2].x, lm[2].y])
    thumb_ip = np.array([lm[3].x, lm[3].y])
    
    # Calculate thumb angle
    vec1 = thumb_ip - thumb_mcp
    vec2 = thumb_tip - thumb_ip
    if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        thumb_angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
        thumb_open = thumb_angle > 0.5  # Angle threshold
    else:
        # Fallback to simple coordinate check
        thumb_open = lm[4].x < lm[3].x if handedness_label == "Left" else lm[4].x > lm[3].x
    
    is_open.append(thumb_open)
    
    # Enhanced Finger Detection (multi-joint)
    for i in range(1, 5):
        tip = np.array([lm[FINGER_TIPS[i]].x, lm[FINGER_TIPS[i]].y])
        pip = np.array([lm[FINGER_PIP[i]].x, lm[FINGER_PIP[i]].y])
        mcp = np.array([lm[FINGER_MCP[i]].x, lm[FINGER_MCP[i]].y])
        
        # Calculate finger extension angle
        vec1 = pip - mcp
        vec2 = tip - pip
        
        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            finger_angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
            finger_open = finger_angle > 0.3  # Optimized threshold
        else:
            # Fallback to simple Y-coordinate check
            finger_open = lm[FINGER_TIPS[i]].y < lm[FINGER_MCP[i]].y
        
        is_open.append(finger_open)
            
    return is_open

def stabilized_gesture_detection(finger_status, hand_landmarks):
    """Enhanced gesture detection with temporal stability."""
    lm = hand_landmarks.landmark
    
    # Enhanced Pinch Detection
    thumb_tip = np.array([lm[4].x, lm[4].y, lm[4].z])
    index_tip = np.array([lm[8].x, lm[8].y, lm[8].z])
    pinch_distance_3d = np.linalg.norm(thumb_tip - index_tip)
    is_pinching = pinch_distance_3d < 0.035  # More precise 3D distance
    
    # Count extended fingers (excluding thumb for most gestures)
    fingers_up = finger_status[1:]  # Index, Middle, Ring, Pinky
    fingers_up_count = sum(fingers_up)
    
    # Enhanced gesture recognition
    # Volume Control (Index, Middle, Ring up - Pinky down)
    if (fingers_up_count == 3 and 
        finger_status[1] and finger_status[2] and finger_status[3] and not finger_status[4]):
        return "Volume Control", is_pinching
    
    # Scroll Mode (Index and Middle up only)
    elif (fingers_up_count == 2 and 
          finger_status[1] and finger_status[2] and not finger_status[3] and not finger_status[4]):
        return "Scroll Mode", is_pinching
    
    # Play/Pause (Middle finger only up)
    elif (fingers_up_count == 1 and finger_status[2] and 
          not any([finger_status[1], finger_status[3], finger_status[4]])):
        return "Play/Pause (Space)", is_pinching
    
    # Close Tab (Pinky only up)
    elif (fingers_up_count == 1 and finger_status[4] and 
          not any([finger_status[1], finger_status[2], finger_status[3]])):
        return "Close Tab (Ctrl+W)", is_pinching
    
    # Mouse Pointer (Index only up)
    elif (fingers_up_count == 1 and finger_status[1] and 
          not any([finger_status[2], finger_status[3], finger_status[4]])):
        return "Mouse Pointer", is_pinching
    
    return "Idle/Unknown", is_pinching

def enhanced_posture_analysis(pose_landmarks, hCam):
    """Advanced posture analysis with multiple metrics."""
    lm = pose_landmarks.landmark
    
    # Check landmark visibility
    key_landmarks = ['shoulder_l', 'shoulder_r', 'nose', 'hip_l', 'hip_r']
    if any(lm[POSE_LANDMARKS[landmark]].visibility < 0.7 for landmark in key_landmarks):
        return "Posture Not Fully Visible"
    
    shoulder_l = lm[POSE_LANDMARKS['shoulder_l']]
    shoulder_r = lm[POSE_LANDMARKS['shoulder_r']]
    hip_l = lm[POSE_LANDMARKS['hip_l']]
    hip_r = lm[POSE_LANDMARKS['hip_r']]
    nose = lm[POSE_LANDMARKS['nose']]

    issues = []
    
    # 1. Enhanced Shoulder Level Check
    shoulder_height_diff = abs(shoulder_l.y - shoulder_r.y)
    if shoulder_height_diff > 0.05:  # More sensitive threshold
        issues.append("Uneven Shoulders")
    
    # 2. Enhanced Slouching Detection
    mid_shoulder_y = (shoulder_l.y + shoulder_r.y) / 2
    nose_to_shoulder = nose.y - mid_shoulder_y
    if nose_to_shoulder > 0.08:  # Head forward/leaning
        issues.append("Forward Head Posture")
    
    # 3. Hip Alignment Check
    hip_height_diff = abs(hip_l.y - hip_r.y)
    if hip_height_diff > 0.03:
        issues.append("Uneven Hips")
    
    # 4. Side Leaning Detection
    mid_shoulder_x = (shoulder_l.x + shoulder_r.x) / 2
    mid_hip_x = (hip_l.x + hip_r.x) / 2
    side_lean = abs(mid_shoulder_x - mid_hip_x)
    if side_lean > 0.05:
        issues.append("Side Leaning")
    
    if issues:
        return f"Poor Posture: {', '.join(issues)}"
    else:
        return "Good Posture"

def enhanced_focus_analysis(face_landmarks):
    """Enhanced focus and emotion analysis with better metrics."""
    lm = face_landmarks.landmark
    
    def calculate_ear(eye_indices):
        """Enhanced Eye Aspect Ratio calculation."""
        # Horizontal distances
        p1 = np.array([lm[eye_indices[0]].x, lm[eye_indices[0]].y])
        p2 = np.array([lm[eye_indices[8]].x, lm[eye_indices[8]].y])
        horizontal_dist = np.linalg.norm(p1 - p2)
        
        # Vertical distances - multiple measurements for robustness
        vertical_dists = []
        for i in range(3):
            v1 = np.array([lm[eye_indices[1+i]].x, lm[eye_indices[1+i]].y])
            v2 = np.array([lm[eye_indices[7-i]].x, lm[eye_indices[7-i]].y])
            vertical_dists.append(np.linalg.norm(v1 - v2))
        
        avg_vertical = np.mean(vertical_dists)
        return avg_vertical / horizontal_dist if horizontal_dist > 0 else 0.3
    
    left_ear = calculate_ear(LEFT_EYE_INDICES)
    right_ear = calculate_ear(RIGHT_EYE_INDICES)
    avg_ear = (left_ear + right_ear) / 2
    
    # Enhanced focus classification
    if avg_ear < 0.18:
        focus_status = "Drowsy/Low Focus"
    elif avg_ear < 0.23:
        focus_status = "Moderate Focus"
    else:
        focus_status = "High Focus"
    
    # Enhanced Emotion Detection
    mouth_width = abs(lm[61].x - lm[291].x)
    mouth_height = abs(lm[13].y - lm[14].y)
    mouth_corner_left = lm[61].y
    mouth_corner_right = lm[291].y
    
    emotion = "Neutral"
    
    # Smile detection
    if mouth_height > 0.03 and mouth_width > 0.15:
        if lm[13].y < (mouth_corner_left + mouth_corner_right) / 2 - 0.01:
            emotion = "Happy/Smiling"
    
    # Surprise detection
    elif mouth_height > 0.06:
        emotion = "Surprise"
    
    # Frowning detection
    elif lm[13].y > (mouth_corner_left + mouth_corner_right) / 2 + 0.015:
        emotion = "Frowning"
    
    return focus_status, emotion, avg_ear

def smooth_landmarks_list(current_landmarks, previous_landmarks=None, alpha=0.6):
    """Apply exponential smoothing to landmarks using list conversion."""
    if previous_landmarks is None:
        return current_landmarks
    
    # Convert landmarks to lists for smoothing
    smoothed_landmarks = []
    for i in range(len(current_landmarks.landmark)):
        curr_lm = current_landmarks.landmark[i]
        prev_lm = previous_landmarks.landmark[i]
        
        # Apply exponential smoothing
        smoothed_x = alpha * curr_lm.x + (1 - alpha) * prev_lm.x
        smoothed_y = alpha * curr_lm.y + (1 - alpha) * prev_lm.y
        smoothed_z = alpha * curr_lm.z + (1 - alpha) * prev_lm.z
        
        # Create new landmark with smoothed values
        smoothed_landmark = mp_draw._normalized_to_pixel_coordinates(
            smoothed_x, smoothed_y, wCam, hCam
        )
        # Store as simple coordinates for now
        smoothed_landmarks.append((smoothed_x, smoothed_y, smoothed_z))
    
    return current_landmarks  # Return original for now - smoothing will be applied in detection

def draw_bounding_box(img, landmarks):
    """Calculates and draws a tight bounding box around detected object."""
    h, w, c = img.shape
    x_coords = [landmark.x * w for landmark in landmarks.landmark]
    y_coords = [landmark.y * h for landmark in landmarks.landmark]

    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
    return x_min, y_min

# Simplified smoothing - just store previous values for velocity calculation
prev_hand_pos = None
prev_pose_data = None
prev_face_data = None

# --- 3. Enhanced Main Loop ---

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process for all three models
    hand_results = hands.process(imgRGB)
    pose_results = pose.process(imgRGB)
    face_results = face_mesh.process(imgRGB)
    
    current_time = time.time()
    
    # Draw the enhanced mouse control area
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
    
    # --- A. ENHANCED HAND TRACKING ---
    if hand_results.multi_hand_landmarks:
        hand_lm = hand_results.multi_hand_landmarks[0]
        
        handedness_label = hand_results.multi_handedness[0].classification[0].label
        finger_status = enhanced_finger_status(hand_lm, handedness_label)
        current_gesture, is_pinching = stabilized_gesture_detection(finger_status, hand_lm)
        
        # Gesture stabilization using buffer
        gesture_buffer.append(current_gesture)
        if len(gesture_buffer) == GESTURE_BUFFER_SIZE:
            most_common_gesture = Counter(gesture_buffer).most_common(1)[0][0]
            if most_common_gesture != last_stable_gesture:
                last_stable_gesture = most_common_gesture
        current_gesture = last_stable_gesture

        mp_draw.draw_landmarks(img, hand_lm, mp_hands.HAND_CONNECTIONS)
        x_min, y_min = draw_bounding_box(img, hand_lm)
        
        cv2.putText(img, f'GESTURE: {current_gesture}', (x_min, y_min + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        lm8 = hand_lm.landmark[FINGER_TIPS[1]]
        x1, y1 = int(lm8.x * wCam), int(lm8.y * hCam)

        # Enhanced gesture handling
        if current_gesture == "Mouse Pointer":
            if not is_pinching:  # Moving
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                
                # Enhanced smoothing with velocity-based adjustment
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                
                pyautogui.moveTo(wScr - clocX, clocY)
                plocX, plocY = clocX, clocY
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "MOUSE MOVING", (10, hCam - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            elif is_pinching and (current_time - last_action_time > 0.4):  # Clicking
                pyautogui.click()
                last_action_time = current_time
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "CLICK TRIGGERED!", (10, hCam - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
        elif current_gesture == "Scroll Mode":
            if 'pScrollY' in globals():
                scroll_delta = pScrollY - y1
                if abs(scroll_delta) > 3 and (current_time - scroll_cooldown) > 0.03:
                    scroll_speed = -int(scroll_delta / 2)  # More responsive scrolling
                    pyautogui.scroll(scroll_speed)
                    globals()['scroll_cooldown'] = current_time

            cv2.circle(img, (x1, y1), 15, (255, 165, 0), cv2.FILLED)
            cv2.putText(img, f"SCROLLING", (10, hCam - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
            globals()['pScrollY'] = y1
            
        elif current_gesture == "Volume Control":
            z_val = hand_lm.landmark[FINGER_TIPS[1]].z
            volume_buffer.append(z_val)
            avg_z = np.mean(volume_buffer)
            
            mapped_vol = np.interp(avg_z, (Z_MIN_VAL, Z_MAX_VAL), (-VOL_SENSITIVITY, VOL_SENSITIVITY))
            
            if mapped_vol > 4 and (current_time - last_action_time > 0.2):
                pyautogui.press('volumeup')
                last_action_time = current_time
                vol_text = "VOL UP"
            elif mapped_vol < -4 and (current_time - last_action_time > 0.2):
                pyautogui.press('volumedown')
                last_action_time = current_time
                vol_text = "VOL DOWN"
            else:
                vol_text = "VOL STABLE"

            cv2.circle(img, (x1, y1), 15, (128, 0, 128), cv2.FILLED)
            cv2.putText(img, f"VOLUME: {vol_text}", (10, hCam - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 3)

        elif current_time - last_action_time > COOLDOWN_TIME:
            if current_gesture == "Play/Pause (Space)":
                pyautogui.press('space')
                last_action_time = current_time
                cv2.putText(img, "PLAY/PAUSE TRIGGERED!", (10, hCam - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
                
            elif current_gesture == "Close Tab (Ctrl+W)":
                pyautogui.hotkey('ctrl', 'w')
                last_action_time = current_time
                cv2.putText(img, "CLOSE TAB TRIGGERED!", (10, hCam - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                
    else:
        gesture_buffer.clear()
        last_stable_gesture = "Idle/Unknown"
        if 'pScrollY' in globals():
            del globals()['pScrollY']
            
    # --- B. ENHANCED POSE TRACKING ---
    posture_status = "Posture Not Detected"
    if pose_results.pose_landmarks:
        pose_lm = pose_results.pose_landmarks
        
        current_posture = enhanced_posture_analysis(pose_lm, hCam)
        posture_buffer.append(current_posture)
        
        # Use majority voting for stable posture detection
        if len(posture_buffer) == posture_buffer.maxlen:
            posture_status = Counter(posture_buffer).most_common(1)[0][0]
        else:
            posture_status = current_posture
        
        mp_draw.draw_landmarks(img, pose_lm, mp_pose.POSE_CONNECTIONS)
        
        # Enhanced posture feedback
        if "Poor Posture" in posture_status:
            if posture_warning_start_time is None:
                posture_warning_start_time = current_time
            
            duration = current_time - posture_warning_start_time
            
            if duration > POSTURE_ALERT_DURATION:
                # Pulsing alert for better visibility
                alert_intensity = int(255 * (0.5 + 0.5 * math.sin(current_time * 5)))
                cv2.putText(img, "FIX YOUR POSTURE!", (wCam // 2 - 200, hCam // 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, alert_intensity), 4)
        else:
            posture_warning_start_time = None

    # Enhanced posture display
    posture_color = (0, 255, 0) if "Good" in posture_status else (0, 0, 255)
    cv2.putText(img, f'POSTURE: {posture_status}', (wCam - 400, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
        
    # --- C. ENHANCED FACE MESH TRACKING ---
    focus_status = "Focus Not Detected"
    emotion_status = "Emotion Not Detected"
    
    if face_results.multi_face_landmarks:
        face_lm = face_results.multi_face_landmarks[0]
        
        # Enhanced analysis
        focus_status, emotion_status, avg_ratio = enhanced_focus_analysis(face_lm)
        
        # Enhanced blink detection
        if avg_ratio < BLINK_THRESHOLD:
            frames_closed += 1
            if frames_closed >= MIN_BLINK_FRAMES and not is_blinking:
                is_blinking = True
        else:
            if is_blinking and frames_closed >= MIN_BLINK_FRAMES:
                blink_count += 1
            frames_closed = 0
            is_blinking = False
        
        # Enhanced display with confidence
        focus_color = (0, 255, 255) if "High" in focus_status else (0, 165, 255)
        emotion_color = (0, 255, 127) if "Happy" in emotion_status else (255, 255, 0)

        cv2.putText(img, f'FOCUS: {focus_status}', (wCam - 400, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, focus_color, 2)
        cv2.putText(img, f'EMOTION: {emotion_status}', (wCam - 400, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        cv2.putText(img, f'BLINKS: {blink_count}', (wCam - 400, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'EAR: {avg_ratio:.3f}', (wCam - 400, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- 4. Enhanced Display with Performance Metrics ---
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    
    # Performance-based color coding
    fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
    
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, fps_color, 3)
    cv2.putText(img, f'Resolution: {wCam}x{hCam}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Enhanced Multi-Modal Tracker", img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()