import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui # Required for mouse control

# --- 1. Initialization and Configuration ---

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize MediaPipe drawing utility
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

# Set the webcam resolution (adjust if needed, but 640x480 is standard)
wCam, hCam = 640, 480
cap.set(3, wCam) # 3 is the ID for width
cap.set(4, hCam) # 4 is the ID for height

# Get screen resolution for mouse mapping
wScr, hScr = pyautogui.size()

# Smoothening parameters for mouse movement
smoothening = 7
plocX, plocY = 0, 0 # Previous location X and Y
clocX, clocY = 0, 0 # Current location X and Y

# Area where the hand movement translates to mouse movement (to avoid edge jitter)
frameR = 100 # Reduction Frame for the mouse zone

# Variables for calculating FPS
pTime = 0 # Previous time

# Helper list for finger tip landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]

# Cooldown for non-mouse gestures to prevent spamming
last_action_time = 0 
COOLDOWN_TIME = 1.5 # 1.5 seconds to prevent spamming actions

# --- 2. Helper Functions ---

def get_finger_status(hand_landmarks, handedness_label):
    """Checks which fingers are extended (open)."""
    lm = hand_landmarks.landmark
    is_open = []
    
    # 1. Thumb Check (Special case: check against the X-axis of the base joint 3)
    if handedness_label == "Right":
        # Right hand: Tip (4) is to the right of the base (3)
        is_open.append(lm[FINGER_TIPS[0]].x > lm[FINGER_TIPS[0]-1].x)
    else: 
        # Left hand: Tip (4) is to the left of the base (3)
        is_open.append(lm[FINGER_TIPS[0]].x < lm[FINGER_TIPS[0]-1].x)
            
    # 2. Other Four Fingers Check (Tip must be higher (smaller Y value) than the MCP joint)
    for id in range(1, 5):
        tip_y = lm[FINGER_TIPS[id]].y
        mcp_y = lm[FINGER_TIPS[id] - 3].y
        is_open.append(tip_y < mcp_y)
            
    return is_open # [Thumb, Index, Middle, Ring, Pinky]

def detect_gesture(finger_status, hand_landmarks):
    """Detects essential hand gestures for Virtual Mouse and System Control."""
    lm = hand_landmarks.landmark
    
    # Check for Pinch Click
    thumb_tip = np.array([lm[FINGER_TIPS[0]].x, lm[FINGER_TIPS[0]].y])
    index_tip = np.array([lm[FINGER_TIPS[1]].x, lm[FINGER_TIPS[1]].y])
    # Normalized distance between thumb and index finger tips
    pinch_distance = np.linalg.norm(thumb_tip - index_tip)
    
    # Define a click action based on a close distance (pinch)
    is_pinching = pinch_distance < 0.05
    
    # --- ONLY ESSENTIAL GESTURES FOR KEYBOARD/SYSTEM ACTIONS ---
    
    # 1. Play/Pause (Middle Finger up only) -> [False, False, True, False, False]
    if finger_status == [False, False, True, False, False]:
        return "Play/Pause (Space)", is_pinching

    # 2. Close Tab (Pinky up only) -> [False, False, False, False, True]
    if finger_status == [False, False, False, False, True]:
        return "Close Tab (Ctrl+W)", is_pinching
    
    # 3. Mouse Pointer (Index up) - Primary movement/click mode
    # If the Index finger is up, we are in mouse mode (move or click)
    if finger_status[1]:
        return "Mouse Pointer", is_pinching
    
    # Default (No recognized action gesture)
    return "Idle/Unknown", is_pinching

def draw_bounding_box(img, hand_landmarks):
    """Calculates and draws a tight bounding box around the hand."""
    h, w, c = img.shape
    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]

    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
    return x_min, y_min # Return top-left for placing text

# --- 3. Main Loop ---
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    img = cv2.flip(img, 1) # Flip for mirror effect
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Draw the mouse control area on the camera feed
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            # Use the first hand found for control
            if idx == 0:
                # Extract Handedness
                handedness_label = results.multi_handedness[idx].classification[0].label
                
                # --- Detection Logic ---
                finger_status = get_finger_status(hand_landmarks, handedness_label)
                current_gesture, is_pinching = detect_gesture(finger_status, hand_landmarks)

                # Get the index finger tip position (Landmark 8) for movement and click feedback
                lm8 = hand_landmarks.landmark[FINGER_TIPS[1]]
                x1, y1 = int(lm8.x * wCam), int(lm8.y * hCam)

                # 1. Draw Landmarks and Bounding Box
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x_min, y_min = draw_bounding_box(img, hand_landmarks)
                
                # 2. Display Handedness and Gesture Text
                display_y = y_min - 10 if y_min > 50 else y_min + 50
                cv2.putText(img, f'Hand: {handedness_label}', (x_min, display_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img, f'Gesture: {current_gesture}', (x_min, display_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # --- VIRTUAL MOUSE CONTROL LOGIC ---
                
                # Check if we are in Mouse Pointer Mode (Index finger up)
                if current_gesture == "Mouse Pointer":
                    
                    # Mode 1: Moving (Index up, NO Pinch)
                    if not is_pinching:
                        # Convert coordinates from camera to screen (mapping the frameR zone)
                        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                        # Apply smoothing
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening
                        
                        # Move Mouse
                        pyautogui.moveTo(wScr - clocX, clocY) # wScr - clocX flips the movement to match the mirror image
                        plocX, plocY = clocX, clocY
                        
                        # Draw a tracking circle on the index tip
                        cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                        cv2.putText(img, "MOUSE MOVING", (10, hCam - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Mode 2: Clicking (Index up, Pinch detected) - with a small cooldown for better UX
                    # The click cooldown is deliberately shorter (0.5s) than the system action cooldown (1.5s)
                    elif is_pinching and (current_time - last_action_time > 0.5):
                        
                        pyautogui.click()
                        last_action_time = current_time # Reset click cooldown
                        
                        # Draw visual feedback for a click
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, "CLICK TRIGGERED!", (10, hCam - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # --- SYSTEM GESTURE CONTROL LOGIC (Non-Mouse Mode) ---

                # Only check system gestures if the cool down is passed
                elif current_time - last_action_time > COOLDOWN_TIME:
                    
                    if current_gesture == "Play/Pause (Space)":
                        cv2.putText(img, "PLAY/PAUSE ACTION TRIGGERED!", (10, hCam - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
                        pyautogui.press('space')
                        last_action_time = current_time # Reset cooldown
                        
                    elif current_gesture == "Close Tab (Ctrl+W)":
                        cv2.putText(img, "CLOSE TAB ACTION TRIGGERED!", (10, hCam - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                        pyautogui.hotkey('ctrl', 'w')
                        last_action_time = current_time # Reset cooldown
                        
                # Default Text if no action or mouse movement is active (Idle/Unknown)
                else:
                    cv2.putText(img, "IDLE", (10, hCam - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


    # --- 4. Display FPS ---
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # --- 5. Display the Frame and Exit Condition ---
    cv2.imshow("Advanced Hand and Gesture Tracker", img)

    # Press 'q' to quit the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
     