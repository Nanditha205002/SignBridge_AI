import os
import cv2
import numpy as np
from src.landmark_extractor import LandmarkExtractor

DATA_PATH = os.path.join('data')
ACTIONS = np.array(['hello', 'hospital', 'doctor', 'yesterday', 'go'])
NO_SEQUENCES = 30  # 30 videos per sign
SEQUENCE_LENGTH = 30  # 30 frames per sequence

extractor = LandmarkExtractor()

# Ensure directories exist
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

cap = cv2.VideoCapture(0)
print("Starting data collection. Press 'q' to quit at any time.")

break_all = False

for action in ACTIONS:
    if break_all:
        break

    for sequence in range(NO_SEQUENCES):
        if break_all:
            break

        # --- PREPARATION / COUNTDOWN PAUSE ---
        # Displays a live preview so camera buffer stays clean and actor can prepare posture
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break_all = True
                break

            image, results = extractor.process_frame(frame)
            extractor.draw_landmarks(image, results)

            cv2.putText(image, f'GET READY: "{action.upper()}" (Video #{sequence})', (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Starting in {countdown}...', (200, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            
            cv2.imshow('SignBridge Data Capture', image)

            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break_all = True
                break

        if break_all:
            break

        # --- FRAME COLLECTION LOOP ---
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break_all = True
                break

            image, results = extractor.process_frame(frame)
            extractor.draw_landmarks(image, results)

            cv2.putText(image, f'Recording "{action}" | Video #{sequence} | Frame {frame_num + 1}/{SEQUENCE_LENGTH}',
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('SignBridge Data Capture', image)

            keypoints = extractor.extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break_all = True
                break

cap.release()
cv2.destroyAllWindows()