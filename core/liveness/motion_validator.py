# motion_validator.py
import cv2
import mediapipe as mp
import numpy as np
import time

mp_face = mp.solutions.face_mesh

def validate_motion(duration=5):
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face.FaceMesh(refine_landmarks=True)
    movements = []
    last_pos = None
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            nose = np.array([lm[1].x * w, lm[1].y * h])

            if last_pos is not None:
                movement = np.linalg.norm(nose - last_pos)
                movements.append(movement)

            last_pos = nose

    cap.release()

    avg_motion = np.mean(movements) if movements else 0

    if avg_motion > 5:
        return 1.0
    elif avg_motion > 2:
        return 0.6
    else:
        return 0.0

if __name__ == "__main__":
    print(validate_motion())
