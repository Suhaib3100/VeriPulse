# blink_detector.py
import cv2
import numpy as np
import mediapipe as mp
import time

try:
    mp_face = mp.solutions.face_mesh
except AttributeError:
    class MockSolutions:
        face_mesh = None
    mp_face = MockSolutions()

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C + 1e-6)

def get_vision_score(duration=10):
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face.FaceMesh(refine_landmarks=True)
    blink_count = 0
    ear_history = []
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]

            left_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in left_eye_idx])
            right_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in right_eye_idx])

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
            ear_history.append(ear)

            if len(ear_history) > 2 and ear_history[-2] > 0.25 and ear < 0.20:
                blink_count += 1

    cap.release()
    cv2.destroyAllWindows()

    blink_score = min(blink_count / 3, 1.0)
    return float(blink_score)

if __name__ == "__main__":
    print(get_vision_score())