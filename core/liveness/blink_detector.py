# blink_detector.py
import cv2
import numpy as np
import mediapipe as mp
import time

class BlinkDetector:
    def __init__(self):
        self.blink_count = 0
        self.ear_history = []
        self.blink_detected = False
        
        # Indices for eyes in MediaPipe Face Mesh
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]

    def _eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C + 1e-6)

    def process(self, landmarks, width, height):
        """
        Process landmarks to detect blinks.
        landmarks: List of normalized landmarks from MediaPipe
        """
        left_eye = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in self.left_eye_idx])
        right_eye = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in self.right_eye_idx])

        ear = (self._eye_aspect_ratio(left_eye) + self._eye_aspect_ratio(right_eye)) / 2
        self.ear_history.append(ear)
        
        # Keep history short
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)

        # Simple blink detection logic
        # EAR drops below threshold then rises
        if len(self.ear_history) > 2:
            if self.ear_history[-2] < 0.20 and self.ear_history[-1] > 0.20:
                self.blink_count += 1
                self.blink_detected = True
                return True
        
        return False

    def get_score(self):
        # If at least one blink detected in the session, return high score
        return 1.0 if self.blink_count > 0 else 0.0
