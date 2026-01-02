# motion_validator.py
import numpy as np

class MotionValidator:
    def __init__(self):
        self.movements = []
        self.last_nose_pos = None

    def process(self, landmarks, width, height):
        """
        Process landmarks to detect head motion.
        """
        # Nose tip is index 1
        nose = np.array([landmarks[1].x * width, landmarks[1].y * height])

        if self.last_nose_pos is not None:
            movement = np.linalg.norm(nose - self.last_nose_pos)
            self.movements.append(movement)

        self.last_nose_pos = nose
        return movement if self.last_nose_pos is not None else 0.0

    def get_score(self):
        if not self.movements:
            return 0.0
        
        avg_motion = np.mean(self.movements)
        # We expect some micro-motion (living), but not too much (shaking)
        # and not zero (static image)
        
        if 0.5 < avg_motion < 10.0:
            return 1.0
        elif avg_motion <= 0.5:
            return 0.2 # Too static (photo?)
        else:
            return 0.5 # Too much motion
        return 0.0

if __name__ == "__main__":
    print(validate_motion())
