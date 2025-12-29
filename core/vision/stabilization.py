"""Frame and ROI stabilization."""

class Stabilizer:
    def __init__(self):
        self.prev_box = None

    def smooth(self, face_box, alpha=0.7):
        if self.prev_box is None:
            self.prev_box = face_box
            return face_box

        smoothed = []
        for prev, curr in zip(self.prev_box, face_box):
            smoothed.append(int(alpha * prev + (1 - alpha) * curr))

        self.prev_box = tuple(smoothed)
        return self.prev_box
