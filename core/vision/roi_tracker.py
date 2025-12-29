"""ROI (Region of Interest) tracking for rPPG extraction."""

class ROITracker:
    def extract_rois(self, frame, face_box):
        x, y, w, h = face_box

        forehead = frame[
            y : y + h // 4,
            x + w // 4 : x + 3 * w // 4
        ]

        left_cheek = frame[
            y + h // 2 : y + 3 * h // 4,
            x : x + w // 3
        ]

        right_cheek = frame[
            y + h // 2 : y + 3 * h // 4,
            x + 2 * w // 3 : x + w
        ]

        return {
            "forehead": forehead,
            "left_cheek": left_cheek,
            "right_cheek": right_cheek
        }