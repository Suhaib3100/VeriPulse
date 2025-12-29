print("SCRIPT STARTED")

import cv2
from core.vision.face_detector import FaceDetector
from core.vision.roi_tracker import ROITracker
from core.vision.stabilization import Stabilizer
from core.scoring.model import TrustModel

def main():
    cap = cv2.VideoCapture(0)
    print("Camera opened:", cap.isOpened())

    face_detector = FaceDetector()
    roi_tracker = ROITracker()
    stabilizer = Stabilizer()
    trust_model = TrustModel()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Detect face
        face = face_detector.detect(frame)
        if face is not None:
            # Smooth face box
            face = stabilizer.smooth(face)
            x, y, w, h = face

            # Draw face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract ROIs
            rois = roi_tracker.extract_rois(frame, face)
            for region, roi in rois.items():
                print(f"{region} ROI shape:", roi.shape)

            # Dummy trust scores (replace with real rPPG/blink/motion)
            rppg_score = 0.7
            blink_score = 0.6
            motion_score = 0.3

            # Evaluate trust
            trust_state = trust_model.evaluate(rppg_score, blink_score, motion_score)
            print("Trust State:", trust_state.value)

            # Display on frame
            cv2.putText(
                frame,
                f"Trust: {trust_state.value}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("Vision + Scoring Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
