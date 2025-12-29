"""Run VeriPulse on video file."""
import sys
import cv2

def main(video_path: str):
    cap = cv2.VideoCapture(video_path)
    print(f"Processing: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process frame
    
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
