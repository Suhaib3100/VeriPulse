"""Run VeriPulse with webcam input."""
import cv2

def main():
    cap = cv2.VideoCapture(0)
    print("VeriPulse Webcam Demo - Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("VeriPulse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
