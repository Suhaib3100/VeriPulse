import cv2
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.vision.video_reader import VideoReader
from core.rppg.processor import RPPGProcessor

def main():
    parser = argparse.ArgumentParser(description="rPPG Liveness Demo")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam or file path)")
    args = parser.parse_args()
    
    # Handle numeric source for webcam
    source = args.source
    if source.isdigit():
        source = int(source)
        
    print(f"Starting demo with source: {source}")
    
    video = VideoReader(source=source)
    processor = RPPGProcessor(fs=30, buffer_size=300) # 10 seconds buffer
    
    print("Press ESC to exit.")
    
    for frame, timestamp in video:
        # Process frame
        result = processor.process_frame(frame)
        
        # Visualization
        bbox = result.get('bbox')
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Overlay info
            label = result['label']
            score = result['liveness_score']
            bpm = result['bpm']
            snr = result['snr']
            
            color = (0, 255, 0) if label == "LIVE" else (0, 0, 255)
            if label == "WAITING":
                color = (255, 255, 0)
                
            text = f"{label} ({score:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            info_text = f"BPM: {bpm:.1f} | SNR: {snr:.1f} dB"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        cv2.imshow("rPPG Liveness Demo", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
