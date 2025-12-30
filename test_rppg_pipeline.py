import sys
import os
import cv2
import numpy as np

# Add root to path
sys.path.append(os.getcwd())

from core.rppg.processor import RPPGProcessor

def main():
    print("Initializing RPPG Processor...")
    processor = RPPGProcessor(fs=30, method='pos')
    
    # Use webcam (0) or a video file if provided
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]
        
    print(f"Processing source: {source}")
    
    # Run for 10 seconds
    try:
        results = processor.process_video(source, duration=10)
        
        print("\nResults:")
        print(f"Liveness Score: {results.get('liveness_score', 0.0):.4f}")
        print(f"Consistency: {results.get('consistency', 0.0):.4f}")
        
        for key in results:
            if key.endswith("_features"):
                print(f"\n{key}:")
                feats = results[key]
                print(f"  SNR: {feats['snr']:.4f}")
                print(f"  HR (BPM): {feats['hr_bpm']:.2f}")
                print(f"  Periodicity: {feats['periodicity']:.4f}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
