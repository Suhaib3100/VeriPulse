"""
Webcam Demo - Real-time liveness and deepfake detection from webcam.

Usage:
    python -m veripulse.demo.webcam_demo
    python -m veripulse.demo.webcam_demo --camera 1
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from veripulse.engine import (
    VideoSource,
    FaceTracker,
    VideoFrameBatch,
    RPPGExtractor,
    LivenessDetector,
    VideoDeepfakeDetector,
    QuickScreener,
)


class RealtimeAnalyzer:
    """
    Real-time video analyzer with rolling window analysis.
    """
    
    def __init__(
        self,
        window_seconds: float = 5.0,
        fps: float = 30.0,
        update_interval: float = 1.0
    ):
        self.window_seconds = window_seconds
        self.fps = fps
        self.update_interval = update_interval
        
        # Components
        self.face_tracker = FaceTracker()
        self.rppg_extractor = RPPGExtractor(fps=fps)
        self.liveness_detector = LivenessDetector()
        self.screener = QuickScreener()
        
        # State
        self.frame_batch = VideoFrameBatch(max_frames=int(window_seconds * fps))
        self.frames = []
        self.landmarks = []
        self.head_poses = []
        self.blink_events = []
        
        self.last_analysis_time = 0
        self.current_result = None
    
    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> dict:
        """
        Process a single frame and return current analysis state.
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Track face
        detection = self.face_tracker.process(frame)
        
        result = {
            'face_detected': detection is not None and detection.landmarks is not None,
            'face_bbox': detection.bbox if detection else None,
            'head_pose': detection.head_pose if detection else None,
            'ear': detection.ear if detection else None,
            'analysis': self.current_result
        }
        
        if detection and detection.landmarks is not None:
            # Store data
            self.frames.append(frame)
            self.landmarks.append(detection.landmarks)
            
            if detection.head_pose:
                self.head_poses.append(detection.head_pose)
            
            # Create frame data
            from veripulse.engine.video_pipeline import Frame
            frame_data = Frame(
                bgr=frame,
                timestamp=timestamp
            )
            self.frame_batch.add_frame(frame_data, detection)
            
            # Check for blink
            if detection.ear is not None and detection.ear < 0.2:
                self.blink_events.append({
                    'time': timestamp,
                    'ear': detection.ear
                })
            
            # Keep window
            max_frames = int(self.window_seconds * self.fps)
            if len(self.frames) > max_frames:
                self.frames = self.frames[-max_frames:]
                self.landmarks = self.landmarks[-max_frames:]
                self.head_poses = self.head_poses[-max_frames:]
        
        # Run analysis periodically
        if timestamp - self.last_analysis_time >= self.update_interval:
            self._run_analysis()
            self.last_analysis_time = timestamp
            result['analysis'] = self.current_result
        
        return result
    
    def _run_analysis(self):
        """Run periodic analysis on collected frames."""
        if len(self.frames) < int(self.fps * 2):  # Need at least 2 seconds
            return
        
        # Get ROI signals
        roi_signals = self.frame_batch.get_roi_signals()
        
        # rPPG
        rppg_result = None
        if roi_signals and len(list(roi_signals.values())[0]) > 60:
            rppg_result = self.rppg_extractor.process(roi_signals)
        
        # Texture features
        texture_features = self._extract_texture()
        
        # Liveness
        liveness_result = self.liveness_detector.analyze(
            rppg_result=rppg_result,
            blink_events=self.blink_events[-20:],  # Recent blinks
            head_poses=self.head_poses,
            texture_features=texture_features,
            fps=self.fps,
            duration_seconds=len(self.frames) / self.fps
        )
        
        # Quick screen
        screen_verdict, screen_conf = self.screener.screen(self.frames[-90:])
        
        self.current_result = {
            'rppg': rppg_result,
            'liveness': liveness_result,
            'quick_screen': (screen_verdict, screen_conf),
            'blink_count': len(self.blink_events),
            'frame_count': len(self.frames)
        }
    
    def _extract_texture(self) -> dict:
        """Extract texture features from recent frames."""
        if not self.frames:
            return {}
        
        frame = self.frames[-1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        
        return {
            'laplacian_var': float(lap.var()),
            'moire_score': 0.1,
            'reflection_consistency': 0.5,
            'depth_consistency': 0.5
        }


def draw_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    """Draw analysis overlay on frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Colors
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    
    # Face bbox
    if result.get('face_bbox'):
        x, y, bw, bh = result['face_bbox']
        color = GREEN if result.get('face_detected') else RED
        cv2.rectangle(overlay, (int(x), int(y)), (int(x+bw), int(y+bh)), color, 2)
    
    # Info panel
    panel_h = 180
    cv2.rectangle(overlay, (10, 10), (350, panel_h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (350, panel_h), WHITE, 1)
    
    y_offset = 30
    line_height = 22
    
    # Title
    cv2.putText(overlay, "VERIPULSE ANALYSIS", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    y_offset += line_height + 5
    
    # Face status
    face_status = "Face: DETECTED" if result.get('face_detected') else "Face: NOT FOUND"
    face_color = GREEN if result.get('face_detected') else RED
    cv2.putText(overlay, face_status, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
    y_offset += line_height
    
    # Analysis results
    analysis = result.get('analysis')
    if analysis:
        # Liveness
        liveness = analysis.get('liveness')
        if liveness:
            status = liveness.verdict
            score = liveness.liveness_score
            color = GREEN if liveness.is_live else RED
            cv2.putText(overlay, f"Liveness: {status} ({score:.0%})", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height
            
            # Sub-details
            cv2.putText(overlay, f"  Pulse: {'Yes' if liveness.physio.has_pulse else 'No'}", 
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
            y_offset += 18
            cv2.putText(overlay, f"  Blinks: {liveness.physio.blink_count}", 
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
            y_offset += 18
        
        # rPPG
        rppg = analysis.get('rppg')
        if rppg:
            bpm = rppg.get_best_bpm()
            cv2.putText(overlay, f"Heart Rate: {bpm:.0f} BPM", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
            y_offset += line_height
        
        # Quick screen
        quick_screen = analysis.get('quick_screen')
        if quick_screen:
            verdict, conf = quick_screen
            color = GREEN if verdict == "PASS" else (YELLOW if verdict == "NEEDS_ANALYSIS" else RED)
            cv2.putText(overlay, f"Quick Screen: {verdict}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        cv2.putText(overlay, "Analyzing... (wait 2s)", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
    
    # Instructions
    cv2.putText(overlay, "Press 'q' to quit | 'r' to reset", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    
    return overlay


def main():
    parser = argparse.ArgumentParser(description="VeriPulse Webcam Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    
    args = parser.parse_args()
    
    print("VeriPulse Webcam Demo")
    print("=" * 40)
    print(f"Camera: {args.camera}")
    print(f"Resolution: {args.width}x{args.height}")
    print("Press 'q' to quit, 'r' to reset analysis")
    print()
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = RealtimeAnalyzer(fps=args.fps)
    
    print("Camera opened. Starting analysis...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Process frame
            result = analyzer.process_frame(frame)
            
            # Draw overlay
            display = draw_overlay(frame, result)
            
            # Show
            cv2.imshow("VeriPulse Demo", display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                analyzer = RealtimeAnalyzer(fps=args.fps)
                print("Analysis reset")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nDemo ended.")


if __name__ == "__main__":
    main()
