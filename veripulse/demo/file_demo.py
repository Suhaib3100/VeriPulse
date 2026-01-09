"""
File Demo - Analyze a video/audio file for deepfakes and liveness.

Usage:
    python -m veripulse.demo.file_demo path/to/video.mp4
    python -m veripulse.demo.file_demo path/to/audio.wav --audio-only
"""

import sys
import time
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from veripulse.engine import (
    VideoSource,
    FaceTracker,
    VideoFrameBatch,
    RPPGExtractor,
    LivenessDetector,
    VideoDeepfakeDetector,
    AudioSource,
    AudioFeatureExtractor,
    AudioDeepfakeDetector,
    MultimodalFusion,
    FusionStrategy,
)


def analyze_video(file_path: str, verbose: bool = True) -> dict:
    """
    Analyze a video file.
    
    Returns dict with all analysis results.
    """
    print(f"\n{'='*60}")
    print(f"VERIPULSE VIDEO ANALYSIS")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    
    start_time = time.time()
    
    # Initialize components
    face_tracker = FaceTracker()
    rppg_extractor = RPPGExtractor()
    liveness_detector = LivenessDetector()
    deepfake_detector = VideoDeepfakeDetector()
    
    # Open video
    print("\n[1/5] Loading video...")
    try:
        video_source = VideoSource.from_file(file_path)
    except Exception as e:
        print(f"Error: Failed to open video - {e}")
        return None
    
    # Collect data
    print("[2/5] Processing frames...")
    frames = []
    landmarks = []
    face_crops = []
    head_poses = []
    blink_events = []
    
    frame_batch = VideoFrameBatch(max_frames=300)
    frame_count = 0
    face_count = 0
    
    for frame_data in video_source.stream_frames():
        frame = frame_data.bgr
        frame_count += 1
        
        # Track face
        detection = face_tracker.process(frame)
        
        if detection and detection.landmarks is not None:
            face_count += 1
            frames.append(frame)
            landmarks.append(detection.landmarks)
            
            if detection.face_crop is not None:
                face_crops.append(detection.face_crop)
            
            if detection.head_pose:
                head_poses.append(detection.head_pose)
            
            frame_batch.add_frame(frame_data, detection)
            
            # Check for blinks
            if detection.ear is not None and detection.ear < 0.2:
                blink_events.append({
                    'time': frame_data.timestamp,
                    'ear': detection.ear,
                    'duration': 100  # Estimate
                })
        
        # Progress
        if frame_count % 100 == 0:
            print(f"    Processed {frame_count} frames, {face_count} with faces...")
        
        # Limit to 10 seconds
        if len(frames) >= 300:
            break
    
    video_source.close()
    
    fps = video_source.fps if hasattr(video_source, 'fps') else 30
    duration = len(frames) / fps if fps > 0 else 0
    
    print(f"    Total: {frame_count} frames, {face_count} with faces ({duration:.1f}s)")
    
    if len(frames) < 30:
        print("Error: Not enough frames with faces detected")
        return None
    
    # rPPG analysis
    print("[3/5] Extracting physiological signals (rPPG)...")
    roi_signals = frame_batch.get_roi_signals()
    
    rppg_result = None
    if roi_signals and len(list(roi_signals.values())[0]) > 60:
        rppg_result = rppg_extractor.process(roi_signals)
        if verbose:
            print(f"    BPM: {rppg_result.get_best_bpm():.1f}")
            print(f"    SNR: {rppg_result.global_features.snr:.1f} dB")
            print(f"    Quality: {rppg_result.quality_score:.2f}")
    
    # Liveness analysis
    print("[4/5] Analyzing liveness...")
    texture_features = _extract_texture_features(frames)
    
    liveness_result = liveness_detector.analyze(
        rppg_result=rppg_result,
        blink_events=blink_events,
        head_poses=head_poses,
        texture_features=texture_features,
        fps=fps,
        duration_seconds=duration
    )
    
    if verbose:
        print(f"    Verdict: {liveness_result.verdict}")
        print(f"    Score: {liveness_result.liveness_score:.2f}")
        print(f"    Has Pulse: {liveness_result.physio.has_pulse}")
        print(f"    Blinks: {liveness_result.physio.blink_count}")
    
    # Deepfake detection
    print("[5/5] Detecting deepfake artifacts...")
    deepfake_result = deepfake_detector.analyze(
        frames=frames,
        landmarks=landmarks,
        face_crops=face_crops,
        fps=fps
    )
    
    if verbose:
        print(f"    Verdict: {deepfake_result.verdict}")
        print(f"    Probability: {deepfake_result.deepfake_probability:.1%}")
        print(f"    Confidence: {deepfake_result.confidence:.1%}")
    
    analysis_time = (time.time() - start_time) * 1000
    
    return {
        'rppg': rppg_result,
        'liveness': liveness_result,
        'video_deepfake': deepfake_result,
        'duration': duration,
        'frame_count': frame_count,
        'face_count': face_count,
        'analysis_time_ms': analysis_time
    }


def analyze_audio(file_path: str, verbose: bool = True) -> dict:
    """Analyze an audio file."""
    print(f"\n{'='*60}")
    print(f"VERIPULSE AUDIO ANALYSIS")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    
    start_time = time.time()
    
    # Initialize
    feature_extractor = AudioFeatureExtractor()
    deepfake_detector = AudioDeepfakeDetector()
    
    # Load audio
    print("\n[1/2] Loading audio...")
    try:
        audio_source = AudioSource.from_file(file_path)
    except Exception as e:
        print(f"Error: Failed to open audio - {e}")
        return None
    
    print(f"    Duration: {audio_source.duration:.2f}s")
    print(f"    Sample rate: {audio_source.sample_rate} Hz")
    
    # Extract features
    print("[2/2] Analyzing...")
    features = feature_extractor.extract(audio_source)
    
    # Detect deepfake
    result = deepfake_detector.analyze(audio_source, features)
    
    audio_source.close()
    
    if verbose:
        print(f"    Verdict: {result.verdict}")
        print(f"    Synthetic Probability: {result.synthetic_probability:.1%}")
        print(f"    Confidence: {result.confidence:.1%}")
        print(f"    Naturalness: {result.forensics.naturalness_score:.2f}")
    
    analysis_time = (time.time() - start_time) * 1000
    
    return {
        'audio_deepfake': result,
        'features': features,
        'duration': features.duration,
        'analysis_time_ms': analysis_time
    }


def _extract_texture_features(frames):
    """Extract texture features from frames."""
    import cv2
    import numpy as np
    
    if not frames:
        return {}
    
    mid_frame = frames[len(frames) // 2]
    
    if len(mid_frame.shape) == 3:
        gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = mid_frame
    
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()
    
    return {
        'laplacian_var': float(lap_var),
        'moire_score': 0.1,
        'reflection_consistency': 0.5,
        'depth_consistency': 0.5
    }


def print_final_verdict(video_results: dict = None, audio_results: dict = None):
    """Print the final combined verdict."""
    
    print(f"\n{'='*60}")
    print(f"FINAL TRUST ASSESSMENT")
    print(f"{'='*60}")
    
    fusion = MultimodalFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)
    
    trust_result = fusion.fuse(
        liveness=video_results.get('liveness') if video_results else None,
        video_deepfake=video_results.get('video_deepfake') if video_results else None,
        audio_deepfake=audio_results.get('audio_deepfake') if audio_results else None,
        rppg=video_results.get('rppg') if video_results else None,
        video_duration=video_results.get('duration', 0) if video_results else 0,
        analysis_time_ms=(video_results.get('analysis_time_ms', 0) if video_results else 0) +
                        (audio_results.get('analysis_time_ms', 0) if audio_results else 0)
    )
    
    # Print result
    print(f"\n  VERDICT: {trust_result.verdict}")
    print(f"  Trust Score: {trust_result.trust_score:.1%}")
    print(f"  Trust Level: {trust_result.trust_level.value.upper()}")
    print(f"  Confidence: {trust_result.overall_confidence:.1%}")
    
    print(f"\n  Component Scores:")
    print(f"    Liveness:          {trust_result.liveness_score:.2f}")
    print(f"    Video Authenticity: {trust_result.video_authenticity:.2f}")
    print(f"    Audio Authenticity: {trust_result.audio_authenticity:.2f}")
    print(f"    rPPG Quality:       {trust_result.rppg_quality:.2f}")
    
    if trust_result.threat_indicators:
        print(f"\n  Threats Detected: {', '.join(trust_result.threat_indicators)}")
    
    print(f"\n  Explanation: {trust_result.explanation}")
    
    print(f"\n  Recommendations:")
    for rec in trust_result.recommendations:
        print(f"    • {rec}")
    
    print(f"\n  Analysis Time: {trust_result.analysis_duration_ms:.0f}ms")
    print(f"{'='*60}\n")
    
    return trust_result


def main():
    parser = argparse.ArgumentParser(
        description="VeriPulse File Analysis Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", help="Path to video or audio file")
    parser.add_argument("--audio-only", action="store_true", help="Analyze only audio")
    parser.add_argument("--video-only", action="store_true", help="Analyze only video")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    file_path = args.file
    verbose = not args.quiet
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    video_results = None
    audio_results = None
    
    # Determine file type
    ext = Path(file_path).suffix.lower()
    is_audio = ext in {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
    is_video = ext in {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
    
    if is_audio or args.audio_only:
        audio_results = analyze_audio(file_path, verbose)
    
    if is_video or (not is_audio and not args.audio_only):
        if not args.audio_only:
            video_results = analyze_video(file_path, verbose)
    
    # Final verdict
    if video_results or audio_results:
        print_final_verdict(video_results, audio_results)
    else:
        print("Error: No analysis results to report")
        sys.exit(1)


if __name__ == "__main__":
    main()
