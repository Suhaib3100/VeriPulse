"""
Video Deepfake Detection - Visual forensics and neural detection.

This module provides:
1. Forensic feature extraction (temporal, spatial, frequency)
2. CNN/LSTM-based deepfake classifier (pluggable models)
3. Ensemble scoring with multiple detection signals

Detection signals include:
- Temporal consistency (facial landmarks over time)
- Frequency artifacts (GAN fingerprints)
- Texture anomalies (skin unnaturalness)
- Face-background blending artifacts
- Eye/mouth region inconsistencies
"""

import numpy as np
from scipy import signal
from scipy.fft import fft2, fftshift
from scipy.ndimage import laplace, sobel
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import cv2


class DeepfakeSignal(Enum):
    """Types of deepfake detection signals."""
    TEMPORAL_CONSISTENCY = "temporal"
    FREQUENCY_ARTIFACTS = "frequency"
    TEXTURE_ANOMALIES = "texture"
    BLENDING_ARTIFACTS = "blending"
    PHYSIOLOGICAL = "physiological"
    NEURAL_CLASSIFIER = "neural"


@dataclass
class ForensicFeatures:
    """Forensic features extracted from video frames."""
    
    # Temporal features
    landmark_stability: float = 0.0      # How stable are landmarks over time
    motion_smoothness: float = 0.0       # Is motion natural or jerky
    temporal_consistency: float = 0.0    # Frame-to-frame consistency
    blink_rate_natural: float = 0.0      # Is blink rate physiological
    
    # Frequency domain features
    high_freq_energy: float = 0.0        # High frequency content
    freq_anomaly_score: float = 0.0      # GAN fingerprint detection
    compression_artifacts: float = 0.0    # JPEG/compression artifacts
    
    # Texture features  
    skin_texture_variance: float = 0.0   # Natural skin texture
    laplacian_variance: float = 0.0      # Edge sharpness
    color_histogram_entropy: float = 0.0 # Color distribution
    noise_pattern: float = 0.0           # Noise characteristics
    
    # Blending features
    face_boundary_score: float = 0.0     # Face-background transition
    color_mismatch: float = 0.0          # Color consistency face vs background
    lighting_consistency: float = 0.0    # Lighting direction consistency
    
    # Regional features
    eye_quality: float = 0.0             # Eye region naturalness
    mouth_quality: float = 0.0           # Mouth region naturalness
    nose_quality: float = 0.0            # Nose region naturalness
    
    # Raw feature vectors (for neural classifier)
    feature_vector: Optional[np.ndarray] = None


@dataclass
class DeepfakeResult:
    """Result of deepfake detection analysis."""
    
    # Primary output
    is_deepfake: bool = False
    deepfake_probability: float = 0.0    # 0-1, higher = more likely fake
    confidence: float = 0.0
    
    # Component scores (all 0-1, higher = more suspicious)
    temporal_score: float = 0.0
    frequency_score: float = 0.0
    texture_score: float = 0.0
    blending_score: float = 0.0
    neural_score: float = 0.0            # From neural network (if available)
    
    # Forensic features
    forensics: ForensicFeatures = field(default_factory=ForensicFeatures)
    
    # Detailed findings
    suspicious_regions: List[str] = field(default_factory=list)
    anomaly_frames: List[int] = field(default_factory=list)
    
    # Verdict
    verdict: str = "UNKNOWN"
    explanation: str = ""


class VideoDeepfakeDetector:
    """
    Multi-signal deepfake detection for video content.
    
    Combines forensic analysis with optional neural network classification.
    
    Example:
        >>> detector = VideoDeepfakeDetector()
        >>> result = detector.analyze(
        ...     frames=video_frames,
        ...     landmarks=landmark_sequence,
        ...     face_crops=face_crops
        ... )
        >>> print(f"Deepfake probability: {result.deepfake_probability:.1%}")
    """
    
    # Detection thresholds
    DEEPFAKE_THRESHOLD = 0.5
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(
        self,
        use_neural: bool = False,
        model_path: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize deepfake detector.
        
        Args:
            use_neural: Whether to use neural network classifier
            model_path: Path to pretrained deepfake model
            weights: Custom weights for combining signals
        """
        self.use_neural = use_neural
        self.model_path = model_path
        self.model = None
        
        # Default weights for signal combination
        self.weights = weights or {
            'temporal': 0.20,
            'frequency': 0.20,
            'texture': 0.25,
            'blending': 0.20,
            'neural': 0.15
        }
        
        if use_neural and model_path:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load neural network model for deepfake detection."""
        # Placeholder - implement with actual model loading
        # Could use EfficientNet, XceptionNet, or custom architecture
        try:
            # Example: self.model = torch.load(self.model_path)
            print(f"Note: Neural model loading not implemented. Path: {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.use_neural = False
    
    def analyze(
        self,
        frames: List[np.ndarray],
        landmarks: Optional[List[np.ndarray]] = None,
        face_crops: Optional[List[np.ndarray]] = None,
        fps: float = 30.0
    ) -> DeepfakeResult:
        """
        Analyze video frames for deepfake indicators.
        
        Args:
            frames: List of video frames (BGR format)
            landmarks: List of facial landmarks per frame
            face_crops: List of cropped face regions
            fps: Video frame rate
            
        Returns:
            DeepfakeResult with detection scores and forensics
        """
        result = DeepfakeResult()
        
        if len(frames) == 0:
            result.verdict = "NO_DATA"
            result.explanation = "No frames provided for analysis"
            return result
        
        # Use face crops if available, otherwise use full frames
        analysis_frames = face_crops if face_crops else frames
        
        # 1. Temporal analysis
        if landmarks is not None and len(landmarks) > 10:
            result.temporal_score = self._analyze_temporal(landmarks, fps)
        
        # 2. Frequency analysis
        result.frequency_score = self._analyze_frequency(analysis_frames)
        
        # 3. Texture analysis
        result.texture_score = self._analyze_texture(analysis_frames)
        
        # 4. Blending analysis (requires both face and full frame)
        if face_crops is not None and len(frames) > 0:
            result.blending_score = self._analyze_blending(frames, face_crops)
        
        # 5. Neural network (if enabled)
        if self.use_neural and self.model is not None:
            result.neural_score = self._neural_classify(analysis_frames)
        
        # Store forensic features
        result.forensics = self._extract_forensic_features(
            analysis_frames, landmarks
        )
        
        # Combine scores
        self._compute_final_score(result)
        
        # Determine verdict
        self._compute_verdict(result)
        
        return result
    
    def _analyze_temporal(
        self,
        landmarks: List[np.ndarray],
        fps: float
    ) -> float:
        """Analyze temporal consistency of facial landmarks."""
        if len(landmarks) < 10:
            return 0.5  # Neutral
        
        scores = []
        
        # 1. Landmark stability (jitter)
        landmark_diffs = []
        for i in range(1, len(landmarks)):
            if landmarks[i] is not None and landmarks[i-1] is not None:
                diff = np.mean(np.abs(landmarks[i] - landmarks[i-1]))
                landmark_diffs.append(diff)
        
        if landmark_diffs:
            jitter = np.std(landmark_diffs)
            # High jitter can indicate deepfake artifacts
            # Normal: 0.5-3, Suspicious: >5 or <0.1 (too stable)
            jitter_score = 0.0
            if jitter < 0.1:
                jitter_score = 0.7  # Suspiciously stable
            elif jitter > 5:
                jitter_score = 0.8  # Too unstable
            else:
                jitter_score = 0.2  # Normal
            scores.append(jitter_score)
        
        # 2. Motion smoothness
        if len(landmark_diffs) > 2:
            # Second derivative (acceleration)
            accel = np.diff(landmark_diffs)
            accel_var = np.var(accel)
            
            # Natural motion has some smoothness
            if accel_var > 10:
                scores.append(0.7)  # Jerky motion
            else:
                scores.append(0.2)  # Smooth motion
        
        # 3. Check for "frozen" regions
        # Deepfakes sometimes have regions that don't move naturally
        if len(landmarks) > 30:
            chunk_size = 30
            chunks = [landmarks[i:i+chunk_size] for i in range(0, len(landmarks)-chunk_size, chunk_size)]
            
            frozen_count = 0
            for chunk in chunks:
                valid = [l for l in chunk if l is not None]
                if len(valid) > 1:
                    variance = np.var([np.mean(l) for l in valid])
                    if variance < 0.01:
                        frozen_count += 1
            
            if frozen_count > len(chunks) * 0.5:
                scores.append(0.8)  # Many frozen regions
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_frequency(self, frames: List[np.ndarray]) -> float:
        """Analyze frequency domain for GAN artifacts."""
        if len(frames) == 0:
            return 0.5
        
        scores = []
        
        # Sample frames evenly
        sample_indices = np.linspace(0, len(frames)-1, min(10, len(frames)), dtype=int)
        
        for idx in sample_indices:
            frame = frames[idx]
            if frame is None or frame.size == 0:
                continue
            
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Resize for consistent analysis
            gray = cv2.resize(gray, (256, 256))
            
            # 2D FFT
            f_transform = fft2(gray.astype(float))
            f_shift = fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Log magnitude spectrum
            magnitude = np.log1p(magnitude)
            
            # Analyze radial profile
            center = np.array(magnitude.shape) // 2
            y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
            
            # Radial average
            r_max = min(center)
            radial_profile = np.bincount(r.ravel(), weights=magnitude.ravel())
            radial_counts = np.bincount(r.ravel())
            radial_profile = radial_profile[:r_max] / (radial_counts[:r_max] + 1e-10)
            
            # Look for anomalies in high frequencies
            # GANs often have characteristic patterns
            if len(radial_profile) > 50:
                high_freq = radial_profile[len(radial_profile)//2:]
                high_freq_var = np.var(high_freq)
                
                # Unusual high frequency patterns
                if high_freq_var > 2 or np.mean(high_freq) > np.mean(radial_profile[:len(radial_profile)//4]) * 0.8:
                    scores.append(0.7)
                else:
                    scores.append(0.3)
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_texture(self, frames: List[np.ndarray]) -> float:
        """Analyze texture characteristics for deepfake artifacts."""
        if len(frames) == 0:
            return 0.5
        
        scores = []
        lap_variances = []
        
        # Sample frames
        sample_indices = np.linspace(0, len(frames)-1, min(15, len(frames)), dtype=int)
        
        for idx in sample_indices:
            frame = frames[idx]
            if frame is None or frame.size == 0:
                continue
            
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Laplacian variance (texture detail)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = lap.var()
            lap_variances.append(lap_var)
            
            # Local binary pattern variance
            # High uniformity can indicate synthetic textures
            
            # Noise estimation
            # Deepfakes often have inconsistent noise patterns
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = np.abs(gray.astype(float) - blur.astype(float))
            noise_std = np.std(noise)
            
            # Natural images: noise_std typically 5-30
            if noise_std < 2 or noise_std > 50:
                scores.append(0.6)
            else:
                scores.append(0.3)
        
        # Check texture consistency across frames
        if len(lap_variances) > 3:
            variance_consistency = 1 - (np.std(lap_variances) / (np.mean(lap_variances) + 1e-10))
            
            # Too consistent can be suspicious
            if variance_consistency > 0.95:
                scores.append(0.6)
            elif variance_consistency < 0.5:
                scores.append(0.6)  # Too inconsistent
            else:
                scores.append(0.2)
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_blending(
        self,
        full_frames: List[np.ndarray],
        face_crops: List[np.ndarray]
    ) -> float:
        """Analyze face-background blending for artifacts."""
        scores = []
        
        # We need to detect the face boundary in full frames
        # For now, use basic color histogram comparison
        
        sample_count = min(10, len(full_frames), len(face_crops))
        
        for i in range(sample_count):
            frame = full_frames[i]
            face = face_crops[i]
            
            if frame is None or face is None:
                continue
            
            # Convert to HSV
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            
            # Compare color histograms
            frame_hist = cv2.calcHist([frame_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            face_hist = cv2.calcHist([face_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            
            cv2.normalize(frame_hist, frame_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(face_hist, face_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Correlation
            corr = cv2.compareHist(frame_hist, face_hist, cv2.HISTCMP_CORREL)
            
            # Too different colors might indicate compositing
            if corr < 0.3:
                scores.append(0.7)
            elif corr < 0.5:
                scores.append(0.5)
            else:
                scores.append(0.2)
            
            # Check for boundary artifacts
            # Look at edge sharpness around face region
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(face_gray, 100, 200)
            
            # High edge density at face boundary is suspicious
            edge_density = np.mean(edges) / 255
            
            if edge_density > 0.3:
                scores.append(0.6)  # Many edges - possible artifact
            else:
                scores.append(0.3)
        
        return np.mean(scores) if scores else 0.5
    
    def _neural_classify(self, frames: List[np.ndarray]) -> float:
        """Use neural network for deepfake classification."""
        if self.model is None:
            return 0.5  # Neutral if no model
        
        # Placeholder for actual neural network inference
        # In production, would:
        # 1. Preprocess frames (resize, normalize)
        # 2. Run through model (e.g., EfficientNet-B4)
        # 3. Average predictions across frames
        
        try:
            # Example pseudocode:
            # predictions = []
            # for frame in frames[:32]:  # Use 32 frames
            #     preprocessed = preprocess(frame)
            #     pred = self.model(preprocessed)
            #     predictions.append(pred)
            # return np.mean(predictions)
            
            return 0.5  # Placeholder
            
        except Exception as e:
            print(f"Neural classification error: {e}")
            return 0.5
    
    def _extract_forensic_features(
        self,
        frames: List[np.ndarray],
        landmarks: Optional[List[np.ndarray]]
    ) -> ForensicFeatures:
        """Extract detailed forensic features."""
        features = ForensicFeatures()
        
        if len(frames) == 0:
            return features
        
        # Sample frame for detailed analysis
        mid_frame = frames[len(frames) // 2]
        
        if mid_frame is not None and mid_frame.size > 0:
            gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY) if len(mid_frame.shape) == 3 else mid_frame
            
            # Laplacian variance
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            features.laplacian_variance = float(lap.var())
            
            # Color histogram entropy
            if len(mid_frame.shape) == 3:
                hist = cv2.calcHist([mid_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = hist.flatten()
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                features.color_histogram_entropy = float(-np.sum(hist * np.log2(hist)))
            
            # Noise pattern
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray.astype(float) - blur.astype(float)
            features.noise_pattern = float(np.std(noise))
        
        # Temporal features from landmarks
        if landmarks is not None and len(landmarks) > 10:
            valid_landmarks = [l for l in landmarks if l is not None]
            if len(valid_landmarks) > 1:
                diffs = [np.mean(np.abs(valid_landmarks[i] - valid_landmarks[i-1])) 
                        for i in range(1, len(valid_landmarks))]
                features.landmark_stability = float(1 / (1 + np.std(diffs)))
                features.temporal_consistency = float(1 - min(1, np.std(diffs) / 5))
        
        return features
    
    def _compute_final_score(self, result: DeepfakeResult) -> None:
        """Combine all signals into final probability."""
        scores = []
        weights = []
        
        if result.temporal_score > 0:
            scores.append(result.temporal_score)
            weights.append(self.weights['temporal'])
        
        if result.frequency_score > 0:
            scores.append(result.frequency_score)
            weights.append(self.weights['frequency'])
        
        if result.texture_score > 0:
            scores.append(result.texture_score)
            weights.append(self.weights['texture'])
        
        if result.blending_score > 0:
            scores.append(result.blending_score)
            weights.append(self.weights['blending'])
        
        if self.use_neural and result.neural_score > 0:
            scores.append(result.neural_score)
            weights.append(self.weights['neural'])
        
        if scores:
            total_weight = sum(weights)
            result.deepfake_probability = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        # Confidence based on signal agreement
        if len(scores) > 1:
            score_std = np.std(scores)
            result.confidence = 1 - min(1, score_std * 2)
        else:
            result.confidence = 0.5
        
        result.is_deepfake = result.deepfake_probability > self.DEEPFAKE_THRESHOLD
    
    def _compute_verdict(self, result: DeepfakeResult) -> None:
        """Compute human-readable verdict."""
        prob = result.deepfake_probability
        conf = result.confidence
        
        # Identify suspicious areas
        if result.temporal_score > 0.6:
            result.suspicious_regions.append("temporal_inconsistency")
        if result.frequency_score > 0.6:
            result.suspicious_regions.append("frequency_artifacts")
        if result.texture_score > 0.6:
            result.suspicious_regions.append("texture_anomalies")
        if result.blending_score > 0.6:
            result.suspicious_regions.append("blending_artifacts")
        
        # Verdict
        if prob < 0.3:
            result.verdict = "AUTHENTIC"
            result.explanation = "Video appears authentic. No significant manipulation detected."
        elif prob < 0.5:
            result.verdict = "LIKELY_AUTHENTIC"
            result.explanation = "Video likely authentic, minor anomalies detected."
        elif prob < 0.7:
            result.verdict = "SUSPICIOUS"
            result.explanation = f"Potential manipulation detected. Suspicious areas: {', '.join(result.suspicious_regions) or 'general'}"
        else:
            result.verdict = "LIKELY_DEEPFAKE"
            result.explanation = f"High probability of manipulation. Key indicators: {', '.join(result.suspicious_regions) or 'multiple signals'}"
        
        # Adjust if high confidence
        if conf > 0.8:
            if result.verdict == "SUSPICIOUS":
                result.verdict = "LIKELY_DEEPFAKE" if prob > 0.55 else "LIKELY_AUTHENTIC"


class QuickScreener:
    """
    Fast initial screening for obvious fakes/real content.
    
    Use this for quick first-pass before detailed analysis.
    """
    
    def __init__(self):
        self.min_frames = 5
    
    def screen(self, frames: List[np.ndarray]) -> Tuple[str, float]:
        """
        Quick screening of frames.
        
        Returns:
            Tuple of (verdict, confidence)
            Verdict: "PASS", "FAIL", "NEEDS_ANALYSIS"
        """
        if len(frames) < self.min_frames:
            return "NEEDS_ANALYSIS", 0.3
        
        # Quick checks
        scores = []
        
        # 1. Check for static content (photo)
        frame_diffs = []
        for i in range(1, min(10, len(frames))):
            diff = np.mean(np.abs(
                frames[i].astype(float) - frames[i-1].astype(float)
            ))
            frame_diffs.append(diff)
        
        if np.mean(frame_diffs) < 1.0:
            return "FAIL", 0.8  # Likely static image
        
        # 2. Check texture variance
        mid_frame = frames[len(frames) // 2]
        if len(mid_frame.shape) == 3:
            gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = mid_frame
        
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if lap_var < 10:
            return "FAIL", 0.7  # Very blurry / low texture
        elif lap_var > 2000:
            return "FAIL", 0.6  # Unusual texture pattern
        
        # 3. Basic color check
        if len(mid_frame.shape) == 3:
            hsv = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2HSV)
            sat_mean = np.mean(hsv[:, :, 1])
            
            if sat_mean > 200:
                scores.append(0.6)  # Oversaturated
        
        if not scores:
            return "PASS", 0.6
        elif np.mean(scores) > 0.5:
            return "NEEDS_ANALYSIS", np.mean(scores)
        else:
            return "PASS", 0.6


if __name__ == "__main__":
    print("Video Deepfake Detection Demo")
    print("=" * 40)
    
    # Create synthetic test frames
    frames = []
    for i in range(30):
        # Create a simple synthetic face-like pattern
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Background
        frame[:, :] = [200, 180, 160]  # Skin-like color
        
        # Add some texture
        noise = np.random.randint(-10, 10, (256, 256, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add slight motion (simulate head movement)
        shift = int(np.sin(i / 5) * 3)
        frame = np.roll(frame, shift, axis=1)
        
        frames.append(frame)
    
    # Create mock landmarks
    landmarks = []
    for i in range(30):
        # Simulate 68 facial landmarks with slight variation
        lm = np.random.randn(68, 2) * 0.5 + np.array([[100, 100]])
        lm += np.sin(i / 5) * 2  # Add temporal variation
        landmarks.append(lm)
    
    # Run detection
    detector = VideoDeepfakeDetector(use_neural=False)
    result = detector.analyze(frames, landmarks, frames)
    
    print(f"\nResults:")
    print(f"  Verdict: {result.verdict}")
    print(f"  Deepfake Probability: {result.deepfake_probability:.1%}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"\nComponent Scores:")
    print(f"  Temporal: {result.temporal_score:.2f}")
    print(f"  Frequency: {result.frequency_score:.2f}")
    print(f"  Texture: {result.texture_score:.2f}")
    print(f"  Blending: {result.blending_score:.2f}")
    
    if result.suspicious_regions:
        print(f"\nSuspicious Regions: {', '.join(result.suspicious_regions)}")
    
    print(f"\nExplanation: {result.explanation}")
    
    # Quick screener test
    print("\n" + "=" * 40)
    print("Quick Screener Test")
    screener = QuickScreener()
    verdict, conf = screener.screen(frames)
    print(f"Quick Screen: {verdict} (confidence: {conf:.1%})")
