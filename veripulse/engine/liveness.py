"""
Liveness Detection - Physiological and Active Challenge-based.

This module combines:
1. Physiological liveness - Analyzing rPPG signals, blinks, micro-movements
2. Active challenge liveness - User follows prompts (turn head, blink, etc.)

The goal is to distinguish real humans from:
- Static photos/printouts
- Replayed videos
- 3D masks/mannequins
- AI-generated deepfakes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from .rppg import RPPGResult, RPPGFeatures


class ChallengeType(Enum):
    """Types of active liveness challenges."""
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    NOD_UP = "nod_up"
    NOD_DOWN = "nod_down"
    SMILE = "smile"
    OPEN_MOUTH = "open_mouth"


@dataclass
class PhysioFeatures:
    """Physiological features for passive liveness."""
    
    # rPPG-based
    has_pulse: bool = False
    pulse_quality: float = 0.0      # 0-1 quality of rPPG signal
    bpm_in_range: bool = False      # Is BPM physiologically valid (40-200)?
    pulse_consistency: float = 0.0  # Consistency across ROIs
    
    # Blink-based
    blink_detected: bool = False
    blink_count: int = 0
    blink_rate: float = 0.0         # Blinks per minute
    blink_naturalness: float = 0.0  # Natural timing/duration
    
    # Micro-movement based
    micro_movements: float = 0.0    # Amount of natural micro-movements
    movement_pattern: float = 0.0   # Is movement pattern natural?
    
    # Texture analysis
    skin_texture: float = 0.0       # Natural skin texture score
    moire_pattern: float = 0.0      # Moire pattern detection (screens)
    reflection_pattern: float = 0.0 # Specular reflection analysis
    
    # Depth cues (from face mesh)
    depth_consistency: float = 0.0  # 3D structure consistency
    face_shape_score: float = 0.0   # Natural face shape
    
    # Combined score
    physio_score: float = 0.0


@dataclass
class ChallengeResult:
    """Result of a single active challenge."""
    challenge_type: ChallengeType = ChallengeType.BLINK
    passed: bool = False
    confidence: float = 0.0
    response_time_ms: float = 0.0
    details: str = ""


@dataclass
class ActiveChallengeResult:
    """Results from active challenge sequence."""
    challenges: List[ChallengeResult] = field(default_factory=list)
    
    challenges_passed: int = 0
    challenges_total: int = 0
    
    # Response analysis
    avg_response_time: float = 0.0  # Average response time in ms
    response_consistency: float = 0.0  # Consistency of responses
    
    # Combined score
    challenge_score: float = 0.0


@dataclass
class LivenessResult:
    """Complete liveness analysis result."""
    
    # Component results
    physio: PhysioFeatures = field(default_factory=PhysioFeatures)
    active: Optional[ActiveChallengeResult] = None
    
    # Final determination
    is_live: bool = False
    liveness_score: float = 0.0  # 0-1 overall liveness score
    confidence: float = 0.0
    
    # Threat indicators
    is_photo: bool = False
    is_video_replay: bool = False
    is_mask: bool = False
    is_synthetic: bool = False
    
    # Details
    verdict: str = "UNKNOWN"
    explanation: str = ""
    
    # Component scores (for API convenience)
    @property
    def texture_score(self) -> float:
        """Get texture analysis score."""
        return self.physio.skin_texture
    
    @property
    def blink_score(self) -> float:
        """Get blink analysis score."""
        return self.physio.blink_naturalness if self.physio.blink_detected else 0.0
    
    @property
    def motion_score(self) -> float:
        """Get motion analysis score."""
        return self.physio.movement_pattern


class LivenessDetector:
    """
    Multi-modal liveness detection system.
    
    Combines physiological signals with optional active challenges
    to determine if the subject is a live human.
    
    Example:
        >>> detector = LivenessDetector()
        >>> # With rPPG result from RPPGExtractor
        >>> liveness = detector.analyze(
        ...     rppg_result=rppg_result,
        ...     blink_events=blink_events,
        ...     head_poses=head_poses,
        ...     texture_features=texture_features
        ... )
        >>> print(f"Liveness: {liveness.verdict} ({liveness.liveness_score:.2f})")
    """
    
    # Thresholds
    MIN_VALID_HR = 40
    MAX_VALID_HR = 200
    MIN_BLINK_RATE = 5   # Per minute
    MAX_BLINK_RATE = 40  # Per minute
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize liveness detector.
        
        Args:
            strict_mode: If True, require all signals to be present and valid.
                        If False, work with available signals.
        """
        self.strict_mode = strict_mode
        
        # Weights for combining scores
        self.weights = {
            'pulse': 0.30,
            'blink': 0.20,
            'movement': 0.15,
            'texture': 0.20,
            'depth': 0.15
        }
    
    def analyze(
        self,
        rppg_result: Optional[RPPGResult] = None,
        blink_events: Optional[List[dict]] = None,
        head_poses: Optional[List[Tuple[float, float, float]]] = None,
        texture_features: Optional[dict] = None,
        challenge_results: Optional[List[ChallengeResult]] = None,
        fps: float = 30.0,
        duration_seconds: float = 10.0
    ) -> LivenessResult:
        """
        Perform complete liveness analysis.
        
        Args:
            rppg_result: Result from RPPGExtractor
            blink_events: List of detected blink events
            head_poses: List of (yaw, pitch, roll) tuples
            texture_features: Dict with texture analysis results
            challenge_results: Results from active challenges (optional)
            fps: Frame rate of input video
            duration_seconds: Duration of analyzed segment
            
        Returns:
            LivenessResult with comprehensive liveness assessment
        """
        result = LivenessResult()
        scores = []
        score_weights = []
        
        # 1. Analyze rPPG (pulse)
        if rppg_result is not None:
            pulse_score = self._analyze_pulse(rppg_result, result.physio)
            scores.append(pulse_score)
            score_weights.append(self.weights['pulse'])
        
        # 2. Analyze blinks
        if blink_events is not None:
            blink_score = self._analyze_blinks(
                blink_events, duration_seconds, result.physio
            )
            scores.append(blink_score)
            score_weights.append(self.weights['blink'])
        
        # 3. Analyze micro-movements
        if head_poses is not None and len(head_poses) > 10:
            movement_score = self._analyze_movements(head_poses, fps, result.physio)
            scores.append(movement_score)
            score_weights.append(self.weights['movement'])
        
        # 4. Analyze texture
        if texture_features is not None:
            texture_score = self._analyze_texture(texture_features, result.physio)
            scores.append(texture_score)
            score_weights.append(self.weights['texture'])
        
        # 5. Active challenges (if available)
        if challenge_results is not None and len(challenge_results) > 0:
            result.active = self._process_challenges(challenge_results)
            # Active challenges can boost or confirm the score
            if result.active.challenge_score > 0.5:
                scores.append(result.active.challenge_score)
                score_weights.append(0.2)  # Extra weight for active verification
        
        # Compute overall score
        if scores:
            total_weight = sum(score_weights)
            result.physio.physio_score = sum(
                s * w for s, w in zip(scores, score_weights)
            ) / total_weight
            result.liveness_score = result.physio.physio_score
        
        # Determine threats
        self._detect_threats(result)
        
        # Final verdict
        self._compute_verdict(result)
        
        return result
    
    def _analyze_pulse(
        self,
        rppg_result: RPPGResult,
        physio: PhysioFeatures
    ) -> float:
        """Analyze rPPG signal for liveness cues."""
        gf = rppg_result.global_features
        
        # Check for valid pulse
        bpm = rppg_result.get_best_bpm()
        physio.bpm_in_range = self.MIN_VALID_HR <= bpm <= self.MAX_VALID_HR
        
        # Quality metrics
        physio.pulse_quality = rppg_result.quality_score
        physio.pulse_consistency = gf.cross_roi_correlation
        
        # Has pulse if quality is sufficient
        physio.has_pulse = (
            physio.bpm_in_range and 
            physio.pulse_quality > 0.3 and
            gf.snr > 2
        )
        
        # Score calculation
        score_components = []
        
        # BPM validity
        if physio.bpm_in_range:
            score_components.append(0.3)
        
        # Signal quality (SNR)
        snr_score = min(1.0, max(0, gf.snr / 8))  # 8 dB = excellent
        score_components.append(snr_score * 0.25)
        
        # Periodicity
        score_components.append(gf.periodicity * 0.20)
        
        # Cross-ROI consistency (important for detecting replays)
        corr_score = (gf.cross_roi_correlation + 1) / 2  # Map to 0-1
        score_components.append(corr_score * 0.15)
        
        # BPM agreement across ROIs
        score_components.append(gf.bpm_agreement * 0.10)
        
        return sum(score_components)
    
    def _analyze_blinks(
        self,
        blink_events: List[dict],
        duration_seconds: float,
        physio: PhysioFeatures
    ) -> float:
        """Analyze blink patterns for liveness."""
        n_blinks = len(blink_events)
        
        physio.blink_detected = n_blinks > 0
        physio.blink_count = n_blinks
        
        if duration_seconds > 0:
            # Blinks per minute
            physio.blink_rate = (n_blinks / duration_seconds) * 60
        
        # Check if blink rate is physiologically plausible
        # Normal rate: 15-20 per minute, range: 5-40
        rate_in_range = self.MIN_BLINK_RATE <= physio.blink_rate <= self.MAX_BLINK_RATE
        
        # Blink naturalness (timing, duration)
        if n_blinks >= 2:
            # Analyze inter-blink intervals
            intervals = []
            durations = []
            
            for blink in blink_events:
                if 'interval' in blink:
                    intervals.append(blink['interval'])
                if 'duration' in blink:
                    durations.append(blink['duration'])
            
            # Natural blinks have some variability
            if intervals:
                interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
                # CV between 0.2-0.8 is natural
                interval_natural = 0.2 <= interval_cv <= 0.8
            else:
                interval_natural = True
            
            # Natural blink duration: 100-400ms
            if durations:
                duration_ok = all(100 <= d <= 400 for d in durations)
            else:
                duration_ok = True
            
            physio.blink_naturalness = 0.7 if (interval_natural and duration_ok) else 0.3
        else:
            physio.blink_naturalness = 0.5  # Not enough data
        
        # Score calculation
        score = 0.0
        
        # Has blinks
        if physio.blink_detected:
            score += 0.3
        
        # Rate in range
        if rate_in_range:
            score += 0.3
        
        # Naturalness
        score += physio.blink_naturalness * 0.4
        
        return score
    
    def _analyze_movements(
        self,
        head_poses: List[Tuple[float, float, float]],
        fps: float,
        physio: PhysioFeatures
    ) -> float:
        """Analyze micro-movements for liveness."""
        poses = np.array(head_poses)
        
        if len(poses) < 10:
            return 0.3  # Insufficient data
        
        # Compute frame-to-frame differences
        diffs = np.diff(poses, axis=0)
        
        # Micro-movement magnitude
        movement_magnitude = np.mean(np.abs(diffs))
        physio.micro_movements = float(movement_magnitude)
        
        # Expected micro-movements: 0.05-2.0 degrees per frame at 30fps
        expected_min = 0.02
        expected_max = 3.0
        
        movement_in_range = expected_min <= movement_magnitude <= expected_max
        
        # Movement pattern analysis
        # Natural movements have some autocorrelation (smooth)
        # but not too much (not robotic)
        autocorr = []
        for i in range(3):  # yaw, pitch, roll
            if len(diffs[:, i]) > 1:
                ac = np.corrcoef(diffs[:-1, i], diffs[1:, i])[0, 1]
                if not np.isnan(ac):
                    autocorr.append(ac)
        
        if autocorr:
            mean_autocorr = np.mean(autocorr)
            # Natural: autocorrelation between 0.2-0.8
            pattern_natural = 0.1 <= mean_autocorr <= 0.85
            physio.movement_pattern = float(mean_autocorr)
        else:
            pattern_natural = True
            physio.movement_pattern = 0.5
        
        # Score
        score = 0.0
        
        if movement_in_range:
            score += 0.5
        elif movement_magnitude > 0:
            score += 0.2  # Some movement
        
        if pattern_natural:
            score += 0.5
        
        return score
    
    def _analyze_texture(
        self,
        texture_features: dict,
        physio: PhysioFeatures
    ) -> float:
        """Analyze facial texture for liveness."""
        # Expected keys:
        # - laplacian_var: Edge/texture sharpness
        # - moire_score: Moire pattern detection (screens)
        # - reflection_consistency: Specular reflection pattern
        # - skin_smoothness: Natural skin texture
        
        score = 0.0
        
        # Laplacian variance (texture detail)
        lap_var = texture_features.get('laplacian_var', 0)
        # Real faces: typically 100-2000+, screens/prints: often different
        if lap_var > 50:
            score += 0.25
            physio.skin_texture = min(1.0, lap_var / 500)
        
        # Moire pattern (screen artifacts)
        moire = texture_features.get('moire_score', 0)
        physio.moire_pattern = moire
        if moire < 0.3:  # Low moire = likely real
            score += 0.25
        
        # Reflection consistency
        reflection = texture_features.get('reflection_consistency', 0.5)
        physio.reflection_pattern = reflection
        if reflection > 0.4:
            score += 0.25
        
        # Depth consistency (if available)
        depth = texture_features.get('depth_consistency', 0.5)
        physio.depth_consistency = depth
        if depth > 0.5:
            score += 0.25
        
        return score
    
    def _process_challenges(
        self,
        challenge_results: List[ChallengeResult]
    ) -> ActiveChallengeResult:
        """Process active challenge results."""
        result = ActiveChallengeResult()
        result.challenges = challenge_results
        result.challenges_total = len(challenge_results)
        result.challenges_passed = sum(1 for c in challenge_results if c.passed)
        
        # Response times
        response_times = [c.response_time_ms for c in challenge_results if c.response_time_ms > 0]
        if response_times:
            result.avg_response_time = np.mean(response_times)
            
            # Consistency: CV of response times
            if len(response_times) > 1:
                cv = np.std(response_times) / np.mean(response_times)
                result.response_consistency = 1 - min(1, cv)  # Lower CV = higher consistency
        
        # Score
        if result.challenges_total > 0:
            pass_rate = result.challenges_passed / result.challenges_total
            result.challenge_score = pass_rate * 0.7 + result.response_consistency * 0.3
        
        return result
    
    def _detect_threats(self, result: LivenessResult) -> None:
        """Detect specific threat types based on features."""
        physio = result.physio
        
        # Photo detection
        # - No pulse, no blinks, no micro-movements
        if (not physio.has_pulse and 
            physio.blink_count == 0 and 
            physio.micro_movements < 0.01):
            result.is_photo = True
        
        # Screen/video replay detection
        # - May have pulse (from video), but moire patterns
        # - Movement patterns may be too smooth or periodic
        if physio.moire_pattern > 0.5:
            result.is_video_replay = True
        
        # Mask detection
        # - Texture anomalies, depth inconsistencies
        if (physio.depth_consistency < 0.3 and 
            physio.skin_texture < 0.2):
            result.is_mask = True
        
        # Synthetic/deepfake indicators
        # - Often have perfect features but fail on temporal consistency
        if (physio.pulse_consistency < 0.3 and
            physio.blink_naturalness < 0.3):
            result.is_synthetic = True
    
    def _compute_verdict(self, result: LivenessResult) -> None:
        """Compute final verdict and explanation."""
        score = result.liveness_score
        
        # Determine if live
        result.is_live = score > 0.5 and not any([
            result.is_photo,
            result.is_video_replay,
            result.is_mask
        ])
        
        # Confidence based on score distance from threshold
        result.confidence = abs(score - 0.5) * 2  # 0-1
        
        # Verdict
        if result.is_photo:
            result.verdict = "PHOTO_DETECTED"
            result.explanation = "Static image detected - no physiological signals"
        elif result.is_video_replay:
            result.verdict = "VIDEO_REPLAY"
            result.explanation = "Screen artifacts detected - likely video replay"
        elif result.is_mask:
            result.verdict = "MASK_DETECTED"
            result.explanation = "Texture/depth anomalies suggest non-human surface"
        elif result.is_synthetic:
            result.verdict = "SYNTHETIC_SUSPECTED"
            result.explanation = "Temporal inconsistencies suggest synthetic content"
        elif result.is_live:
            result.verdict = "LIVE_HUMAN"
            result.explanation = "Physiological signals consistent with live human"
        else:
            result.verdict = "UNCERTAIN"
            result.explanation = "Insufficient signals for confident determination"


class ChallengeGenerator:
    """Generate and verify active liveness challenges."""
    
    # Challenge configurations
    CHALLENGE_CONFIG = {
        ChallengeType.BLINK: {
            'instruction': "Please blink your eyes",
            'timeout_ms': 5000,
            'detection_threshold': 0.3
        },
        ChallengeType.TURN_LEFT: {
            'instruction': "Please turn your head to the left",
            'timeout_ms': 5000,
            'angle_threshold': 20  # degrees
        },
        ChallengeType.TURN_RIGHT: {
            'instruction': "Please turn your head to the right",
            'timeout_ms': 5000,
            'angle_threshold': 20
        },
        ChallengeType.NOD_UP: {
            'instruction': "Please look up",
            'timeout_ms': 5000,
            'angle_threshold': 15
        },
        ChallengeType.NOD_DOWN: {
            'instruction': "Please look down",
            'timeout_ms': 5000,
            'angle_threshold': 15
        },
        ChallengeType.SMILE: {
            'instruction': "Please smile",
            'timeout_ms': 5000,
            'detection_threshold': 0.5
        },
        ChallengeType.OPEN_MOUTH: {
            'instruction': "Please open your mouth",
            'timeout_ms': 5000,
            'detection_threshold': 0.3
        }
    }
    
    def __init__(self, n_challenges: int = 3, randomize: bool = True):
        """
        Initialize challenge generator.
        
        Args:
            n_challenges: Number of challenges to generate per session
            randomize: If True, randomly select challenges
        """
        self.n_challenges = n_challenges
        self.randomize = randomize
    
    def generate_sequence(self) -> List[ChallengeType]:
        """Generate a sequence of challenges."""
        available = list(ChallengeType)
        
        if self.randomize:
            import random
            return random.sample(available, min(self.n_challenges, len(available)))
        else:
            return available[:self.n_challenges]
    
    def get_instruction(self, challenge: ChallengeType) -> str:
        """Get user-facing instruction for a challenge."""
        return self.CHALLENGE_CONFIG[challenge]['instruction']
    
    def verify_blink(
        self,
        ear_values: List[float],
        fps: float = 30.0
    ) -> ChallengeResult:
        """Verify blink challenge completion."""
        result = ChallengeResult(challenge_type=ChallengeType.BLINK)
        
        threshold = 0.25
        
        # Detect blink: EAR drops below threshold then recovers
        below_threshold = np.array(ear_values) < threshold
        
        # Find transitions
        transitions = np.diff(below_threshold.astype(int))
        closing = np.where(transitions == 1)[0]
        opening = np.where(transitions == -1)[0]
        
        if len(closing) > 0 and len(opening) > 0:
            # Check for valid blink (close then open)
            for close_idx in closing:
                opens_after = opening[opening > close_idx]
                if len(opens_after) > 0:
                    duration_frames = opens_after[0] - close_idx
                    duration_ms = (duration_frames / fps) * 1000
                    
                    # Valid blink: 100-400ms
                    if 50 <= duration_ms <= 500:
                        result.passed = True
                        result.confidence = 0.9
                        result.response_time_ms = (close_idx / fps) * 1000
                        result.details = f"Blink detected, duration: {duration_ms:.0f}ms"
                        break
        
        if not result.passed:
            result.confidence = 0.3
            result.details = "No valid blink detected"
        
        return result
    
    def verify_head_turn(
        self,
        head_poses: List[Tuple[float, float, float]],
        challenge: ChallengeType,
        fps: float = 30.0
    ) -> ChallengeResult:
        """Verify head turn challenge."""
        result = ChallengeResult(challenge_type=challenge)
        config = self.CHALLENGE_CONFIG[challenge]
        
        if len(head_poses) < 5:
            result.details = "Insufficient pose data"
            return result
        
        poses = np.array(head_poses)
        yaw = poses[:, 0]  # Left-right
        pitch = poses[:, 1]  # Up-down
        
        threshold = config['angle_threshold']
        
        if challenge == ChallengeType.TURN_LEFT:
            max_angle = np.max(yaw)  # Positive = left
            result.passed = max_angle > threshold
            result.confidence = min(1.0, max_angle / threshold)
            
        elif challenge == ChallengeType.TURN_RIGHT:
            min_angle = np.min(yaw)  # Negative = right
            result.passed = min_angle < -threshold
            result.confidence = min(1.0, abs(min_angle) / threshold)
            
        elif challenge == ChallengeType.NOD_UP:
            min_pitch = np.min(pitch)  # Negative = up
            result.passed = min_pitch < -threshold
            result.confidence = min(1.0, abs(min_pitch) / threshold)
            
        elif challenge == ChallengeType.NOD_DOWN:
            max_pitch = np.max(pitch)  # Positive = down
            result.passed = max_pitch > threshold
            result.confidence = min(1.0, max_pitch / threshold)
        
        # Find response time (when threshold first crossed)
        if result.passed:
            if challenge in [ChallengeType.TURN_LEFT, ChallengeType.NOD_DOWN]:
                angles = yaw if challenge == ChallengeType.TURN_LEFT else pitch
                first_cross = np.argmax(angles > threshold)
            else:
                angles = yaw if challenge == ChallengeType.TURN_RIGHT else pitch
                first_cross = np.argmax(angles < -threshold)
            
            result.response_time_ms = (first_cross / fps) * 1000
        
        result.details = f"Challenge {'passed' if result.passed else 'failed'}"
        
        return result


if __name__ == "__main__":
    # Demo
    print("Liveness Detection Demo")
    print("=" * 40)
    
    # Simulate liveness analysis with mock data
    detector = LivenessDetector()
    
    # Mock rPPG result
    from .rppg import RPPGResult, RPPGFeatures
    rppg = RPPGResult()
    rppg.global_features = RPPGFeatures(
        bpm=72,
        bpm_confidence=0.8,
        snr=6.5,
        periodicity=0.7,
        cross_roi_correlation=0.85
    )
    rppg.quality_score = 0.7
    
    # Mock blinks
    blinks = [
        {'time': 1.0, 'duration': 150, 'interval': 0},
        {'time': 3.5, 'duration': 180, 'interval': 2500},
        {'time': 7.2, 'duration': 160, 'interval': 3700}
    ]
    
    # Mock head poses (10 seconds at 30fps)
    import random
    head_poses = []
    base_pose = [0, 0, 0]
    for _ in range(300):
        pose = [
            base_pose[0] + random.gauss(0, 0.5),
            base_pose[1] + random.gauss(0, 0.3),
            base_pose[2] + random.gauss(0, 0.2)
        ]
        head_poses.append(tuple(pose))
        base_pose = list(pose)
    
    # Mock texture
    texture = {
        'laplacian_var': 450,
        'moire_score': 0.1,
        'reflection_consistency': 0.7,
        'depth_consistency': 0.8
    }
    
    # Analyze
    result = detector.analyze(
        rppg_result=rppg,
        blink_events=blinks,
        head_poses=head_poses,
        texture_features=texture,
        duration_seconds=10.0
    )
    
    print(f"\nVerdict: {result.verdict}")
    print(f"Liveness Score: {result.liveness_score:.2f}")
    print(f"Is Live: {result.is_live}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nExplanation: {result.explanation}")
    
    print(f"\nPhysiological Features:")
    print(f"  - Has Pulse: {result.physio.has_pulse}")
    print(f"  - Blink Count: {result.physio.blink_count}")
    print(f"  - Micro-movements: {result.physio.micro_movements:.4f}")
    print(f"  - Skin Texture: {result.physio.skin_texture:.2f}")
