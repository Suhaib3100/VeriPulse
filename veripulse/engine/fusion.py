"""
Multimodal Fusion - Combining video and audio analysis for final trust score.

This module provides:
- MultimodalTrustResult: Unified result from all analysis components
- fuse_multimodal_scores(): Combine liveness, video deepfake, audio deepfake
- Configurable fusion strategies (weighted average, Bayesian, learned)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

from .liveness import LivenessResult
from .deepfake_video import DeepfakeResult as VideoDeepfakeResult
from .deepfake_audio import AudioDeepfakeResult
from .rppg import RPPGResult


class FusionStrategy(Enum):
    """Strategies for combining multimodal scores."""
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN = "bayesian"
    MIN_SCORE = "min_score"       # Most suspicious signal wins
    MAX_CONFIDENCE = "max_confidence"  # Weight by confidence
    LEARNED = "learned"           # Neural fusion (placeholder)


class TrustLevel(Enum):
    """Final trust level categories."""
    VERY_HIGH = "very_high"       # >0.9 trust
    HIGH = "high"                 # 0.7-0.9 trust
    MEDIUM = "medium"             # 0.5-0.7 trust
    LOW = "low"                   # 0.3-0.5 trust
    VERY_LOW = "very_low"         # <0.3 trust


class ThreatType(Enum):
    """Types of threats detected."""
    NONE = "none"
    PHOTO_ATTACK = "photo_attack"
    VIDEO_REPLAY = "video_replay"
    MASK_ATTACK = "mask_attack"
    VIDEO_DEEPFAKE = "video_deepfake"
    AUDIO_DEEPFAKE = "audio_deepfake"
    AV_MISMATCH = "av_mismatch"   # Audio-video desync/mismatch
    UNKNOWN = "unknown"


@dataclass
class ComponentScore:
    """Score from a single analysis component."""
    name: str
    score: float           # 0-1, higher = more trustworthy
    confidence: float      # Confidence in this score
    weight: float = 1.0    # Weight in fusion
    available: bool = True # Whether this component was run
    details: str = ""


@dataclass 
class MultimodalTrustResult:
    """
    Unified trust assessment from all analysis modalities.
    
    This is the final output of the VeriPulse Engine.
    """
    
    # === Final Verdict ===
    trust_score: float = 0.0       # 0-1, overall trust score
    trust_level: TrustLevel = TrustLevel.MEDIUM
    is_trustworthy: bool = False   # Final boolean decision
    
    # === Component Scores ===
    # Each component contributes to the final score
    liveness_score: float = 0.0    # From liveness detection
    video_authenticity: float = 0.0 # From video deepfake detection
    audio_authenticity: float = 0.0 # From audio deepfake detection
    rppg_quality: float = 0.0      # From rPPG signal quality
    av_consistency: float = 0.0    # Audio-video sync/consistency
    
    # === Component Results (detailed) ===
    liveness_result: Optional[LivenessResult] = None
    video_deepfake_result: Optional[VideoDeepfakeResult] = None
    audio_deepfake_result: Optional[AudioDeepfakeResult] = None
    rppg_result: Optional[RPPGResult] = None
    
    # === Confidence & Quality ===
    overall_confidence: float = 0.0  # Confidence in final verdict
    signal_quality: float = 0.0      # Quality of input signals
    
    # === Threat Assessment ===
    primary_threat: ThreatType = ThreatType.NONE
    threat_indicators: List[str] = field(default_factory=list)
    
    # === Metadata ===
    analysis_duration_ms: float = 0.0
    video_duration_seconds: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # === Human-Readable Output ===
    verdict: str = "UNKNOWN"
    explanation: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trust_score": self.trust_score,
            "trust_level": self.trust_level.value,
            "is_trustworthy": self.is_trustworthy,
            "verdict": self.verdict,
            "explanation": self.explanation,
            "confidence": self.overall_confidence,
            "components": {
                "liveness": self.liveness_score,
                "video_authenticity": self.video_authenticity,
                "audio_authenticity": self.audio_authenticity,
                "rppg_quality": self.rppg_quality,
                "av_consistency": self.av_consistency
            },
            "threats": {
                "primary": self.primary_threat.value,
                "indicators": self.threat_indicators
            },
            "metadata": {
                "has_video": self.has_video,
                "has_audio": self.has_audio,
                "video_duration_seconds": self.video_duration_seconds,
                "analysis_duration_ms": self.analysis_duration_ms,
                "timestamp": self.timestamp
            },
            "recommendations": self.recommendations
        }


class MultimodalFusion:
    """
    Fuse results from multiple analysis modalities into unified trust score.
    
    Example:
        >>> fusion = MultimodalFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)
        >>> result = fusion.fuse(
        ...     liveness=liveness_result,
        ...     video_deepfake=video_result,
        ...     audio_deepfake=audio_result,
        ...     rppg=rppg_result
        ... )
        >>> print(f"Trust: {result.trust_score:.1%} - {result.verdict}")
    """
    
    # Default weights for different modalities
    DEFAULT_WEIGHTS = {
        'liveness': 0.30,
        'video_deepfake': 0.30,
        'audio_deepfake': 0.25,
        'rppg': 0.10,
        'av_consistency': 0.05
    }
    
    # Trust thresholds
    TRUST_THRESHOLDS = {
        TrustLevel.VERY_HIGH: 0.9,
        TrustLevel.HIGH: 0.7,
        TrustLevel.MEDIUM: 0.5,
        TrustLevel.LOW: 0.3,
        TrustLevel.VERY_LOW: 0.0
    }
    
    def __init__(
        self,
        strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        weights: Optional[Dict[str, float]] = None,
        trust_threshold: float = 0.5
    ):
        """
        Initialize fusion module.
        
        Args:
            strategy: Fusion strategy to use
            weights: Custom weights for each modality
            trust_threshold: Threshold for is_trustworthy decision
        """
        self.strategy = strategy
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.trust_threshold = trust_threshold
    
    def fuse(
        self,
        liveness: Optional[LivenessResult] = None,
        video_deepfake: Optional[VideoDeepfakeResult] = None,
        audio_deepfake: Optional[AudioDeepfakeResult] = None,
        rppg: Optional[RPPGResult] = None,
        video_duration: float = 0.0,
        analysis_time_ms: float = 0.0
    ) -> MultimodalTrustResult:
        """
        Fuse all analysis results into unified trust score.
        
        Args:
            liveness: Liveness detection result
            video_deepfake: Video deepfake detection result
            audio_deepfake: Audio deepfake detection result
            rppg: rPPG analysis result
            video_duration: Duration of analyzed video
            analysis_time_ms: Total analysis time
            
        Returns:
            MultimodalTrustResult with unified assessment
        """
        result = MultimodalTrustResult()
        result.video_duration_seconds = video_duration
        result.analysis_duration_ms = analysis_time_ms
        
        # Store component results
        result.liveness_result = liveness
        result.video_deepfake_result = video_deepfake
        result.audio_deepfake_result = audio_deepfake
        result.rppg_result = rppg
        
        # Determine what modalities are available
        result.has_video = (liveness is not None or video_deepfake is not None)
        result.has_audio = (audio_deepfake is not None)
        
        # Collect component scores
        components = self._collect_component_scores(
            liveness, video_deepfake, audio_deepfake, rppg
        )
        
        # Apply fusion strategy
        if self.strategy == FusionStrategy.WEIGHTED_AVERAGE:
            self._fuse_weighted_average(components, result)
        elif self.strategy == FusionStrategy.MIN_SCORE:
            self._fuse_min_score(components, result)
        elif self.strategy == FusionStrategy.MAX_CONFIDENCE:
            self._fuse_max_confidence(components, result)
        elif self.strategy == FusionStrategy.BAYESIAN:
            self._fuse_bayesian(components, result)
        else:
            self._fuse_weighted_average(components, result)
        
        # Audio-video consistency check
        if result.has_video and result.has_audio:
            result.av_consistency = self._compute_av_consistency(
                video_deepfake, audio_deepfake
            )
        
        # Determine trust level
        result.trust_level = self._determine_trust_level(result.trust_score)
        result.is_trustworthy = result.trust_score >= self.trust_threshold
        
        # Threat assessment
        self._assess_threats(result)
        
        # Generate verdict and explanation
        self._generate_verdict(result)
        
        return result
    
    def _collect_component_scores(
        self,
        liveness: Optional[LivenessResult],
        video_deepfake: Optional[VideoDeepfakeResult],
        audio_deepfake: Optional[AudioDeepfakeResult],
        rppg: Optional[RPPGResult]
    ) -> List[ComponentScore]:
        """Collect and normalize scores from all components."""
        components = []
        
        # Liveness (higher = more live = more trust)
        if liveness is not None:
            components.append(ComponentScore(
                name='liveness',
                score=liveness.liveness_score,
                confidence=liveness.confidence,
                weight=self.weights.get('liveness', 0.3),
                available=True,
                details=liveness.verdict
            ))
        
        # Video deepfake (lower probability = more trust)
        if video_deepfake is not None:
            # Convert: low deepfake probability = high trust
            authenticity = 1.0 - video_deepfake.deepfake_probability
            components.append(ComponentScore(
                name='video_deepfake',
                score=authenticity,
                confidence=video_deepfake.confidence,
                weight=self.weights.get('video_deepfake', 0.3),
                available=True,
                details=video_deepfake.verdict
            ))
        
        # Audio deepfake (lower probability = more trust)
        if audio_deepfake is not None:
            authenticity = 1.0 - audio_deepfake.synthetic_probability
            components.append(ComponentScore(
                name='audio_deepfake',
                score=authenticity,
                confidence=audio_deepfake.confidence,
                weight=self.weights.get('audio_deepfake', 0.25),
                available=True,
                details=audio_deepfake.verdict
            ))
        
        # rPPG quality (higher = more trust in physiological signals)
        if rppg is not None:
            components.append(ComponentScore(
                name='rppg',
                score=rppg.quality_score,
                confidence=rppg.global_features.bpm_confidence,
                weight=self.weights.get('rppg', 0.1),
                available=True,
                details=f"BPM: {rppg.get_best_bpm():.0f}"
            ))
        
        return components
    
    def _fuse_weighted_average(
        self,
        components: List[ComponentScore],
        result: MultimodalTrustResult
    ) -> None:
        """Weighted average fusion."""
        if not components:
            result.trust_score = 0.5
            result.overall_confidence = 0.0
            return
        
        total_weight = sum(c.weight for c in components if c.available)
        
        if total_weight == 0:
            result.trust_score = 0.5
            return
        
        # Weighted sum of scores
        weighted_sum = sum(c.score * c.weight for c in components if c.available)
        result.trust_score = weighted_sum / total_weight
        
        # Confidence: weighted average of component confidences
        confidence_sum = sum(c.confidence * c.weight for c in components if c.available)
        result.overall_confidence = confidence_sum / total_weight
        
        # Store individual component scores
        for c in components:
            if c.name == 'liveness':
                result.liveness_score = c.score
            elif c.name == 'video_deepfake':
                result.video_authenticity = c.score
            elif c.name == 'audio_deepfake':
                result.audio_authenticity = c.score
            elif c.name == 'rppg':
                result.rppg_quality = c.score
    
    def _fuse_min_score(
        self,
        components: List[ComponentScore],
        result: MultimodalTrustResult
    ) -> None:
        """Conservative fusion: take minimum score."""
        if not components:
            result.trust_score = 0.5
            return
        
        available = [c for c in components if c.available]
        
        if not available:
            result.trust_score = 0.5
            return
        
        # Find minimum score (most suspicious signal)
        min_component = min(available, key=lambda c: c.score)
        result.trust_score = min_component.score
        result.overall_confidence = min_component.confidence
        
        # Still store all component scores
        for c in components:
            if c.name == 'liveness':
                result.liveness_score = c.score
            elif c.name == 'video_deepfake':
                result.video_authenticity = c.score
            elif c.name == 'audio_deepfake':
                result.audio_authenticity = c.score
            elif c.name == 'rppg':
                result.rppg_quality = c.score
    
    def _fuse_max_confidence(
        self,
        components: List[ComponentScore],
        result: MultimodalTrustResult
    ) -> None:
        """Weight by confidence."""
        if not components:
            result.trust_score = 0.5
            return
        
        available = [c for c in components if c.available and c.confidence > 0]
        
        if not available:
            # Fallback to regular weighted average
            self._fuse_weighted_average(components, result)
            return
        
        # Weight by confidence
        total_conf = sum(c.confidence * c.weight for c in available)
        
        if total_conf == 0:
            self._fuse_weighted_average(components, result)
            return
        
        weighted_sum = sum(c.score * c.confidence * c.weight for c in available)
        result.trust_score = weighted_sum / total_conf
        result.overall_confidence = max(c.confidence for c in available)
        
        # Store component scores
        for c in components:
            if c.name == 'liveness':
                result.liveness_score = c.score
            elif c.name == 'video_deepfake':
                result.video_authenticity = c.score
            elif c.name == 'audio_deepfake':
                result.audio_authenticity = c.score
            elif c.name == 'rppg':
                result.rppg_quality = c.score
    
    def _fuse_bayesian(
        self,
        components: List[ComponentScore],
        result: MultimodalTrustResult
    ) -> None:
        """Bayesian fusion using log-odds."""
        if not components:
            result.trust_score = 0.5
            return
        
        available = [c for c in components if c.available]
        
        if not available:
            result.trust_score = 0.5
            return
        
        # Prior: 0.5 (neutral)
        prior_log_odds = 0
        
        # Combine evidence using log-odds
        combined_log_odds = prior_log_odds
        
        for c in available:
            # Clamp scores to avoid log(0)
            score = np.clip(c.score, 0.01, 0.99)
            
            # Log-odds ratio
            log_odds = np.log(score / (1 - score))
            
            # Weight by component weight and confidence
            combined_log_odds += log_odds * c.weight * c.confidence
        
        # Convert back to probability
        result.trust_score = 1 / (1 + np.exp(-combined_log_odds))
        result.overall_confidence = np.mean([c.confidence for c in available])
        
        # Store component scores
        for c in components:
            if c.name == 'liveness':
                result.liveness_score = c.score
            elif c.name == 'video_deepfake':
                result.video_authenticity = c.score
            elif c.name == 'audio_deepfake':
                result.audio_authenticity = c.score
            elif c.name == 'rppg':
                result.rppg_quality = c.score
    
    def _compute_av_consistency(
        self,
        video_result: Optional[VideoDeepfakeResult],
        audio_result: Optional[AudioDeepfakeResult]
    ) -> float:
        """Check audio-video consistency."""
        if video_result is None or audio_result is None:
            return 0.5  # Neutral
        
        # Simple consistency: both should agree
        video_auth = 1 - video_result.deepfake_probability
        audio_auth = 1 - audio_result.synthetic_probability
        
        # High consistency = similar authenticity scores
        diff = abs(video_auth - audio_auth)
        consistency = 1 - diff
        
        return consistency
    
    def _determine_trust_level(self, trust_score: float) -> TrustLevel:
        """Convert trust score to categorical level."""
        for level, threshold in self.TRUST_THRESHOLDS.items():
            if trust_score >= threshold:
                return level
        return TrustLevel.VERY_LOW
    
    def _assess_threats(self, result: MultimodalTrustResult) -> None:
        """Identify specific threats based on component results."""
        threats = []
        
        # Check liveness threats
        if result.liveness_result:
            lr = result.liveness_result
            if lr.is_photo:
                threats.append(("photo_attack", ThreatType.PHOTO_ATTACK, 0.9))
            if lr.is_video_replay:
                threats.append(("video_replay", ThreatType.VIDEO_REPLAY, 0.8))
            if lr.is_mask:
                threats.append(("mask_attack", ThreatType.MASK_ATTACK, 0.85))
            if lr.is_synthetic:
                threats.append(("synthetic_face", ThreatType.VIDEO_DEEPFAKE, 0.7))
        
        # Check video deepfake
        if result.video_deepfake_result:
            vr = result.video_deepfake_result
            if vr.is_deepfake:
                threats.append(("video_deepfake", ThreatType.VIDEO_DEEPFAKE, 
                              vr.deepfake_probability))
        
        # Check audio deepfake
        if result.audio_deepfake_result:
            ar = result.audio_deepfake_result
            if ar.is_synthetic:
                threats.append(("audio_synthetic", ThreatType.AUDIO_DEEPFAKE,
                              ar.synthetic_probability))
        
        # Check AV consistency
        if result.av_consistency < 0.5:
            threats.append(("av_mismatch", ThreatType.AV_MISMATCH, 
                          1 - result.av_consistency))
        
        # Set primary threat (highest severity)
        if threats:
            threats.sort(key=lambda x: x[2], reverse=True)
            result.primary_threat = threats[0][1]
            result.threat_indicators = [t[0] for t in threats]
        else:
            result.primary_threat = ThreatType.NONE
    
    def _generate_verdict(self, result: MultimodalTrustResult) -> None:
        """Generate human-readable verdict and explanation."""
        score = result.trust_score
        level = result.trust_level
        
        # Verdict
        if level == TrustLevel.VERY_HIGH:
            result.verdict = "HIGHLY_TRUSTWORTHY"
        elif level == TrustLevel.HIGH:
            result.verdict = "TRUSTWORTHY"
        elif level == TrustLevel.MEDIUM:
            result.verdict = "UNCERTAIN"
        elif level == TrustLevel.LOW:
            result.verdict = "SUSPICIOUS"
        else:
            result.verdict = "NOT_TRUSTWORTHY"
        
        # Explanation
        explanations = []
        
        if result.liveness_result:
            explanations.append(f"Liveness: {result.liveness_result.verdict}")
        
        if result.video_deepfake_result:
            explanations.append(f"Video: {result.video_deepfake_result.verdict}")
        
        if result.audio_deepfake_result:
            explanations.append(f"Audio: {result.audio_deepfake_result.verdict}")
        
        if result.threat_indicators:
            explanations.append(f"Threats detected: {', '.join(result.threat_indicators)}")
        
        result.explanation = " | ".join(explanations) if explanations else "Analysis complete."
        
        # Recommendations
        recommendations = []
        
        if score < 0.3:
            recommendations.append("Do not trust this media. Strong indicators of manipulation.")
        elif score < 0.5:
            recommendations.append("Exercise caution. Some indicators of potential manipulation.")
        elif score < 0.7:
            recommendations.append("Proceed with moderate caution. Verify through other means if critical.")
        else:
            recommendations.append("Media appears authentic. Standard verification passed.")
        
        if result.primary_threat != ThreatType.NONE:
            if result.primary_threat == ThreatType.VIDEO_DEEPFAKE:
                recommendations.append("Video shows signs of face manipulation or synthesis.")
            elif result.primary_threat == ThreatType.AUDIO_DEEPFAKE:
                recommendations.append("Audio shows signs of voice synthesis or cloning.")
            elif result.primary_threat == ThreatType.PHOTO_ATTACK:
                recommendations.append("Detected possible presentation of a photo instead of live person.")
            elif result.primary_threat == ThreatType.VIDEO_REPLAY:
                recommendations.append("Detected possible screen replay attack.")
        
        result.recommendations = recommendations


def fuse_multimodal_scores(
    liveness: Optional[LivenessResult] = None,
    video_deepfake: Optional[VideoDeepfakeResult] = None,
    audio_deepfake: Optional[AudioDeepfakeResult] = None,
    rppg: Optional[RPPGResult] = None,
    strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
    **kwargs
) -> MultimodalTrustResult:
    """
    Convenience function to fuse multimodal analysis results.
    
    Example:
        >>> result = fuse_multimodal_scores(
        ...     liveness=liveness_result,
        ...     video_deepfake=video_result,
        ...     audio_deepfake=audio_result
        ... )
    """
    fusion = MultimodalFusion(strategy=strategy)
    return fusion.fuse(
        liveness=liveness,
        video_deepfake=video_deepfake,
        audio_deepfake=audio_deepfake,
        rppg=rppg,
        **kwargs
    )


if __name__ == "__main__":
    print("Multimodal Fusion Demo")
    print("=" * 40)
    
    # Create mock results for testing
    from .liveness import LivenessResult, PhysioFeatures
    from .deepfake_video import DeepfakeResult as VideoDeepfakeResult
    from .deepfake_audio import AudioDeepfakeResult
    from .rppg import RPPGResult, RPPGFeatures
    
    # Mock liveness result (good)
    liveness = LivenessResult()
    liveness.is_live = True
    liveness.liveness_score = 0.85
    liveness.confidence = 0.9
    liveness.verdict = "LIVE_HUMAN"
    
    # Mock video deepfake (clean)
    video_df = VideoDeepfakeResult()
    video_df.is_deepfake = False
    video_df.deepfake_probability = 0.15
    video_df.confidence = 0.8
    video_df.verdict = "AUTHENTIC"
    
    # Mock audio deepfake (slightly suspicious)
    audio_df = AudioDeepfakeResult()
    audio_df.is_synthetic = False
    audio_df.synthetic_probability = 0.35
    audio_df.confidence = 0.7
    audio_df.verdict = "LIKELY_AUTHENTIC"
    
    # Mock rPPG
    rppg = RPPGResult()
    rppg.global_features = RPPGFeatures(bpm=72, bpm_confidence=0.8)
    rppg.quality_score = 0.7
    
    # Fuse results
    fusion = MultimodalFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)
    result = fusion.fuse(
        liveness=liveness,
        video_deepfake=video_df,
        audio_deepfake=audio_df,
        rppg=rppg,
        video_duration=10.0,
        analysis_time_ms=1500
    )
    
    print(f"\n{'='*50}")
    print(f"VERIPULSE TRUST ASSESSMENT")
    print(f"{'='*50}")
    print(f"\nFinal Verdict: {result.verdict}")
    print(f"Trust Score: {result.trust_score:.1%}")
    print(f"Trust Level: {result.trust_level.value.upper()}")
    print(f"Confidence: {result.overall_confidence:.1%}")
    
    print(f"\n--- Component Scores ---")
    print(f"  Liveness: {result.liveness_score:.2f}")
    print(f"  Video Authenticity: {result.video_authenticity:.2f}")
    print(f"  Audio Authenticity: {result.audio_authenticity:.2f}")
    print(f"  rPPG Quality: {result.rppg_quality:.2f}")
    print(f"  AV Consistency: {result.av_consistency:.2f}")
    
    print(f"\n--- Threat Assessment ---")
    print(f"  Primary Threat: {result.primary_threat.value}")
    print(f"  Indicators: {', '.join(result.threat_indicators) or 'None'}")
    
    print(f"\n--- Explanation ---")
    print(f"  {result.explanation}")
    
    print(f"\n--- Recommendations ---")
    for rec in result.recommendations:
        print(f"  • {rec}")
    
    print(f"\n--- Metadata ---")
    print(f"  Has Video: {result.has_video}")
    print(f"  Has Audio: {result.has_audio}")
    print(f"  Analysis Time: {result.analysis_duration_ms:.0f}ms")
    print(f"  Timestamp: {result.timestamp}")
    
    # Test dictionary export
    print(f"\n--- JSON Export ---")
    import json
    print(json.dumps(result.to_dict(), indent=2))
