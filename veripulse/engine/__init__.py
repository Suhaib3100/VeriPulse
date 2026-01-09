"""
VeriPulse Engine - Core analysis components.

This module exports all engine classes and functions for:
- Video pipeline (VideoSource, FaceTracker, VideoFrameBatch)
- rPPG extraction (RPPGExtractor, RPPGResult)
- Liveness detection (LivenessDetector, LivenessResult)
- Video deepfake detection (VideoDeepfakeDetector)
- Audio pipeline (AudioSource, AudioFeatureExtractor)
- Audio deepfake detection (AudioDeepfakeDetector)
- Multimodal fusion (MultimodalFusion, fuse_multimodal_scores)
"""

# Video Pipeline
from .video_pipeline import (
    VideoSource,
    VideoSourceType,
    FaceTracker,
    FaceDetection,
    VideoFrameBatch,
    Frame,
)

# rPPG
from .rppg import (
    RPPGExtractor,
    RPPGResult,
    RPPGFeatures,
    RPPGMethod,
)

# Liveness
from .liveness import (
    LivenessDetector,
    LivenessResult,
    PhysioFeatures,
    ChallengeGenerator,
    ChallengeType,
    ActiveChallengeResult,
)

# Video Deepfake
from .deepfake_video import (
    VideoDeepfakeDetector,
    DeepfakeResult,
    QuickScreener,
    ForensicFeatures,
)

# Audio Pipeline
from .audio_pipeline import (
    AudioSource,
    AudioSourceType,
    AudioChunk,
    AudioFeatures,
    AudioFeatureExtractor,
    VoiceActivityDetector,
)

# Audio Deepfake
from .deepfake_audio import (
    AudioDeepfakeDetector,
    AudioDeepfakeResult,
    AudioForensicFeatures,
)

# Fusion
from .fusion import (
    MultimodalFusion,
    MultimodalTrustResult,
    FusionStrategy,
    TrustLevel,
    ThreatType,
    fuse_multimodal_scores,
)


__all__ = [
    # Video Pipeline
    "VideoSource",
    "VideoSourceType",
    "FaceTracker",
    "FaceDetection",
    "VideoFrameBatch",
    "Frame",
    
    # rPPG
    "RPPGExtractor",
    "RPPGResult",
    "RPPGFeatures",
    "RPPGMethod",
    
    # Liveness
    "LivenessDetector",
    "LivenessResult",
    "PhysioFeatures",
    "ChallengeGenerator",
    "ChallengeType",
    "ActiveChallengeResult",
    
    # Video Deepfake
    "VideoDeepfakeDetector",
    "DeepfakeResult",
    "QuickScreener",
    "ForensicFeatures",
    
    # Audio Pipeline
    "AudioSource",
    "AudioSourceType",
    "AudioChunk",
    "AudioFeatures",
    "AudioFeatureExtractor",
    "VoiceActivityDetector",
    
    # Audio Deepfake
    "AudioDeepfakeDetector",
    "AudioDeepfakeResult",
    "AudioForensicFeatures",
    
    # Fusion
    "MultimodalFusion",
    "MultimodalTrustResult",
    "FusionStrategy",
    "TrustLevel",
    "ThreatType",
    "fuse_multimodal_scores",
]
