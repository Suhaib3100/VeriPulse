"""
VeriPulse Engine - Production-ready multimodal liveness and deepfake detection.

This package provides:
- Video pipeline for liveness detection and rPPG analysis
- Audio pipeline for voice authenticity detection
- Deepfake detection for both video and audio
- Unified trust scoring and fusion
- REST API for integration

Quick Start:
    # Analyze a file
    from veripulse.api import AnalysisEngine
    engine = AnalysisEngine()
    result = engine.analyze_media_file("video.mp4")
    print(f"Trust: {result.trust_score:.1%}")
    
    # Run API server
    from veripulse.api import run_server
    run_server(host="0.0.0.0", port=8000)
"""

__version__ = "1.0.0"
__author__ = "VeriPulse Team"

# Top-level imports for convenience
from .engine import (
    VideoSource,
    FaceTracker,
    RPPGExtractor,
    LivenessDetector,
    VideoDeepfakeDetector,
    AudioSource,
    AudioFeatureExtractor,
    AudioDeepfakeDetector,
    MultimodalFusion,
    MultimodalTrustResult,
    fuse_multimodal_scores,
)

__all__ = [
    "__version__",
    "VideoSource",
    "FaceTracker",
    "RPPGExtractor",
    "LivenessDetector",
    "VideoDeepfakeDetector",
    "AudioSource",
    "AudioFeatureExtractor",
    "AudioDeepfakeDetector",
    "MultimodalFusion",
    "MultimodalTrustResult",
    "fuse_multimodal_scores",
]
