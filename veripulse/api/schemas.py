"""
API Schemas - Pydantic models for request/response validation.

These schemas define the API contract for the VeriPulse Engine.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


# ============================================================
# Enums
# ============================================================

class TrustLevelEnum(str, Enum):
    """Trust level categories."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ThreatTypeEnum(str, Enum):
    """Types of threats."""
    NONE = "none"
    PHOTO_ATTACK = "photo_attack"
    VIDEO_REPLAY = "video_replay"
    MASK_ATTACK = "mask_attack"
    VIDEO_DEEPFAKE = "video_deepfake"
    AUDIO_DEEPFAKE = "audio_deepfake"
    AV_MISMATCH = "av_mismatch"
    UNKNOWN = "unknown"


class AnalysisMode(str, Enum):
    """Analysis modes."""
    QUICK = "quick"           # Fast screening
    STANDARD = "standard"     # Balanced
    THOROUGH = "thorough"     # Deep analysis


# ============================================================
# Component Schemas
# ============================================================

class ComponentScore(BaseModel):
    """Score from a single analysis component."""
    
    name: str
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    details: Optional[str] = None


class LivenessDetails(BaseModel):
    """Detailed liveness analysis results."""
    
    is_live: bool = False
    score: float = Field(ge=0, le=1, default=0.0)
    confidence: float = Field(ge=0, le=1, default=0.0)
    
    has_pulse: bool = False
    pulse_quality: float = Field(ge=0, le=1, default=0.0)
    blink_detected: bool = False
    blink_count: int = 0
    micro_movements: float = 0.0
    
    verdict: str = "UNKNOWN"
    explanation: str = ""


class VideoDeepfakeDetails(BaseModel):
    """Detailed video deepfake analysis results."""
    
    is_deepfake: bool = False
    probability: float = Field(ge=0, le=1, default=0.0)
    confidence: float = Field(ge=0, le=1, default=0.0)
    
    temporal_score: float = Field(ge=0, le=1, default=0.0)
    frequency_score: float = Field(ge=0, le=1, default=0.0)
    texture_score: float = Field(ge=0, le=1, default=0.0)
    blending_score: float = Field(ge=0, le=1, default=0.0)
    
    suspicious_regions: List[str] = []
    verdict: str = "UNKNOWN"
    explanation: str = ""


class AudioDeepfakeDetails(BaseModel):
    """Detailed audio deepfake analysis results."""
    
    is_synthetic: bool = False
    probability: float = Field(ge=0, le=1, default=0.0)
    confidence: float = Field(ge=0, le=1, default=0.0)
    
    spectral_score: float = Field(ge=0, le=1, default=0.0)
    prosody_score: float = Field(ge=0, le=1, default=0.0)
    quality_score: float = Field(ge=0, le=1, default=0.0)
    temporal_score: float = Field(ge=0, le=1, default=0.0)
    
    naturalness_score: float = Field(ge=0, le=1, default=0.0)
    anomalies: List[str] = []
    verdict: str = "UNKNOWN"
    explanation: str = ""


class RPPGDetails(BaseModel):
    """rPPG signal analysis results."""
    
    bpm: float = 0.0
    bpm_confidence: float = Field(ge=0, le=1, default=0.0)
    snr: float = 0.0
    periodicity: float = Field(ge=0, le=1, default=0.0)
    quality_score: float = Field(ge=0, le=1, default=0.0)


# ============================================================
# Request Schemas
# ============================================================

class AnalyzeFileRequest(BaseModel):
    """Request to analyze an uploaded file."""
    
    mode: AnalysisMode = AnalysisMode.STANDARD
    include_video: bool = True
    include_audio: bool = True
    return_details: bool = True


class AnalyzeURLRequest(BaseModel):
    """Request to analyze media from URL."""
    
    url: str
    mode: AnalysisMode = AnalysisMode.STANDARD
    include_video: bool = True
    include_audio: bool = True


class WebSocketConfig(BaseModel):
    """Configuration for WebSocket streaming analysis."""
    
    mode: AnalysisMode = AnalysisMode.STANDARD
    send_intermediate: bool = True
    target_fps: int = Field(default=30, ge=1, le=60)


# ============================================================
# Response Schemas
# ============================================================

class TrustAssessment(BaseModel):
    """Main trust assessment response."""
    
    # Final verdict
    trust_score: float = Field(ge=0, le=1, description="Overall trust score 0-1")
    trust_level: TrustLevelEnum = TrustLevelEnum.MEDIUM
    is_trustworthy: bool = False
    verdict: str = "UNKNOWN"
    
    # Component scores (summary)
    components: Dict[str, float] = Field(
        default_factory=lambda: {
            "liveness": 0.0,
            "video_authenticity": 0.0,
            "audio_authenticity": 0.0,
            "rppg_quality": 0.0,
            "av_consistency": 0.0
        }
    )
    
    # Threat info
    primary_threat: ThreatTypeEnum = ThreatTypeEnum.NONE
    threat_indicators: List[str] = []
    
    # Quality & confidence
    overall_confidence: float = Field(ge=0, le=1, default=0.0)
    signal_quality: float = Field(ge=0, le=1, default=0.0)
    
    # Human-readable
    explanation: str = ""
    recommendations: List[str] = []
    
    # Metadata
    analysis_duration_ms: float = 0.0
    media_duration_seconds: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DetailedAnalysisResponse(BaseModel):
    """Full detailed analysis response."""
    
    # Main assessment
    assessment: TrustAssessment
    
    # Detailed component results
    liveness: Optional[LivenessDetails] = None
    video_deepfake: Optional[VideoDeepfakeDetails] = None
    audio_deepfake: Optional[AudioDeepfakeDetails] = None
    rppg: Optional[RPPGDetails] = None
    
    # Request info
    request_id: str = ""
    mode: AnalysisMode = AnalysisMode.STANDARD


class StreamingUpdate(BaseModel):
    """Update message for WebSocket streaming."""
    
    type: str  # "progress", "intermediate", "final", "error"
    progress: float = Field(ge=0, le=1, default=0.0)
    message: str = ""
    
    # Intermediate results (optional)
    partial_assessment: Optional[TrustAssessment] = None
    
    # For final
    final_result: Optional[DetailedAnalysisResponse] = None


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    version: str = "1.0.0"
    uptime_seconds: float = 0.0
    components: Dict[str, bool] = Field(
        default_factory=lambda: {
            "video_pipeline": True,
            "audio_pipeline": True,
            "deepfake_video": True,
            "deepfake_audio": True,
            "liveness": True,
            "rppg": True
        }
    )


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


# ============================================================
# Utility Functions
# ============================================================

def trust_result_to_response(
    trust_result,  # MultimodalTrustResult
    request_id: str = "",
    mode: AnalysisMode = AnalysisMode.STANDARD,
    include_details: bool = True
) -> DetailedAnalysisResponse:
    """Convert engine result to API response."""
    
    # Main assessment
    assessment = TrustAssessment(
        trust_score=trust_result.trust_score,
        trust_level=TrustLevelEnum(trust_result.trust_level.value),
        is_trustworthy=trust_result.is_trustworthy,
        verdict=trust_result.verdict,
        components={
            "liveness": trust_result.liveness_score,
            "video_authenticity": trust_result.video_authenticity,
            "audio_authenticity": trust_result.audio_authenticity,
            "rppg_quality": trust_result.rppg_quality,
            "av_consistency": trust_result.av_consistency
        },
        primary_threat=ThreatTypeEnum(trust_result.primary_threat.value),
        threat_indicators=trust_result.threat_indicators,
        overall_confidence=trust_result.overall_confidence,
        explanation=trust_result.explanation,
        recommendations=trust_result.recommendations,
        analysis_duration_ms=trust_result.analysis_duration_ms,
        media_duration_seconds=trust_result.video_duration_seconds,
        has_video=trust_result.has_video,
        has_audio=trust_result.has_audio,
        timestamp=trust_result.timestamp
    )
    
    response = DetailedAnalysisResponse(
        assessment=assessment,
        request_id=request_id,
        mode=mode
    )
    
    # Add detailed results if requested
    if include_details:
        if trust_result.liveness_result:
            lr = trust_result.liveness_result
            response.liveness = LivenessDetails(
                is_live=lr.is_live,
                score=lr.liveness_score,
                confidence=lr.confidence,
                has_pulse=lr.physio.has_pulse,
                pulse_quality=lr.physio.pulse_quality,
                blink_detected=lr.physio.blink_detected,
                blink_count=lr.physio.blink_count,
                micro_movements=lr.physio.micro_movements,
                verdict=lr.verdict,
                explanation=lr.explanation
            )
        
        if trust_result.video_deepfake_result:
            vr = trust_result.video_deepfake_result
            response.video_deepfake = VideoDeepfakeDetails(
                is_deepfake=vr.is_deepfake,
                probability=vr.deepfake_probability,
                confidence=vr.confidence,
                temporal_score=vr.temporal_score,
                frequency_score=vr.frequency_score,
                texture_score=vr.texture_score,
                blending_score=vr.blending_score,
                suspicious_regions=vr.suspicious_regions,
                verdict=vr.verdict,
                explanation=vr.explanation
            )
        
        if trust_result.audio_deepfake_result:
            ar = trust_result.audio_deepfake_result
            response.audio_deepfake = AudioDeepfakeDetails(
                is_synthetic=ar.is_synthetic,
                probability=ar.synthetic_probability,
                confidence=ar.confidence,
                spectral_score=ar.spectral_score,
                prosody_score=ar.prosody_score,
                quality_score=ar.quality_score,
                temporal_score=ar.temporal_score,
                naturalness_score=ar.forensics.naturalness_score,
                anomalies=ar.anomalies,
                verdict=ar.verdict,
                explanation=ar.explanation
            )
        
        if trust_result.rppg_result:
            rr = trust_result.rppg_result
            response.rppg = RPPGDetails(
                bpm=rr.get_best_bpm(),
                bpm_confidence=rr.global_features.bpm_confidence,
                snr=rr.global_features.snr,
                periodicity=rr.global_features.periodicity,
                quality_score=rr.quality_score
            )
    
    return response
