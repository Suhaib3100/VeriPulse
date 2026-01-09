"""
VeriPulse API - REST and WebSocket server components.

This module exports:
- FastAPI app instance
- Pydantic schemas for request/response
- Server runner utility
"""

from .schemas import (
    # Enums
    TrustLevelEnum,
    ThreatTypeEnum,
    AnalysisMode,
    
    # Component schemas
    ComponentScore,
    LivenessDetails,
    VideoDeepfakeDetails,
    AudioDeepfakeDetails,
    RPPGDetails,
    
    # Request schemas
    AnalyzeFileRequest,
    AnalyzeURLRequest,
    WebSocketConfig,
    
    # Response schemas
    TrustAssessment,
    DetailedAnalysisResponse,
    StreamingUpdate,
    HealthResponse,
    ErrorResponse,
    
    # Utilities
    trust_result_to_response,
)

from .server import app, run_server, AnalysisEngine

__all__ = [
    # App
    "app",
    "run_server",
    "AnalysisEngine",
    
    # Enums
    "TrustLevelEnum",
    "ThreatTypeEnum",
    "AnalysisMode",
    
    # Schemas
    "ComponentScore",
    "LivenessDetails",
    "VideoDeepfakeDetails",
    "AudioDeepfakeDetails",
    "RPPGDetails",
    "AnalyzeFileRequest",
    "AnalyzeURLRequest",
    "WebSocketConfig",
    "TrustAssessment",
    "DetailedAnalysisResponse",
    "StreamingUpdate",
    "HealthResponse",
    "ErrorResponse",
    "trust_result_to_response",
]
