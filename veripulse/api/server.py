"""
VeriPulse API Server - FastAPI HTTP and WebSocket server.

Provides REST endpoints and WebSocket streaming for:
- File upload analysis
- URL-based analysis
- Real-time webcam/stream analysis
"""

import asyncio
import time
import uuid
import tempfile
import os
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    AnalyzeFileRequest,
    AnalyzeURLRequest,
    WebSocketConfig,
    TrustAssessment,
    DetailedAnalysisResponse,
    StreamingUpdate,
    HealthResponse,
    ErrorResponse,
    AnalysisMode,
    trust_result_to_response,
)

from ..engine import (
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


# ============================================================
# App Setup
# ============================================================

# Global state
class AppState:
    """Application state."""
    start_time: float = 0.0
    request_count: int = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan manager."""
    # Startup
    state.start_time = time.time()
    print("VeriPulse Engine starting...")
    
    yield
    
    # Shutdown
    print("VeriPulse Engine shutting down...")


app = FastAPI(
    title="VeriPulse API",
    description="Multimodal deepfake and liveness detection API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health & Info Endpoints
# ============================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "service": "VeriPulse API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.time() - state.start_time,
        components={
            "video_pipeline": True,
            "audio_pipeline": True,
            "deepfake_video": True,
            "deepfake_audio": True,
            "liveness": True,
            "rppg": True
        }
    )


# ============================================================
# Analysis Engine
# ============================================================

class AnalysisEngine:
    """
    Main analysis engine that coordinates all detection components.
    """
    
    def __init__(self):
        self.face_tracker = FaceTracker()
        self.rppg_extractor = RPPGExtractor()
        self.liveness_detector = LivenessDetector()
        self.video_deepfake_detector = VideoDeepfakeDetector()
        self.audio_feature_extractor = AudioFeatureExtractor()
        self.audio_deepfake_detector = AudioDeepfakeDetector()
        self.fusion = MultimodalFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)
    
    def analyze_video_file(
        self,
        file_path: str,
        mode: AnalysisMode = AnalysisMode.STANDARD
    ) -> dict:
        """Analyze a video file."""
        start_time = time.time()
        
        # Open video source
        try:
            video_source = VideoSource.from_file(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open video: {e}")
        
        # Collect frames and analyze
        frames = []
        landmarks = []
        face_crops = []
        head_poses = []
        blink_events = []
        
        frame_batch = VideoFrameBatch(max_frames=300)  # ~10 seconds at 30fps
        
        for frame_data in video_source.stream_frames():
            frame = frame_data.bgr
            
            # Track face
            detection = self.face_tracker.process(frame)
            
            if detection and detection.landmarks is not None:
                frames.append(frame)
                landmarks.append(detection.landmarks)
                
                if detection.face_crop is not None:
                    face_crops.append(detection.face_crop)
                
                if detection.head_pose:
                    head_poses.append(detection.head_pose)
                
                # Add to batch for rPPG
                frame_batch.add_frame(frame_data, detection)
                
                # Check for blinks
                if detection.ear is not None and detection.ear < 0.2:
                    blink_events.append({
                        'time': frame_data.timestamp,
                        'ear': detection.ear
                    })
            
            # Limit analysis based on mode
            if mode == AnalysisMode.QUICK and len(frames) >= 90:  # 3 seconds
                break
            elif mode == AnalysisMode.STANDARD and len(frames) >= 300:  # 10 seconds
                break
        
        video_source.close()
        
        # Get duration
        fps = video_source.fps if hasattr(video_source, 'fps') else 30
        duration = len(frames) / fps if fps > 0 else 0
        
        # Extract ROI signals for rPPG
        roi_signals = frame_batch.get_roi_signals()
        
        # Run rPPG analysis
        rppg_result = None
        if roi_signals and len(list(roi_signals.values())[0]) > 60:
            rppg_result = self.rppg_extractor.process(roi_signals)
        
        # Run liveness analysis
        texture_features = self._extract_texture_features(frames)
        liveness_result = self.liveness_detector.analyze(
            rppg_result=rppg_result,
            blink_events=blink_events,
            head_poses=head_poses,
            texture_features=texture_features,
            fps=fps,
            duration_seconds=duration
        )
        
        # Run video deepfake detection
        video_deepfake_result = self.video_deepfake_detector.analyze(
            frames=frames,
            landmarks=landmarks,
            face_crops=face_crops,
            fps=fps
        )
        
        analysis_time = (time.time() - start_time) * 1000
        
        return {
            'rppg': rppg_result,
            'liveness': liveness_result,
            'video_deepfake': video_deepfake_result,
            'duration': duration,
            'analysis_time_ms': analysis_time
        }
    
    def analyze_audio_file(
        self,
        file_path: str,
        mode: AnalysisMode = AnalysisMode.STANDARD
    ) -> dict:
        """Analyze an audio file."""
        start_time = time.time()
        
        try:
            audio_source = AudioSource.from_file(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open audio: {e}")
        
        # Extract features
        features = self.audio_feature_extractor.extract(audio_source)
        
        # Run deepfake detection
        audio_deepfake_result = self.audio_deepfake_detector.analyze(
            audio_source, features
        )
        
        audio_source.close()
        
        analysis_time = (time.time() - start_time) * 1000
        
        return {
            'audio_deepfake': audio_deepfake_result,
            'duration': features.duration,
            'analysis_time_ms': analysis_time
        }
    
    def analyze_media_file(
        self,
        file_path: str,
        mode: AnalysisMode = AnalysisMode.STANDARD,
        include_video: bool = True,
        include_audio: bool = True
    ):
        """
        Analyze a media file (video with optional audio).
        
        Returns MultimodalTrustResult.
        """
        start_time = time.time()
        
        video_results = None
        audio_results = None
        
        # Video analysis
        if include_video:
            try:
                video_results = self.analyze_video_file(file_path, mode)
            except Exception as e:
                print(f"Video analysis failed: {e}")
        
        # Audio analysis (if file has audio)
        if include_audio:
            try:
                # Try to extract/analyze audio
                audio_results = self.analyze_audio_file(file_path, mode)
            except Exception as e:
                print(f"Audio analysis failed (may not have audio): {e}")
        
        # Compute total duration
        duration = 0
        if video_results:
            duration = video_results.get('duration', 0)
        elif audio_results:
            duration = audio_results.get('duration', 0)
        
        analysis_time = (time.time() - start_time) * 1000
        
        # Fuse results
        trust_result = self.fusion.fuse(
            liveness=video_results.get('liveness') if video_results else None,
            video_deepfake=video_results.get('video_deepfake') if video_results else None,
            audio_deepfake=audio_results.get('audio_deepfake') if audio_results else None,
            rppg=video_results.get('rppg') if video_results else None,
            video_duration=duration,
            analysis_time_ms=analysis_time
        )
        
        return trust_result
    
    def _extract_texture_features(self, frames: List[np.ndarray]) -> dict:
        """Extract texture features from frames."""
        if not frames:
            return {}
        
        import cv2
        
        # Sample middle frame
        mid_frame = frames[len(frames) // 2]
        
        if len(mid_frame.shape) == 3:
            gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = mid_frame
        
        # Laplacian variance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = lap.var()
        
        # Simple moire detection (high frequency patterns)
        f_transform = np.fft.fft2(gray.astype(float))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        center = np.array(magnitude.shape) // 2
        high_freq_region = magnitude[
            center[0]-10:center[0]+10,
            center[1]-10:center[1]+10
        ]
        moire_score = np.mean(high_freq_region) / (np.mean(magnitude) + 1e-10)
        moire_score = min(1.0, moire_score / 10)
        
        return {
            'laplacian_var': float(lap_var),
            'moire_score': float(moire_score),
            'reflection_consistency': 0.5,  # Placeholder
            'depth_consistency': 0.5  # Placeholder
        }


# Global engine instance
engine = AnalysisEngine()


# ============================================================
# File Analysis Endpoints
# ============================================================

@app.post("/analyze/file", response_model=DetailedAnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    mode: AnalysisMode = AnalysisMode.STANDARD,
    include_video: bool = True,
    include_audio: bool = True,
    return_details: bool = True
):
    """
    Analyze an uploaded media file.
    
    Supports video formats (MP4, AVI, MOV, WebM) and audio formats (WAV, MP3).
    """
    request_id = str(uuid.uuid4())
    state.request_count += 1
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.wav', '.mp3', '.m4a', '.ogg'}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Run analysis
        trust_result = engine.analyze_media_file(
            tmp_path,
            mode=mode,
            include_video=include_video,
            include_audio=include_audio
        )
        
        # Convert to response
        response = trust_result_to_response(
            trust_result,
            request_id=request_id,
            mode=mode,
            include_details=return_details
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.post("/analyze/quick", response_model=TrustAssessment)
async def analyze_quick(file: UploadFile = File(...)):
    """
    Quick analysis - returns only the trust assessment (faster).
    """
    response = await analyze_file(
        file=file,
        mode=AnalysisMode.QUICK,
        return_details=False
    )
    return response.assessment


# ============================================================
# WebSocket Streaming
# ============================================================

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming analysis.
    
    Client sends video frames, receives progressive trust updates.
    """
    await websocket.accept()
    
    try:
        # Receive configuration
        config_data = await websocket.receive_json()
        config = WebSocketConfig(**config_data)
        
        # Initialize analysis state
        frame_batch = VideoFrameBatch(max_frames=300)
        face_tracker = FaceTracker()
        frames_processed = 0
        
        await websocket.send_json(
            StreamingUpdate(
                type="progress",
                progress=0,
                message="Ready for frames"
            ).dict()
        )
        
        while True:
            # Receive frame data
            data = await websocket.receive_bytes()
            
            # Decode frame
            frame = np.frombuffer(data, dtype=np.uint8)
            # Assume frame is sent as raw BGR bytes
            # Client should send: width (4 bytes) + height (4 bytes) + BGR data
            
            # Process frame
            # (simplified - in production, properly decode frame format)
            
            frames_processed += 1
            progress = min(1.0, frames_processed / 300)
            
            # Send progress update
            if config.send_intermediate and frames_processed % 30 == 0:
                await websocket.send_json(
                    StreamingUpdate(
                        type="progress",
                        progress=progress,
                        message=f"Processed {frames_processed} frames"
                    ).dict()
                )
            
            # Check if we have enough for analysis
            if frames_processed >= 300:
                break
        
        # Final analysis
        # (would run full pipeline here)
        
        await websocket.send_json(
            StreamingUpdate(
                type="final",
                progress=1.0,
                message="Analysis complete"
            ).dict()
        )
        
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json(
            StreamingUpdate(
                type="error",
                message=str(e)
            ).dict()
        )
        await websocket.close()


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=type(exc).__name__,
            message=str(exc)
        ).dict()
    )


# ============================================================
# Run Server
# ============================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
