"""FastAPI entry point."""
import sys
import os

# Add project root to path to allow imports from core and veripulse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading
import time

# Import the new VeriPulse API (always available)
from apps.backend.api import veripulse_api

# Try to import legacy modules (optional, may fail if core module has issues)
try:
    from apps.backend.api import scoring, upload
    LEGACY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Legacy scoring/upload not available: {e}")
    LEGACY_AVAILABLE = False

# Sentinel is always optional
try:
    from apps.backend.api import sentinel
    SENTINEL_AVAILABLE_API = True
except ImportError:
    SENTINEL_AVAILABLE_API = False

# Try to import Sentinel-X (optional)
try:
    from sentinel_x.collectors.log_collector import LogCollector
    from sentinel_x.analysis.anomaly_detector import AnomalyDetector
    SENTINEL_AVAILABLE = True
except ImportError:
    SENTINEL_AVAILABLE = False
    print("⚠️ Sentinel-X not available, running without agent monitoring")

# Background Task Wrapper
def run_sentinel_background_tasks():
    if not SENTINEL_AVAILABLE:
        return
    
    try:
        print("🛡️ Starting Sentinel-X Background Services...")
        
        # 1. Start Log Collector
        collector = LogCollector()
        collector_thread = threading.Thread(target=collector.run, daemon=True)
        collector_thread.start()
        
        # 2. Start Anomaly Detector (for a demo agent)
        # In production, this might iterate over all active agents
        try:
            detector = AnomalyDetector("agent-007")
            detector_thread = threading.Thread(target=detector.run, daemon=True)
            detector_thread.start()
        except Exception as e:
            print(f"⚠️ AnomalyDetector failed to start: {e}")
    except Exception as e:
        print(f"⚠️ Sentinel-X background tasks failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 VeriPulse Engine Starting...")
    run_sentinel_background_tasks()
    yield
    # Shutdown (Threads are daemon, so they will die with the process)
    print("👋 VeriPulse Engine Shutting Down...")

app = FastAPI(title="VeriPulse API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Legacy routes (for backward compatibility - optional)
if LEGACY_AVAILABLE:
    try:
        app.include_router(scoring.router, prefix="/api/v1", tags=["scoring"])
        app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
    except:
        pass

# New VeriPulse Engine routes (main API)
app.include_router(veripulse_api.router, prefix="/api/v1", tags=["veripulse"])
app.include_router(veripulse_api.router, tags=["websocket"])

# Sentinel-X API (new comprehensive API)
try:
    from sentinel_x.api.server import router as sentinel_x_router
    app.include_router(sentinel_x_router, prefix="/api", tags=["sentinel-x"])
    print("✅ Sentinel-X API routes loaded")
except ImportError as e:
    print(f"⚠️ Sentinel-X API not available: {e}")

# Legacy Sentinel (optional - kept for backward compatibility)
if SENTINEL_AVAILABLE_API:
    try:
        app.include_router(sentinel.router, prefix="/api/sentinel", tags=["sentinel"])
    except:
        pass

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    from apps.backend.config import settings
    uvicorn.run(app, host="0.0.0.0", port=settings.ws_port)
