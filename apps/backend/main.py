"""FastAPI entry point."""
import sys
import os

# Add project root to path to allow imports from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading
import time

from apps.backend.api import scoring, ws, sentinel, upload
from sentinel_x.collectors.log_collector import LogCollector
from sentinel_x.analysis.anomaly_detector import AnomalyDetector

# Background Task Wrapper
def run_sentinel_background_tasks():
    print("🛡️ Starting Sentinel-X Background Services...")
    
    # 1. Start Log Collector
    collector = LogCollector()
    collector_thread = threading.Thread(target=collector.run, daemon=True)
    collector_thread.start()
    
    # 2. Start Anomaly Detector (for a demo agent)
    # In production, this might iterate over all active agents
    detector = AnomalyDetector("agent-007")
    detector_thread = threading.Thread(target=detector.run, daemon=True)
    detector_thread.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    run_sentinel_background_tasks()
    yield
    # Shutdown (Threads are daemon, so they will die with the process)

app = FastAPI(title="VeriPulse API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scoring.router, prefix="/api/v1", tags=["scoring"])
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(ws.router, tags=["websocket"])
app.include_router(sentinel.router, prefix="/api/sentinel", tags=["sentinel"])

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    from apps.backend.config import settings
    uvicorn.run(app, host="0.0.0.0", port=settings.ws_port)
