"""FastAPI entry point."""
import sys
import os

# Add project root to path to allow imports from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apps.backend.api import scoring, ws

app = FastAPI(title="VeriPulse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scoring.router, prefix="/api/v1", tags=["scoring"])
app.include_router(ws.router, tags=["websocket"])

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    from apps.backend.config import settings
    uvicorn.run(app, host="0.0.0.0", port=settings.ws_port)
