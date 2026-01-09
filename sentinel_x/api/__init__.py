"""
Sentinel-X API Package

Component 5: API Server - FastAPI endpoints for Sentinel-X
"""

from .server import router, app

__all__ = ["router", "app"]
