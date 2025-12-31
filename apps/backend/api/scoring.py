"""Trust scoring API endpoint."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/score")
def get_score():
    """
    Get the current trust score.
    Note: Real-time scoring happens via WebSocket at /ws/liveness.
    This endpoint could be used to retrieve the final session score.
    """
    return {
        "trust_score": 0.0, 
        "state": "unknown",
        "info": "Connect to /ws/liveness for real-time scoring."
    }
