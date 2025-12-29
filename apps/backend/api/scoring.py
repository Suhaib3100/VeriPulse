"""Trust scoring API endpoint."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/score")
def get_score():
    return {"trust_score": 0.0, "state": "unknown"}
