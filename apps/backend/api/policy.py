"""Policy API - action allow/block."""
from fastapi import APIRouter

router = APIRouter()

@router.post("/check")
def check_action(action: str, trust_score: float):
    return {"allowed": trust_score >= 0.75}
