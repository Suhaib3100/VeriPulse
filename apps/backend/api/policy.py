"""Policy API - action allow/block."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class PolicyRequest(BaseModel):
    action: str
    trust_score: float

@router.post("/check")
def check_action(req: PolicyRequest):
    """
    Check if an action is allowed based on the trust score.
    Thresholds:
    - >= 0.7: Verified (Allow high-risk actions)
    - 0.4 - 0.7: Suspicious (Allow low-risk, prompt for high-risk)
    - < 0.4: Denied (Block all)
    """
    score = req.trust_score
    action = req.action
    
    if score >= 0.7:
        return {"allowed": True, "level": "HIGH", "message": "Verified session."}
    elif score >= 0.4:
        if action in ["transfer_money", "view_sensitive_data"]:
            return {"allowed": False, "level": "MEDIUM", "message": "Additional verification required."}
        return {"allowed": True, "level": "MEDIUM", "message": "Proceed with caution."}
    else:
        return {"allowed": False, "level": "LOW", "message": "Session untrusted. Action blocked."}
