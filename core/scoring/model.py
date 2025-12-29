"""Trust scoring model."""

from .trust_state import TrustState
from .thresholds import RPPG_MIN, BLINK_MIN, MOTION_MAX

class TrustModel:
    def evaluate(self, rppg_score, blink_score, motion_score):
        # High trust if all conditions are satisfied
        if (
            rppg_score >= RPPG_MIN
            and blink_score >= BLINK_MIN
            and motion_score <= MOTION_MAX
        ):
            return TrustState.VERIFIED

        # Medium trust if partial conditions are met
        if rppg_score >= 0.4 or blink_score >= 0.4:
            return TrustState.SUSPICIOUS

        # Otherwise low trust
        return TrustState.SYNTHETIC