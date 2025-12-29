"""Trust score thresholds."""

# Thresholds for physiological and behavioral signals
RPPG_MIN = 0.6       # minimum acceptable rPPG score
BLINK_MIN = 0.5      # minimum acceptable blink score
MOTION_MAX = 0.4     # maximum allowed motion score

# General scoring thresholds (optional, for overall classification)
VERIFIED_THRESHOLD = 0.75
SUSPICIOUS_THRESHOLD = 0.4

class Thresholds:
    verified = VERIFIED_THRESHOLD
    suspicious = SUSPICIOUS_THRESHOLD