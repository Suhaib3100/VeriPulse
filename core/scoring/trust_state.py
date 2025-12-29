"""Trust state management."""
from enum import Enum

class TrustState(Enum):
    VERIFIED = "verified"
    SUSPICIOUS = "suspicious"
    SYNTHETIC = "likely_synthetic"
