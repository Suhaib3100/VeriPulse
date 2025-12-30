"""Signal filtering - bandpass, detrending."""

import numpy as np
from scipy.signal import butter, filtfilt

class BandpassFilter:
    def __init__(self, low=0.7, high=3.0, fs=30):
        self.low = low
        self.high = high
        self.fs = fs
        
    def apply(self, signal):
        """Apply bandpass filter to rPPG signal."""
        if len(signal) == 0:
            return signal
            
        nyquist = 0.5 * self.fs
        low = self.low / nyquist
        high = self.high / nyquist
        
        # Ensure valid bounds
        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))
        
        if low >= high:
            return signal
            
        b, a = butter(3, [low, high], btype='band')
        return filtfilt(b, a, signal)
