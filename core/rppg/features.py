"""Feature extraction from rPPG signal."""

import numpy as np
from scipy.signal import welch, find_peaks

class FeatureExtractor:
    def __init__(self, fs=30):
        self.fs = fs
    
    def extract(self, signal):
        """
        Extract SNR, heart rate, periodicity features.
        
        Args:
            signal: 1D numpy array of rPPG signal
            
        Returns:
            dict: Dictionary containing 'snr', 'hr_bpm', 'periodicity'
        """
        if len(signal) < self.fs: # Need at least 1 second
            return {"snr": 0.0, "hr_bpm": 0.0, "periodicity": 0.0}

        # Power Spectral Density
        # nperseg should be enough to cover the signal or a window
        nperseg = min(len(signal), 256)
        freqs, psd = welch(signal, self.fs, nperseg=nperseg)
        
        # Find peak frequency in valid range [0.7, 3.0] Hz (42-180 BPM)
        valid_mask = (freqs >= 0.7) & (freqs <= 3.0)
        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]
        
        if len(valid_psd) == 0:
            return {"snr": 0.0, "hr_bpm": 0.0, "periodicity": 0.0}
            
        peak_idx = np.argmax(valid_psd)
        peak_freq = valid_freqs[peak_idx]
        peak_power = valid_psd[peak_idx]
        
        # SNR: Peak power / (Total power - Peak power)
        # We calculate noise power as the average power of non-peak frequencies in the valid range
        # or just ratio of peak to everything else.
        
        # Let's define SNR as Peak Power / Median Power of the spectrum (in valid range)
        # This is robust to other peaks.
        median_power = np.median(valid_psd)
        if median_power > 0:
            snr = peak_power / median_power
        else:
            snr = 0.0
            
        # Periodicity: We can use the max autocorrelation value
        # Normalize signal
        norm_signal = (signal - np.mean(signal))
        if np.std(norm_signal) > 0:
            norm_signal = norm_signal / np.std(norm_signal)
            
        # Autocorrelation
        corr = np.correlate(norm_signal, norm_signal, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find peaks in autocorrelation
        # We expect a peak at lag corresponding to heart rate
        # Lag range for 0.7-3.0 Hz: fs/3.0 to fs/0.7
        min_lag = int(self.fs / 3.0)
        max_lag = int(self.fs / 0.7)
        
        periodicity = 0.0
        if len(corr) > max_lag:
            roi_corr = corr[min_lag:max_lag]
            if len(roi_corr) > 0:
                periodicity = np.max(roi_corr) / len(signal) # Normalize by length
        
        # IBI / Temporal Stability
        # Find peaks in the time domain signal
        # Use distance corresponding to max HR (3.0 Hz -> 0.33s -> fs*0.33 samples)
        peaks, _ = find_peaks(signal, distance=int(self.fs/3.0))
        
        ibi_mean = 0.0
        ibi_std = 0.0
        ibi_cv = 0.0
        
        if len(peaks) > 1:
            ibis = np.diff(peaks) / self.fs # in seconds
            ibi_mean = np.mean(ibis)
            if len(ibis) > 1:
                ibi_std = np.std(ibis)
                if ibi_mean > 0:
                    ibi_cv = ibi_std / ibi_mean
        
        return {
            "snr": float(snr),
            "hr_bpm": float(peak_freq * 60),
            "periodicity": float(periodicity),
            "ibi_mean": float(ibi_mean),
            "ibi_std": float(ibi_std),
            "ibi_cv": float(ibi_cv)
        }
