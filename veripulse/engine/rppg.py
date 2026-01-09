"""
rPPG (Remote Photoplethysmography) - Pulse extraction and feature analysis.

This module provides:
- POS (Plane-Orthogonal-to-Skin) algorithm for rPPG extraction
- CHROM (Chrominance-based) algorithm as alternative
- Feature extraction: BPM, SNR, IBI statistics, cross-ROI correlation
- Quality metrics for signal reliability

Reference:
- POS: Wang et al., "Algorithmic Principles of Remote-PPG"
- CHROM: De Haan et al., "Robust Pulse Rate from Chrominance-Based rPPG"
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch, find_peaks, butter, filtfilt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class RPPGMethod(Enum):
    """Available rPPG extraction methods."""
    POS = "pos"          # Plane-Orthogonal-to-Skin
    CHROM = "chrom"      # Chrominance-based
    GREEN = "green"      # Simple green channel (baseline)
    ICA = "ica"          # Independent Component Analysis (placeholder)


@dataclass
class RPPGFeatures:
    """Features extracted from rPPG signal."""
    
    # Primary metrics
    bpm: float = 0.0                    # Dominant heart rate in BPM
    bpm_confidence: float = 0.0         # Confidence in BPM estimate
    snr: float = 0.0                    # Signal-to-noise ratio in dB
    
    # Signal quality
    signal_strength: float = 0.0        # Overall signal amplitude
    periodicity: float = 0.0            # Autocorrelation peak strength
    spectral_purity: float = 0.0        # How clean the frequency peak is
    
    # Inter-beat interval (IBI) statistics
    ibi_mean: float = 0.0               # Mean IBI in ms
    ibi_std: float = 0.0                # Std of IBI
    ibi_cv: float = 0.0                 # Coefficient of variation (std/mean)
    ibi_rmssd: float = 0.0              # Root mean square of successive differences
    
    # Heart Rate Variability (HRV) metrics
    hrv_sdnn: float = 0.0               # Standard deviation of NN intervals
    hrv_pnn50: float = 0.0              # % of successive IBIs differing by >50ms
    
    # Cross-ROI analysis (for multi-ROI setups)
    cross_roi_correlation: float = 0.0  # Correlation between ROI signals
    bpm_agreement: float = 0.0          # Agreement of BPM across ROIs
    
    # Raw signal (for debugging/visualization)
    signal: Optional[np.ndarray] = None
    psd_freqs: Optional[np.ndarray] = None
    psd_power: Optional[np.ndarray] = None


@dataclass 
class RPPGResult:
    """Complete rPPG analysis result for a video segment."""
    
    # Per-ROI features
    forehead: RPPGFeatures = field(default_factory=RPPGFeatures)
    left_cheek: RPPGFeatures = field(default_factory=RPPGFeatures)
    right_cheek: RPPGFeatures = field(default_factory=RPPGFeatures)
    
    # Fused/global features
    global_features: RPPGFeatures = field(default_factory=RPPGFeatures)
    
    # Metadata
    duration_seconds: float = 0.0
    fps: float = 30.0
    method: str = "pos"
    quality_score: float = 0.0  # Overall signal quality 0-1
    
    @property
    def features(self) -> RPPGFeatures:
        """Alias for global_features for API compatibility."""
        return self.global_features
    
    def get_best_bpm(self) -> float:
        """Get the most reliable BPM estimate."""
        # Weight by confidence
        rois = [self.forehead, self.left_cheek, self.right_cheek]
        total_conf = sum(r.bpm_confidence for r in rois)
        
        if total_conf == 0:
            return self.global_features.bpm
        
        weighted_bpm = sum(r.bpm * r.bpm_confidence for r in rois) / total_conf
        return weighted_bpm


class RPPGExtractor:
    """
    Extract rPPG signals and compute physiological features.
    
    Example:
        >>> extractor = RPPGExtractor(fps=30, method=RPPGMethod.POS)
        >>> signals = batch.get_roi_signals()  # From VideoFrameBatch
        >>> result = extractor.process(signals)
        >>> print(f"Heart rate: {result.get_best_bpm():.1f} BPM")
    """
    
    # Physiological constants
    MIN_HR = 40     # Minimum heart rate (BPM)
    MAX_HR = 200    # Maximum heart rate (BPM)
    
    def __init__(
        self,
        fps: float = 30.0,
        method: RPPGMethod = RPPGMethod.POS,
        window_seconds: float = 10.0
    ):
        self.fps = fps
        self.method = method
        self.window_seconds = window_seconds
        
        # Bandpass filter for heart rate band (0.7-4 Hz = 42-240 BPM)
        self.lowcut = 0.7
        self.highcut = 4.0
        
    def process(self, roi_signals: Dict[str, np.ndarray]) -> RPPGResult:
        """
        Process ROI signals and extract rPPG features.
        
        Args:
            roi_signals: Dict with keys 'forehead', 'left_cheek', 'right_cheek'
                        Each value is (N, 3) array of RGB values
                        
        Returns:
            RPPGResult with features for each ROI and global fusion
        """
        result = RPPGResult(
            fps=self.fps,
            method=self.method.value
        )
        
        extracted_signals = {}
        
        # Process each ROI
        for roi_name, rgb_signal in roi_signals.items():
            if len(rgb_signal) < self.fps * 2:  # Need at least 2 seconds
                continue
                
            # Extract rPPG signal using selected method
            rppg_signal = self._extract_signal(rgb_signal)
            
            # Bandpass filter
            filtered_signal = self._bandpass_filter(rppg_signal)
            
            # Extract features
            features = self._extract_features(filtered_signal)
            features.signal = filtered_signal
            
            extracted_signals[roi_name] = filtered_signal
            
            # Assign to result
            if roi_name == "forehead":
                result.forehead = features
            elif roi_name == "left_cheek":
                result.left_cheek = features
            elif roi_name == "right_cheek":
                result.right_cheek = features
        
        # Compute cross-ROI metrics
        if len(extracted_signals) >= 2:
            self._compute_cross_roi_metrics(result, extracted_signals)
        
        # Compute global features (fusion of all ROIs)
        result.global_features = self._fuse_roi_features(result)
        
        # Overall quality score
        result.quality_score = self._compute_quality_score(result)
        result.duration_seconds = len(list(roi_signals.values())[0]) / self.fps if roi_signals else 0
        
        return result
    
    def _extract_signal(self, rgb_signal: np.ndarray) -> np.ndarray:
        """Extract 1D rPPG signal from RGB time series using selected method."""
        if self.method == RPPGMethod.POS:
            return self._pos_method(rgb_signal)
        elif self.method == RPPGMethod.CHROM:
            return self._chrom_method(rgb_signal)
        elif self.method == RPPGMethod.GREEN:
            return self._green_method(rgb_signal)
        else:
            return self._pos_method(rgb_signal)
    
    def _pos_method(self, rgb_signal: np.ndarray) -> np.ndarray:
        """
        Plane-Orthogonal-to-Skin (POS) algorithm.
        
        Projects RGB signal onto plane orthogonal to skin tone.
        """
        if len(rgb_signal) == 0:
            return np.array([])
        
        # Normalize by temporal mean
        rgb_mean = np.mean(rgb_signal, axis=0, keepdims=True)
        rgb_mean = np.where(rgb_mean == 0, 1, rgb_mean)  # Avoid division by zero
        normalized = rgb_signal / rgb_mean
        
        # POS projection matrix
        # S1 = (G - B) / sqrt(2), S2 = (2R - G - B) / sqrt(6)
        # Then combine: P = S1 + alpha * S2 where alpha = std(S1) / std(S2)
        
        R, G, B = normalized[:, 0], normalized[:, 1], normalized[:, 2]
        
        S1 = (G - B) / np.sqrt(2)
        S2 = (2 * R - G - B) / np.sqrt(6)
        
        # Compute alpha in sliding window
        window_size = int(self.fps * 1.6)  # ~1.6 second window
        
        if len(S1) < window_size:
            # Too short, use global alpha
            std_s1 = np.std(S1)
            std_s2 = np.std(S2)
            alpha = std_s1 / std_s2 if std_s2 != 0 else 0
            pulse = S1 + alpha * S2
        else:
            # Sliding window alpha
            pulse = np.zeros_like(S1)
            half_window = window_size // 2
            
            for i in range(len(S1)):
                start = max(0, i - half_window)
                end = min(len(S1), i + half_window)
                
                std_s1 = np.std(S1[start:end])
                std_s2 = np.std(S2[start:end])
                alpha = std_s1 / std_s2 if std_s2 != 0 else 0
                
                pulse[i] = S1[i] + alpha * S2[i]
        
        return pulse
    
    def _chrom_method(self, rgb_signal: np.ndarray) -> np.ndarray:
        """
        Chrominance-based (CHROM) method.
        
        Uses chrominance features that are robust to illumination changes.
        """
        if len(rgb_signal) == 0:
            return np.array([])
        
        # Normalize
        rgb_mean = np.mean(rgb_signal, axis=0, keepdims=True)
        rgb_mean = np.where(rgb_mean == 0, 1, rgb_mean)
        normalized = rgb_signal / rgb_mean
        
        R, G, B = normalized[:, 0], normalized[:, 1], normalized[:, 2]
        
        # CHROM: X = 3R - 2G, Y = 1.5R + G - 1.5B
        X = 3 * R - 2 * G
        Y = 1.5 * R + G - 1.5 * B
        
        # Combine with adaptive alpha
        std_x = np.std(X)
        std_y = np.std(Y)
        alpha = std_x / std_y if std_y != 0 else 0
        
        pulse = X - alpha * Y
        
        return pulse
    
    def _green_method(self, rgb_signal: np.ndarray) -> np.ndarray:
        """Simple green channel extraction (baseline method)."""
        if len(rgb_signal) == 0:
            return np.array([])
        
        green = rgb_signal[:, 1]
        
        # Detrend
        green_detrended = signal.detrend(green)
        
        return green_detrended
    
    def _bandpass_filter(self, sig: np.ndarray, order: int = 4) -> np.ndarray:
        """Apply bandpass filter to isolate heart rate frequencies."""
        if len(sig) < 12:  # Need minimum samples for filter
            return sig
        
        # Nyquist frequency
        nyq = self.fps / 2
        
        # Ensure cutoff frequencies are valid
        low = min(self.lowcut / nyq, 0.99)
        high = min(self.highcut / nyq, 0.99)
        
        if low >= high:
            return sig
        
        try:
            b, a = butter(order, [low, high], btype='band')
            filtered = filtfilt(b, a, sig)
            return filtered
        except Exception:
            return sig
    
    def _extract_features(self, sig: np.ndarray) -> RPPGFeatures:
        """Extract all features from filtered rPPG signal."""
        features = RPPGFeatures()
        
        if len(sig) < self.fps:
            return features
        
        # Power Spectral Density
        freqs, psd = welch(sig, fs=self.fps, nperseg=min(len(sig), 256))
        features.psd_freqs = freqs
        features.psd_power = psd
        
        # Find peak in valid HR range
        min_freq = self.MIN_HR / 60  # Convert BPM to Hz
        max_freq = self.MAX_HR / 60
        
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]
        
        if len(valid_psd) == 0:
            return features
        
        # Peak detection
        peak_idx = np.argmax(valid_psd)
        peak_freq = valid_freqs[peak_idx]
        peak_power = valid_psd[peak_idx]
        
        features.bpm = peak_freq * 60
        features.signal_strength = float(np.std(sig))
        
        # SNR: Peak power vs noise floor
        noise_mask = ~((freqs >= peak_freq - 0.2) & (freqs <= peak_freq + 0.2))
        noise_power = np.mean(psd[noise_mask & valid_mask]) if np.any(noise_mask & valid_mask) else 1e-10
        features.snr = float(10 * np.log10(peak_power / noise_power)) if noise_power > 0 else 0
        
        # Spectral purity (ratio of peak to total power in HR band)
        total_power = np.sum(valid_psd)
        features.spectral_purity = float(peak_power / total_power) if total_power > 0 else 0
        
        # BPM confidence based on SNR and spectral purity
        features.bpm_confidence = min(1.0, (features.snr / 10) * features.spectral_purity)
        
        # Periodicity via autocorrelation
        features.periodicity = self._compute_periodicity(sig)
        
        # IBI analysis
        self._compute_ibi_features(sig, features)
        
        return features
    
    def _compute_periodicity(self, sig: np.ndarray) -> float:
        """Compute periodicity using autocorrelation."""
        if len(sig) < 2:
            return 0.0
        
        # Normalize
        sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-10)
        
        # Autocorrelation
        corr = np.correlate(sig_norm, sig_norm, mode='full')
        corr = corr[len(corr) // 2:]
        corr = corr / corr[0] if corr[0] != 0 else corr
        
        # Find first significant peak (excluding lag 0)
        min_lag = int(self.fps * 60 / self.MAX_HR)  # Minimum lag for max HR
        max_lag = int(self.fps * 60 / self.MIN_HR)  # Maximum lag for min HR
        
        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        
        if min_lag >= max_lag:
            return 0.0
        
        search_region = corr[min_lag:max_lag]
        
        if len(search_region) == 0:
            return 0.0
        
        # Peak strength
        peak_val = np.max(search_region)
        
        return float(max(0, peak_val))
    
    def _compute_ibi_features(self, sig: np.ndarray, features: RPPGFeatures) -> None:
        """Compute Inter-Beat Interval (IBI) features."""
        # Find peaks in signal
        min_distance = int(self.fps * 60 / self.MAX_HR)
        
        try:
            peaks, _ = find_peaks(sig, distance=min_distance, prominence=np.std(sig) * 0.3)
        except Exception:
            return
        
        if len(peaks) < 3:
            return
        
        # Compute IBIs (in milliseconds)
        ibis = np.diff(peaks) / self.fps * 1000
        
        # Filter outliers
        median_ibi = np.median(ibis)
        valid_ibis = ibis[(ibis > median_ibi * 0.5) & (ibis < median_ibi * 1.5)]
        
        if len(valid_ibis) < 2:
            return
        
        features.ibi_mean = float(np.mean(valid_ibis))
        features.ibi_std = float(np.std(valid_ibis))
        features.ibi_cv = features.ibi_std / features.ibi_mean if features.ibi_mean > 0 else 0
        
        # RMSSD
        successive_diffs = np.diff(valid_ibis)
        features.ibi_rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
        
        # HRV metrics
        features.hrv_sdnn = features.ibi_std
        
        # pNN50
        nn50 = np.sum(np.abs(successive_diffs) > 50)
        features.hrv_pnn50 = float(nn50 / len(successive_diffs) * 100) if len(successive_diffs) > 0 else 0
    
    def _compute_cross_roi_metrics(
        self,
        result: RPPGResult,
        signals: Dict[str, np.ndarray]
    ) -> None:
        """Compute cross-ROI correlation and BPM agreement."""
        signal_list = list(signals.values())
        
        # Ensure all signals have same length
        min_len = min(len(s) for s in signal_list)
        signal_list = [s[:min_len] for s in signal_list]
        
        # Correlation between all pairs
        correlations = []
        for i in range(len(signal_list)):
            for j in range(i + 1, len(signal_list)):
                corr = np.corrcoef(signal_list[i], signal_list[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if correlations:
            result.global_features.cross_roi_correlation = float(np.mean(correlations))
        
        # BPM agreement
        bpms = [result.forehead.bpm, result.left_cheek.bpm, result.right_cheek.bpm]
        bpms = [b for b in bpms if b > 0]
        
        if len(bpms) >= 2:
            bpm_std = np.std(bpms)
            bpm_mean = np.mean(bpms)
            # Agreement: 1 - CV (lower CV = higher agreement)
            result.global_features.bpm_agreement = float(1 - bpm_std / bpm_mean) if bpm_mean > 0 else 0
    
    def _fuse_roi_features(self, result: RPPGResult) -> RPPGFeatures:
        """Fuse features from all ROIs into global features."""
        rois = [result.forehead, result.left_cheek, result.right_cheek]
        
        # Weight by confidence
        total_conf = sum(r.bpm_confidence for r in rois)
        
        if total_conf == 0:
            return RPPGFeatures()
        
        fused = RPPGFeatures()
        
        # Weighted average of key metrics
        for attr in ['bpm', 'snr', 'signal_strength', 'periodicity', 
                     'ibi_mean', 'ibi_std', 'ibi_cv', 'ibi_rmssd']:
            weighted_sum = sum(
                getattr(r, attr) * r.bpm_confidence 
                for r in rois
            )
            setattr(fused, attr, weighted_sum / total_conf)
        
        # Max confidence
        fused.bpm_confidence = max(r.bpm_confidence for r in rois)
        fused.spectral_purity = max(r.spectral_purity for r in rois)
        
        # Cross-ROI metrics (already computed)
        fused.cross_roi_correlation = result.global_features.cross_roi_correlation
        fused.bpm_agreement = result.global_features.bpm_agreement
        
        return fused
    
    def _compute_quality_score(self, result: RPPGResult) -> float:
        """Compute overall signal quality score."""
        gf = result.global_features
        
        # Components of quality
        scores = []
        
        # SNR contribution (0-10 dB maps to 0-1)
        snr_score = min(1.0, max(0, gf.snr / 10))
        scores.append(snr_score * 0.3)
        
        # Periodicity contribution
        scores.append(gf.periodicity * 0.2)
        
        # Spectral purity
        scores.append(gf.spectral_purity * 0.2)
        
        # Cross-ROI correlation
        corr_score = (gf.cross_roi_correlation + 1) / 2  # Map [-1, 1] to [0, 1]
        scores.append(corr_score * 0.15)
        
        # BPM agreement
        scores.append(gf.bpm_agreement * 0.15)
        
        return sum(scores)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Testing rPPG extraction with synthetic signal...")
    
    # Generate synthetic RGB signal with embedded pulse
    fps = 30
    duration = 10  # seconds
    n_samples = fps * duration
    t = np.arange(n_samples) / fps
    
    # Simulate heart rate at 72 BPM (1.2 Hz)
    hr_hz = 1.2
    pulse_component = 0.02 * np.sin(2 * np.pi * hr_hz * t)
    
    # Add noise and baseline
    rgb_signal = np.zeros((n_samples, 3))
    for i, color_base in enumerate([0.6, 0.4, 0.3]):  # R, G, B baseline
        rgb_signal[:, i] = color_base + pulse_component + 0.005 * np.random.randn(n_samples)
    
    # Process
    extractor = RPPGExtractor(fps=fps, method=RPPGMethod.POS)
    result = extractor.process({
        "forehead": rgb_signal,
        "left_cheek": rgb_signal * 0.95,
        "right_cheek": rgb_signal * 1.05
    })
    
    print(f"\nResults:")
    print(f"  Detected BPM: {result.get_best_bpm():.1f} (expected: 72)")
    print(f"  SNR: {result.global_features.snr:.1f} dB")
    print(f"  Periodicity: {result.global_features.periodicity:.3f}")
    print(f"  Cross-ROI Correlation: {result.global_features.cross_roi_correlation:.3f}")
    print(f"  Quality Score: {result.quality_score:.3f}")
