"""
Audio Deepfake Detection - Voice synthesis and cloning detection.

This module provides:
1. Spectral analysis for synthetic voice artifacts
2. Neural classifier stub for voice deepfake detection
3. Prosody anomaly detection
4. Voice naturalness scoring
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from .audio_pipeline import AudioSource, AudioFeatures, AudioFeatureExtractor


class AudioDeepfakeSignal(Enum):
    """Types of audio deepfake detection signals."""
    SPECTRAL_ARTIFACTS = "spectral"
    PROSODY_ANOMALIES = "prosody"
    VOICE_QUALITY = "quality"
    TEMPORAL_PATTERNS = "temporal"
    NEURAL_CLASSIFIER = "neural"


@dataclass
class AudioForensicFeatures:
    """Forensic features for audio deepfake detection."""
    
    # Spectral artifacts (synthetic voice often has different patterns)
    spectral_flatness: float = 0.0       # Very flat = suspicious
    spectral_variability: float = 0.0    # Natural voice has more variation
    high_freq_energy_ratio: float = 0.0  # TTS often lacks high frequencies
    harmonic_ratio: float = 0.0          # Harmonics-to-noise
    sub_band_correlation: float = 0.0    # Correlation between frequency bands
    
    # Prosody analysis
    pitch_naturalness: float = 0.0       # Natural pitch variation
    pitch_contour_smoothness: float = 0.0 # Too smooth = synthetic
    pause_pattern: float = 0.0           # Natural pauses
    speaking_rate_variance: float = 0.0  # Natural rate variation
    
    # Voice quality
    breathiness: float = 0.0             # Natural voices have breath
    creakiness: float = 0.0              # Voice quality markers
    vocal_tremor: float = 0.0            # Natural micro-variations
    formant_stability: float = 0.0       # Formant tracking
    
    # Temporal patterns
    attack_smoothness: float = 0.0       # Word onset patterns
    frame_correlation: float = 0.0       # Frame-to-frame similarity
    silence_noise_floor: float = 0.0     # Noise in silent parts
    
    # Compression/artifact detection
    quantization_noise: float = 0.0      # Quantization artifacts
    codec_artifacts: float = 0.0         # Compression artifacts
    
    # Overall quality
    naturalness_score: float = 0.0


@dataclass
class AudioDeepfakeResult:
    """Result of audio deepfake analysis."""
    
    # Primary output
    is_synthetic: bool = False
    synthetic_probability: float = 0.0   # 0-1, higher = more likely fake
    confidence: float = 0.0
    
    # Component scores (0-1, higher = more suspicious)
    spectral_score: float = 0.0
    prosody_score: float = 0.0
    quality_score: float = 0.0
    temporal_score: float = 0.0
    neural_score: float = 0.0
    
    # Forensic features
    forensics: AudioForensicFeatures = field(default_factory=AudioForensicFeatures)
    
    # Detected anomalies
    anomalies: List[str] = field(default_factory=list)
    
    # Verdict
    verdict: str = "UNKNOWN"
    explanation: str = ""


class AudioDeepfakeDetector:
    """
    Multi-signal audio deepfake detection.
    
    Detects:
    - Text-to-Speech (TTS) synthesis
    - Voice cloning
    - Voice conversion
    - AI-generated speech
    
    Example:
        >>> detector = AudioDeepfakeDetector()
        >>> source = AudioSource.from_file("speech.wav")
        >>> result = detector.analyze(source)
        >>> print(f"Synthetic probability: {result.synthetic_probability:.1%}")
    """
    
    # Detection threshold
    SYNTHETIC_THRESHOLD = 0.5
    
    def __init__(
        self,
        use_neural: bool = False,
        model_path: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize audio deepfake detector.
        
        Args:
            use_neural: Whether to use neural network classifier
            model_path: Path to pretrained audio deepfake model
            weights: Custom weights for combining signals
        """
        self.use_neural = use_neural
        self.model_path = model_path
        self.model = None
        
        self.feature_extractor = AudioFeatureExtractor()
        
        # Default weights
        self.weights = weights or {
            'spectral': 0.25,
            'prosody': 0.25,
            'quality': 0.20,
            'temporal': 0.15,
            'neural': 0.15
        }
        
        if use_neural and model_path:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load neural network model."""
        try:
            print(f"Note: Neural model loading not implemented. Path: {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.use_neural = False
    
    def analyze(
        self,
        source: AudioSource,
        audio_features: Optional[AudioFeatures] = None
    ) -> AudioDeepfakeResult:
        """
        Analyze audio for synthetic/deepfake indicators.
        
        Args:
            source: AudioSource containing audio to analyze
            audio_features: Pre-extracted features (optional)
            
        Returns:
            AudioDeepfakeResult with detection scores
        """
        result = AudioDeepfakeResult()
        
        samples = source.get_all_samples()
        if len(samples) == 0:
            result.verdict = "NO_DATA"
            result.explanation = "No audio data provided"
            return result
        
        # Convert to mono
        if len(samples.shape) > 1:
            samples = np.mean(samples, axis=1)
        
        # Extract features if not provided
        if audio_features is None:
            audio_features = self.feature_extractor.extract(source)
        
        sr = source.sample_rate
        
        # 1. Spectral analysis
        result.spectral_score = self._analyze_spectral(samples, sr, audio_features, result.forensics)
        
        # 2. Prosody analysis
        result.prosody_score = self._analyze_prosody(samples, sr, audio_features, result.forensics)
        
        # 3. Voice quality analysis
        result.quality_score = self._analyze_quality(samples, sr, audio_features, result.forensics)
        
        # 4. Temporal pattern analysis
        result.temporal_score = self._analyze_temporal(samples, sr, result.forensics)
        
        # 5. Neural classifier (if enabled)
        if self.use_neural and self.model is not None:
            result.neural_score = self._neural_classify(samples, sr, audio_features)
        
        # Combine scores
        self._compute_final_score(result)
        
        # Compute naturalness
        result.forensics.naturalness_score = 1 - result.synthetic_probability
        
        # Determine verdict
        self._compute_verdict(result)
        
        return result
    
    def _analyze_spectral(
        self,
        samples: np.ndarray,
        sr: int,
        features: AudioFeatures,
        forensics: AudioForensicFeatures
    ) -> float:
        """Analyze spectral characteristics for synthetic artifacts."""
        scores = []
        
        # 1. Spectral flatness (synthetic voices often more uniform)
        if features.spectral_flatness > 0:
            forensics.spectral_flatness = features.spectral_flatness
            
            # Very high flatness is suspicious
            if features.spectral_flatness > 0.3:
                scores.append(0.7)
            elif features.spectral_flatness > 0.15:
                scores.append(0.4)
            else:
                scores.append(0.2)
        
        # 2. High frequency energy analysis
        # TTS/synthetic often lacks natural high frequency content
        n_fft = 2048
        hop = 512
        
        # Compute spectrogram
        n_frames = 1 + (len(samples) - n_fft) // hop
        if n_frames > 0:
            frames = np.array([samples[i*hop:i*hop+n_fft] for i in range(n_frames)])
            frames = frames * np.hanning(n_fft)
            spec = np.abs(np.fft.rfft(frames, axis=1))
            
            # Split into low (0-4kHz) and high (4kHz+) bands
            freq_bins = np.fft.rfftfreq(n_fft, 1/sr)
            low_mask = freq_bins < 4000
            high_mask = freq_bins >= 4000
            
            low_energy = np.mean(spec[:, low_mask])
            high_energy = np.mean(spec[:, high_mask])
            
            if low_energy > 0:
                hf_ratio = high_energy / low_energy
                forensics.high_freq_energy_ratio = float(hf_ratio)
                
                # Very low high-freq ratio is suspicious
                if hf_ratio < 0.05:
                    scores.append(0.7)
                elif hf_ratio < 0.1:
                    scores.append(0.5)
                else:
                    scores.append(0.2)
        
        # 3. Spectral variability over time
        if features.mfccs is not None and len(features.mfccs) > 10:
            mfcc_var = np.mean(np.var(features.mfccs, axis=0))
            forensics.spectral_variability = float(mfcc_var)
            
            # Low variability can indicate synthetic
            if mfcc_var < 5:
                scores.append(0.6)
            elif mfcc_var < 10:
                scores.append(0.4)
            else:
                scores.append(0.2)
        
        # 4. Sub-band correlation
        # Natural speech has characteristic correlations between bands
        if n_frames > 10:
            n_bands = 4
            band_size = spec.shape[1] // n_bands
            bands = [spec[:, i*band_size:(i+1)*band_size].mean(axis=1) for i in range(n_bands)]
            
            correlations = []
            for i in range(len(bands)):
                for j in range(i+1, len(bands)):
                    corr = np.corrcoef(bands[i], bands[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                mean_corr = np.mean(correlations)
                forensics.sub_band_correlation = float(mean_corr)
                
                # Too high correlation is suspicious (overly coherent)
                if mean_corr > 0.9:
                    scores.append(0.6)
                elif mean_corr < 0.3:
                    scores.append(0.5)  # Too low also suspicious
                else:
                    scores.append(0.2)
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_prosody(
        self,
        samples: np.ndarray,
        sr: int,
        features: AudioFeatures,
        forensics: AudioForensicFeatures
    ) -> float:
        """Analyze prosodic features for naturalness."""
        scores = []
        
        # 1. Pitch naturalness
        if features.pitch_std > 0:
            # Natural speech has pitch std typically 20-60 Hz
            pitch_cv = features.pitch_std / features.pitch_mean if features.pitch_mean > 0 else 0
            forensics.pitch_naturalness = float(pitch_cv)
            
            if pitch_cv < 0.05:  # Too monotone
                scores.append(0.7)
            elif pitch_cv > 0.5:  # Too variable
                scores.append(0.6)
            else:
                scores.append(0.2)
        
        # 2. Pitch contour smoothness
        # Estimate pitch contour from autocorrelation
        frame_size = int(sr * 0.03)  # 30ms frames
        hop = int(sr * 0.01)  # 10ms hop
        
        n_frames = (len(samples) - frame_size) // hop
        if n_frames > 10:
            pitch_estimates = []
            
            for i in range(n_frames):
                frame = samples[i*hop:i*hop+frame_size]
                # Simple autocorrelation pitch detection
                corr = np.correlate(frame, frame, mode='full')
                corr = corr[len(corr)//2:]
                
                # Find peak in typical pitch range (80-400 Hz)
                min_lag = int(sr / 400)
                max_lag = int(sr / 80)
                
                if max_lag < len(corr):
                    search_region = corr[min_lag:max_lag]
                    if len(search_region) > 0:
                        peak_idx = np.argmax(search_region) + min_lag
                        pitch_estimates.append(sr / peak_idx if peak_idx > 0 else 0)
            
            if len(pitch_estimates) > 5:
                # Check smoothness of pitch contour
                pitch_diff = np.diff(pitch_estimates)
                smoothness = 1 / (1 + np.std(pitch_diff))
                forensics.pitch_contour_smoothness = float(smoothness)
                
                # Too smooth is suspicious
                if smoothness > 0.8:
                    scores.append(0.6)
                elif smoothness < 0.2:  # Too jerky
                    scores.append(0.5)
                else:
                    scores.append(0.2)
        
        # 3. Speaking rate variance
        # Natural speech has variable rate
        if features.vad_ratio > 0:
            # Using energy to estimate speech segments
            frame_energies = []
            frame_size = int(sr * 0.02)
            
            for i in range(0, len(samples) - frame_size, frame_size):
                frame = samples[i:i+frame_size]
                frame_energies.append(np.mean(frame ** 2))
            
            if frame_energies:
                energy_var = np.std(frame_energies) / (np.mean(frame_energies) + 1e-10)
                forensics.speaking_rate_variance = float(energy_var)
                
                if energy_var < 0.5:  # Very uniform
                    scores.append(0.6)
                else:
                    scores.append(0.2)
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_quality(
        self,
        samples: np.ndarray,
        sr: int,
        features: AudioFeatures,
        forensics: AudioForensicFeatures
    ) -> float:
        """Analyze voice quality markers."""
        scores = []
        
        # 1. Breathiness detection (natural voices have breath noise)
        # Check for noise in unvoiced regions
        frame_size = int(sr * 0.02)
        
        unvoiced_energy = []
        voiced_energy = []
        
        for i in range(0, len(samples) - frame_size, frame_size):
            frame = samples[i:i+frame_size]
            energy = np.mean(frame ** 2)
            
            # Simple voiced/unvoiced detection using zero crossing
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            
            if zcr > 0.3:  # Likely unvoiced
                unvoiced_energy.append(energy)
            else:  # Likely voiced
                voiced_energy.append(energy)
        
        if unvoiced_energy and voiced_energy:
            breath_ratio = np.mean(unvoiced_energy) / (np.mean(voiced_energy) + 1e-10)
            forensics.breathiness = float(breath_ratio)
            
            # Very low breathiness (too clean) is suspicious
            if breath_ratio < 0.01:
                scores.append(0.6)
            elif breath_ratio > 0.5:  # Too breathy (rare but possible artifact)
                scores.append(0.5)
            else:
                scores.append(0.2)
        
        # 2. Vocal tremor (micro-variations)
        # Natural voices have slight amplitude variations
        if len(samples) > sr * 0.5:  # Need at least 0.5 seconds
            # Compute short-time energy
            hop = int(sr * 0.005)  # 5ms hop
            energies = []
            
            for i in range(0, len(samples) - hop, hop):
                energies.append(np.mean(samples[i:i+hop] ** 2))
            
            if len(energies) > 10:
                # Look for natural micro-variations (3-8 Hz tremor range)
                energy_signal = np.array(energies) - np.mean(energies)
                
                # FFT of energy envelope
                fft_energy = np.abs(fft(energy_signal))
                freqs = fftfreq(len(energy_signal), d=hop/sr)
                
                # Energy in tremor band
                tremor_mask = (freqs > 3) & (freqs < 8)
                tremor_energy = np.mean(fft_energy[tremor_mask]) if np.any(tremor_mask) else 0
                total_energy = np.mean(fft_energy) + 1e-10
                
                tremor_ratio = tremor_energy / total_energy
                forensics.vocal_tremor = float(tremor_ratio)
                
                # Too little tremor is suspicious
                if tremor_ratio < 0.01:
                    scores.append(0.6)
                else:
                    scores.append(0.2)
        
        # 3. Formant stability
        # Check consistency of formant-like features (using spectral peaks)
        if features.spectral_contrast is not None:
            contrast_var = np.std(features.spectral_contrast)
            forensics.formant_stability = float(1 / (1 + contrast_var))
            
            # Too stable (perfect formants) is suspicious
            if forensics.formant_stability > 0.8:
                scores.append(0.5)
            else:
                scores.append(0.2)
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_temporal(
        self,
        samples: np.ndarray,
        sr: int,
        forensics: AudioForensicFeatures
    ) -> float:
        """Analyze temporal patterns."""
        scores = []
        
        frame_size = int(sr * 0.02)
        hop = int(sr * 0.01)
        
        # 1. Frame-to-frame correlation
        frames = []
        for i in range(0, len(samples) - frame_size, hop):
            frames.append(samples[i:i+frame_size])
        
        if len(frames) > 10:
            correlations = []
            for i in range(len(frames) - 1):
                corr = np.corrcoef(frames[i], frames[i+1])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            if correlations:
                mean_corr = np.mean(correlations)
                forensics.frame_correlation = float(mean_corr)
                
                # Too high correlation (frames too similar) is suspicious
                if mean_corr > 0.95:
                    scores.append(0.6)
                elif mean_corr < 0.3:  # Too different
                    scores.append(0.4)
                else:
                    scores.append(0.2)
        
        # 2. Attack smoothness (word onsets)
        # Natural speech has characteristic attack transients
        energy = np.array([np.mean(f ** 2) for f in frames]) if frames else np.array([])
        
        if len(energy) > 10:
            # Find energy peaks (word onsets)
            energy_diff = np.diff(energy)
            onset_indices = np.where(energy_diff > np.std(energy_diff) * 2)[0]
            
            if len(onset_indices) > 2:
                # Analyze attack slopes
                attack_slopes = []
                for idx in onset_indices:
                    if idx > 0 and idx < len(energy) - 1:
                        slope = energy[idx] - energy[idx-1]
                        attack_slopes.append(slope)
                
                if attack_slopes:
                    slope_var = np.std(attack_slopes)
                    forensics.attack_smoothness = float(1 / (1 + slope_var))
                    
                    # Very uniform attacks are suspicious
                    if forensics.attack_smoothness > 0.9:
                        scores.append(0.6)
                    else:
                        scores.append(0.2)
        
        # 3. Silence noise floor
        # Synthetic audio often has different silence characteristics
        silence_threshold = np.std(samples) * 0.1
        silent_samples = samples[np.abs(samples) < silence_threshold]
        
        if len(silent_samples) > sr * 0.1:  # At least 100ms of silence
            noise_floor = np.std(silent_samples)
            forensics.silence_noise_floor = float(noise_floor)
            
            # Very low noise floor (too clean) is suspicious
            if noise_floor < 1e-4:
                scores.append(0.6)
            else:
                scores.append(0.2)
        
        return np.mean(scores) if scores else 0.5
    
    def _neural_classify(
        self,
        samples: np.ndarray,
        sr: int,
        features: AudioFeatures
    ) -> float:
        """Use neural network for classification."""
        if self.model is None:
            return 0.5
        
        # Placeholder for actual neural inference
        # Would use mel spectrogram or raw waveform as input
        return 0.5
    
    def _compute_final_score(self, result: AudioDeepfakeResult) -> None:
        """Combine all signals into final probability."""
        scores = []
        weights = []
        
        if result.spectral_score > 0:
            scores.append(result.spectral_score)
            weights.append(self.weights['spectral'])
        
        if result.prosody_score > 0:
            scores.append(result.prosody_score)
            weights.append(self.weights['prosody'])
        
        if result.quality_score > 0:
            scores.append(result.quality_score)
            weights.append(self.weights['quality'])
        
        if result.temporal_score > 0:
            scores.append(result.temporal_score)
            weights.append(self.weights['temporal'])
        
        if self.use_neural and result.neural_score > 0:
            scores.append(result.neural_score)
            weights.append(self.weights['neural'])
        
        if scores:
            total_weight = sum(weights)
            result.synthetic_probability = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        # Confidence based on signal agreement
        if len(scores) > 1:
            score_std = np.std(scores)
            result.confidence = 1 - min(1, score_std * 2)
        else:
            result.confidence = 0.5
        
        result.is_synthetic = result.synthetic_probability > self.SYNTHETIC_THRESHOLD
    
    def _compute_verdict(self, result: AudioDeepfakeResult) -> None:
        """Compute human-readable verdict."""
        prob = result.synthetic_probability
        forensics = result.forensics
        
        # Identify anomalies
        if result.spectral_score > 0.5:
            result.anomalies.append("spectral_artifacts")
        if result.prosody_score > 0.5:
            result.anomalies.append("prosody_anomalies")
        if result.quality_score > 0.5:
            result.anomalies.append("voice_quality_issues")
        if result.temporal_score > 0.5:
            result.anomalies.append("temporal_irregularities")
        
        # Verdict
        if prob < 0.3:
            result.verdict = "AUTHENTIC"
            result.explanation = "Voice appears authentic. Natural characteristics detected."
        elif prob < 0.5:
            result.verdict = "LIKELY_AUTHENTIC"
            result.explanation = "Voice likely authentic, minor anomalies detected."
        elif prob < 0.7:
            result.verdict = "SUSPICIOUS"
            result.explanation = f"Potential synthetic voice. Anomalies: {', '.join(result.anomalies) or 'multiple signals'}"
        else:
            result.verdict = "LIKELY_SYNTHETIC"
            result.explanation = f"High probability of synthetic voice. Key indicators: {', '.join(result.anomalies) or 'multiple signals'}"


class VoicePrintMatcher:
    """
    Simple voice print matching for speaker verification.
    
    Note: This is a basic implementation. Production systems should
    use specialized speaker verification models (e.g., ECAPA-TDNN, ResNet).
    """
    
    def __init__(self, n_mfcc: int = 20):
        self.n_mfcc = n_mfcc
        self.enrolled_prints: Dict[str, np.ndarray] = {}
    
    def enroll(self, speaker_id: str, features: AudioFeatures) -> None:
        """Enroll a speaker with their voice features."""
        if features.mfccs is not None:
            # Simple voice print: mean and std of MFCCs
            mean_mfcc = np.mean(features.mfccs, axis=0)
            std_mfcc = np.std(features.mfccs, axis=0)
            voice_print = np.concatenate([mean_mfcc, std_mfcc])
            self.enrolled_prints[speaker_id] = voice_print
    
    def verify(
        self,
        claimed_id: str,
        features: AudioFeatures
    ) -> Tuple[bool, float]:
        """
        Verify if audio matches claimed speaker.
        
        Returns:
            Tuple of (is_match, confidence)
        """
        if claimed_id not in self.enrolled_prints:
            return False, 0.0
        
        if features.mfccs is None:
            return False, 0.0
        
        # Extract voice print from test sample
        mean_mfcc = np.mean(features.mfccs, axis=0)
        std_mfcc = np.std(features.mfccs, axis=0)
        test_print = np.concatenate([mean_mfcc, std_mfcc])
        
        # Compare with enrolled print
        enrolled_print = self.enrolled_prints[claimed_id]
        
        # Cosine similarity
        similarity = np.dot(test_print, enrolled_print) / (
            np.linalg.norm(test_print) * np.linalg.norm(enrolled_print) + 1e-10
        )
        
        # Threshold
        threshold = 0.85
        is_match = similarity > threshold
        confidence = float(similarity)
        
        return is_match, confidence


if __name__ == "__main__":
    print("Audio Deepfake Detection Demo")
    print("=" * 40)
    
    # Create synthetic test audio
    sample_rate = 16000
    duration = 3.0
    t = np.arange(int(sample_rate * duration)) / sample_rate
    
    # Simulate speech-like signal
    f0 = 150
    signal_audio = np.zeros_like(t)
    for harmonic in range(1, 10):
        signal_audio += (1 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    
    # Add variation
    signal_audio *= 1 + 0.3 * np.sin(2 * np.pi * 3 * t)
    
    # Add natural-like noise
    signal_audio += 0.05 * np.random.randn(len(t))
    
    # Normalize
    signal_audio = signal_audio / np.max(np.abs(signal_audio)) * 0.8
    signal_audio = signal_audio.astype(np.float32)
    
    # Create source
    from .audio_pipeline import AudioSource
    source = AudioSource.from_buffer(signal_audio, sample_rate)
    
    # Run detection
    detector = AudioDeepfakeDetector(use_neural=False)
    result = detector.analyze(source)
    
    print(f"\nResults:")
    print(f"  Verdict: {result.verdict}")
    print(f"  Synthetic Probability: {result.synthetic_probability:.1%}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"\nComponent Scores:")
    print(f"  Spectral: {result.spectral_score:.2f}")
    print(f"  Prosody: {result.prosody_score:.2f}")
    print(f"  Quality: {result.quality_score:.2f}")
    print(f"  Temporal: {result.temporal_score:.2f}")
    
    if result.anomalies:
        print(f"\nDetected Anomalies: {', '.join(result.anomalies)}")
    
    print(f"\nForensic Features:")
    print(f"  Spectral Flatness: {result.forensics.spectral_flatness:.3f}")
    print(f"  Breathiness: {result.forensics.breathiness:.3f}")
    print(f"  Vocal Tremor: {result.forensics.vocal_tremor:.4f}")
    print(f"  Naturalness Score: {result.forensics.naturalness_score:.2f}")
    
    print(f"\nExplanation: {result.explanation}")
    
    source.close()
