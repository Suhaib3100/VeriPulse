"""
Audio Pipeline - Audio input abstraction and feature extraction.

This module provides:
- AudioSource: Unified interface for microphone, files, streams
- Feature extraction: MFCCs, mel spectrograms, prosody, spectral features
- Voice Activity Detection (VAD)
- Audio quality metrics
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Generator, Any, Union
from enum import Enum
from pathlib import Path
import struct


class AudioSourceType(Enum):
    """Types of audio sources."""
    MICROPHONE = "microphone"
    FILE = "file"
    STREAM = "stream"
    BUFFER = "buffer"


@dataclass
class AudioChunk:
    """A chunk of audio data with metadata."""
    
    samples: np.ndarray          # Audio samples (1D or 2D for stereo)
    sample_rate: int             # Sample rate in Hz
    timestamp: float = 0.0       # Timestamp in seconds
    chunk_index: int = 0         # Sequential chunk index
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.samples) / self.sample_rate
    
    @property
    def is_stereo(self) -> bool:
        """Check if audio is stereo."""
        return len(self.samples.shape) > 1 and self.samples.shape[1] == 2
    
    def to_mono(self) -> np.ndarray:
        """Convert to mono if stereo."""
        if self.is_stereo:
            return np.mean(self.samples, axis=1)
        return self.samples
    
    def normalize(self) -> np.ndarray:
        """Normalize to [-1, 1] range."""
        mono = self.to_mono()
        max_val = np.max(np.abs(mono))
        if max_val > 0:
            return mono / max_val
        return mono


@dataclass
class AudioFeatures:
    """Extracted audio features for analysis."""
    
    # Basic properties
    duration: float = 0.0            # Total duration in seconds
    sample_rate: int = 16000         # Sample rate
    
    # Spectral features
    mfccs: Optional[np.ndarray] = None           # MFCCs (N_frames x N_mfcc)
    mel_spectrogram: Optional[np.ndarray] = None # Log-mel spectrogram
    spectral_centroid: float = 0.0               # Average spectral centroid
    spectral_bandwidth: float = 0.0              # Average spectral bandwidth
    spectral_rolloff: float = 0.0                # Average spectral rolloff
    spectral_flatness: float = 0.0               # Average spectral flatness
    spectral_contrast: Optional[np.ndarray] = None  # Spectral contrast
    
    # Prosody features
    pitch_mean: float = 0.0          # Average pitch (F0) in Hz
    pitch_std: float = 0.0           # Pitch standard deviation
    pitch_range: float = 0.0         # Pitch range
    energy_mean: float = 0.0         # Average energy/loudness
    energy_std: float = 0.0          # Energy variation
    speech_rate: float = 0.0         # Estimated speech rate
    
    # Voice quality
    jitter: float = 0.0              # Pitch perturbation
    shimmer: float = 0.0             # Amplitude perturbation
    hnr: float = 0.0                 # Harmonics-to-noise ratio
    
    # Temporal features
    zero_crossing_rate: float = 0.0  # Average zero crossing rate
    tempo: float = 0.0               # Estimated tempo/rhythm
    
    # Voice activity
    vad_ratio: float = 0.0           # Ratio of voiced frames
    silence_ratio: float = 0.0       # Ratio of silent frames
    
    # Quality metrics
    snr: float = 0.0                 # Signal-to-noise ratio
    clipping_ratio: float = 0.0      # Ratio of clipped samples


class AudioSource:
    """
    Unified audio input abstraction.
    
    Supports reading from microphone, files, and streams with consistent interface.
    
    Example:
        >>> # From file
        >>> source = AudioSource.from_file("audio.wav")
        >>> for chunk in source.stream_chunks(chunk_size=1024):
        ...     process(chunk)
        
        >>> # From microphone
        >>> source = AudioSource.from_microphone(sample_rate=16000)
        >>> source.start()
        >>> chunk = source.get_chunk()
    """
    
    def __init__(
        self,
        source_type: AudioSourceType,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        self.source_type = source_type
        self.sample_rate = sample_rate
        self.channels = channels
        
        self._samples: Optional[np.ndarray] = None
        self._position: int = 0
        self._is_open: bool = False
        self._stream = None
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'AudioSource':
        """
        Create AudioSource from audio file.
        
        Supports: WAV, MP3 (if librosa available), other formats via scipy
        """
        path = Path(file_path)
        source = cls(AudioSourceType.FILE)
        
        if path.suffix.lower() == '.wav':
            sample_rate, samples = wavfile.read(path)
            source.sample_rate = sample_rate
            
            # Convert to float
            if samples.dtype == np.int16:
                samples = samples.astype(np.float32) / 32768.0
            elif samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 2147483648.0
            elif samples.dtype == np.uint8:
                samples = (samples.astype(np.float32) - 128) / 128.0
            
            source._samples = samples
            source.channels = 2 if len(samples.shape) > 1 else 1
            
        else:
            # Try librosa for other formats
            try:
                import librosa
                samples, sample_rate = librosa.load(path, sr=None, mono=False)
                source._samples = samples.T if len(samples.shape) > 1 else samples
                source.sample_rate = sample_rate
                source.channels = samples.shape[0] if len(samples.shape) > 1 else 1
            except ImportError:
                raise ValueError(f"Cannot read {path.suffix} files without librosa")
        
        source._is_open = True
        return source
    
    @classmethod
    def from_buffer(cls, samples: np.ndarray, sample_rate: int = 16000) -> 'AudioSource':
        """Create AudioSource from numpy array."""
        source = cls(AudioSourceType.BUFFER, sample_rate=sample_rate)
        source._samples = samples.astype(np.float32)
        source.channels = 2 if len(samples.shape) > 1 else 1
        source._is_open = True
        return source
    
    @classmethod
    def from_microphone(cls, sample_rate: int = 16000, channels: int = 1) -> 'AudioSource':
        """Create AudioSource from microphone input."""
        source = cls(AudioSourceType.MICROPHONE, sample_rate=sample_rate, channels=channels)
        return source
    
    def start(self) -> None:
        """Start audio capture (for microphone/stream sources)."""
        if self.source_type == AudioSourceType.MICROPHONE:
            try:
                import sounddevice as sd
                self._buffer = []
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    callback=self._audio_callback
                )
                self._stream.start()
                self._is_open = True
            except ImportError:
                raise RuntimeError("sounddevice required for microphone input")
    
    def stop(self) -> None:
        """Stop audio capture."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._is_open = False
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if hasattr(self, '_buffer'):
            self._buffer.append(indata.copy())
    
    def get_all_samples(self) -> np.ndarray:
        """Get all audio samples."""
        if self._samples is not None:
            return self._samples
        
        if hasattr(self, '_buffer') and self._buffer:
            return np.concatenate(self._buffer, axis=0)
        
        return np.array([])
    
    def stream_chunks(
        self,
        chunk_duration: float = 0.5,
        overlap: float = 0.0
    ) -> Generator[AudioChunk, None, None]:
        """
        Stream audio in chunks.
        
        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks (0-1)
            
        Yields:
            AudioChunk objects
        """
        if self._samples is None:
            return
        
        chunk_size = int(chunk_duration * self.sample_rate)
        hop_size = int(chunk_size * (1 - overlap))
        
        position = 0
        chunk_index = 0
        
        while position + chunk_size <= len(self._samples):
            chunk_samples = self._samples[position:position + chunk_size]
            
            yield AudioChunk(
                samples=chunk_samples,
                sample_rate=self.sample_rate,
                timestamp=position / self.sample_rate,
                chunk_index=chunk_index
            )
            
            position += hop_size
            chunk_index += 1
        
        # Handle remaining samples
        if position < len(self._samples):
            remaining = self._samples[position:]
            # Pad to chunk size
            padded = np.zeros(chunk_size, dtype=remaining.dtype)
            padded[:len(remaining)] = remaining
            
            yield AudioChunk(
                samples=padded,
                sample_rate=self.sample_rate,
                timestamp=position / self.sample_rate,
                chunk_index=chunk_index
            )
    
    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        samples = self.get_all_samples()
        return len(samples) / self.sample_rate if len(samples) > 0 else 0.0
    
    def close(self) -> None:
        """Clean up resources."""
        self.stop()
        self._samples = None


class AudioFeatureExtractor:
    """
    Extract comprehensive audio features for deepfake detection.
    
    Example:
        >>> extractor = AudioFeatureExtractor()
        >>> source = AudioSource.from_file("speech.wav")
        >>> features = extractor.extract(source)
        >>> print(f"MFCCs shape: {features.mfccs.shape}")
    """
    
    # Feature extraction parameters
    N_MFCC = 20
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048
    
    def __init__(
        self,
        n_mfcc: int = 20,
        n_mels: int = 128,
        use_librosa: bool = True
    ):
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.use_librosa = use_librosa
        
        # Check if librosa is available
        self._has_librosa = False
        try:
            import librosa
            self._has_librosa = True
        except ImportError:
            if use_librosa:
                print("Warning: librosa not available, using basic feature extraction")
    
    def extract(self, source: AudioSource) -> AudioFeatures:
        """Extract all features from audio source."""
        samples = source.get_all_samples()
        
        if len(samples) == 0:
            return AudioFeatures()
        
        # Convert to mono if needed
        if len(samples.shape) > 1:
            samples = np.mean(samples, axis=1)
        
        features = AudioFeatures(
            duration=len(samples) / source.sample_rate,
            sample_rate=source.sample_rate
        )
        
        # Extract features
        if self._has_librosa:
            self._extract_with_librosa(samples, source.sample_rate, features)
        else:
            self._extract_basic(samples, source.sample_rate, features)
        
        # Quality metrics (always compute)
        self._extract_quality_metrics(samples, features)
        
        return features
    
    def _extract_with_librosa(
        self,
        samples: np.ndarray,
        sr: int,
        features: AudioFeatures
    ) -> None:
        """Extract features using librosa."""
        import librosa
        
        # Ensure float32
        samples = samples.astype(np.float32)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=samples, sr=sr, n_mfcc=self.n_mfcc,
            hop_length=self.HOP_LENGTH, n_fft=self.N_FFT
        )
        features.mfccs = mfccs.T  # (time, n_mfcc)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=samples, sr=sr, n_mels=self.n_mels,
            hop_length=self.HOP_LENGTH, n_fft=self.N_FFT
        )
        features.mel_spectrogram = librosa.power_to_db(mel_spec, ref=np.max).T
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=samples, sr=sr, hop_length=self.HOP_LENGTH)
        features.spectral_centroid = float(np.mean(spec_cent))
        
        spec_bw = librosa.feature.spectral_bandwidth(y=samples, sr=sr, hop_length=self.HOP_LENGTH)
        features.spectral_bandwidth = float(np.mean(spec_bw))
        
        spec_rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr, hop_length=self.HOP_LENGTH)
        features.spectral_rolloff = float(np.mean(spec_rolloff))
        
        spec_flat = librosa.feature.spectral_flatness(y=samples, hop_length=self.HOP_LENGTH)
        features.spectral_flatness = float(np.mean(spec_flat))
        
        spec_contrast = librosa.feature.spectral_contrast(y=samples, sr=sr, hop_length=self.HOP_LENGTH)
        features.spectral_contrast = np.mean(spec_contrast, axis=1)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=samples, hop_length=self.HOP_LENGTH)
        features.zero_crossing_rate = float(np.mean(zcr))
        
        # Pitch extraction (F0)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                samples, fmin=50, fmax=500, sr=sr,
                hop_length=self.HOP_LENGTH
            )
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                features.pitch_mean = float(np.mean(valid_f0))
                features.pitch_std = float(np.std(valid_f0))
                features.pitch_range = float(np.max(valid_f0) - np.min(valid_f0))
                features.vad_ratio = float(np.sum(voiced_flag) / len(voiced_flag))
        except Exception:
            pass
        
        # Energy/RMS
        rms = librosa.feature.rms(y=samples, hop_length=self.HOP_LENGTH)
        features.energy_mean = float(np.mean(rms))
        features.energy_std = float(np.std(rms))
        
        # Tempo estimation
        try:
            tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
            features.tempo = float(tempo)
        except Exception:
            pass
    
    def _extract_basic(
        self,
        samples: np.ndarray,
        sr: int,
        features: AudioFeatures
    ) -> None:
        """Basic feature extraction without librosa."""
        # Compute STFT
        hop_length = self.HOP_LENGTH
        n_fft = self.N_FFT
        
        # Frame the signal
        n_frames = 1 + (len(samples) - n_fft) // hop_length
        if n_frames < 1:
            return
        
        frames = np.zeros((n_frames, n_fft))
        for i in range(n_frames):
            start = i * hop_length
            frames[i] = samples[start:start + n_fft]
        
        # Apply window
        window = np.hanning(n_fft)
        frames = frames * window
        
        # FFT
        spectrogram = np.abs(np.fft.rfft(frames, axis=1))
        
        # Basic MFCCs using mel filterbank
        n_mels = self.n_mels
        mel_filters = self._create_mel_filterbank(sr, n_fft, n_mels)
        mel_spec = np.dot(spectrogram, mel_filters.T)
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel = np.log(mel_spec)
        
        # DCT to get MFCCs
        features.mfccs = self._dct(log_mel, self.n_mfcc)
        features.mel_spectrogram = log_mel
        
        # Spectral centroid
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        features.spectral_centroid = float(np.mean(
            np.sum(spectrogram * freqs, axis=1) / (np.sum(spectrogram, axis=1) + 1e-10)
        ))
        
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples))
        features.zero_crossing_rate = float(zcr)
        
        # Energy
        features.energy_mean = float(np.mean(np.sqrt(np.mean(frames**2, axis=1))))
        features.energy_std = float(np.std(np.sqrt(np.mean(frames**2, axis=1))))
    
    def _create_mel_filterbank(
        self,
        sr: int,
        n_fft: int,
        n_mels: int
    ) -> np.ndarray:
        """Create mel filterbank matrix."""
        # Frequency to mel conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Mel points
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(sr / 2)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bins
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        
        # Create filterbank
        n_bins = n_fft // 2 + 1
        filterbank = np.zeros((n_mels, n_bins))
        
        for m in range(1, n_mels + 1):
            f_left = bin_points[m - 1]
            f_center = bin_points[m]
            f_right = bin_points[m + 1]
            
            for k in range(f_left, f_center):
                if f_center != f_left:
                    filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
            
            for k in range(f_center, f_right):
                if f_right != f_center:
                    filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
        
        return filterbank
    
    def _dct(self, x: np.ndarray, n_out: int) -> np.ndarray:
        """Compute DCT for MFCC computation."""
        n_in = x.shape[1]
        n = np.arange(n_in)
        k = np.arange(n_out)
        
        # DCT-II
        dct_matrix = np.cos(np.pi / n_in * (n + 0.5) * k[:, np.newaxis])
        dct_matrix *= np.sqrt(2 / n_in)
        dct_matrix[0] *= 1 / np.sqrt(2)
        
        return np.dot(x, dct_matrix.T)
    
    def _extract_quality_metrics(self, samples: np.ndarray, features: AudioFeatures) -> None:
        """Extract audio quality metrics."""
        # SNR estimation (simple method)
        # Estimate noise from quiet segments
        frame_size = 1024
        n_frames = len(samples) // frame_size
        
        if n_frames < 2:
            return
        
        frame_energies = []
        for i in range(n_frames):
            frame = samples[i * frame_size:(i + 1) * frame_size]
            frame_energies.append(np.mean(frame ** 2))
        
        frame_energies = np.array(frame_energies)
        
        # Assume bottom 10% are noise
        noise_threshold = np.percentile(frame_energies, 10)
        signal_power = np.mean(frame_energies)
        noise_power = noise_threshold
        
        if noise_power > 0:
            features.snr = float(10 * np.log10(signal_power / noise_power))
        
        # Silence ratio
        silence_threshold = 0.01 * np.max(np.abs(samples))
        features.silence_ratio = float(np.mean(np.abs(samples) < silence_threshold))
        
        # Clipping detection
        max_val = 0.99  # Near clipping threshold
        features.clipping_ratio = float(np.mean(np.abs(samples) > max_val))


class VoiceActivityDetector:
    """
    Simple Voice Activity Detection (VAD).
    
    Detects speech vs non-speech segments.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.energy_threshold = energy_threshold
    
    def detect(self, samples: np.ndarray) -> List[Tuple[float, float, bool]]:
        """
        Detect voice activity in audio.
        
        Returns:
            List of (start_time, end_time, is_speech) tuples
        """
        results = []
        
        n_frames = len(samples) // self.frame_size
        
        for i in range(n_frames):
            start_sample = i * self.frame_size
            end_sample = start_sample + self.frame_size
            frame = samples[start_sample:end_sample]
            
            # Energy-based detection
            energy = np.mean(frame ** 2)
            is_speech = energy > self.energy_threshold
            
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            results.append((start_time, end_time, is_speech))
        
        # Merge consecutive segments
        merged = []
        if results:
            current_start, current_end, current_state = results[0]
            
            for start, end, state in results[1:]:
                if state == current_state:
                    current_end = end
                else:
                    merged.append((current_start, current_end, current_state))
                    current_start, current_end, current_state = start, end, state
            
            merged.append((current_start, current_end, current_state))
        
        return merged
    
    def get_speech_segments(self, samples: np.ndarray) -> List[np.ndarray]:
        """Extract only speech segments."""
        segments = self.detect(samples)
        speech_audio = []
        
        for start_time, end_time, is_speech in segments:
            if is_speech:
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                speech_audio.append(samples[start_sample:end_sample])
        
        return speech_audio


if __name__ == "__main__":
    print("Audio Pipeline Demo")
    print("=" * 40)
    
    # Create synthetic audio for testing
    sample_rate = 16000
    duration = 3.0
    t = np.arange(int(sample_rate * duration)) / sample_rate
    
    # Simulate speech-like signal (sum of harmonics)
    f0 = 150  # Fundamental frequency
    signal_audio = np.zeros_like(t)
    for harmonic in range(1, 10):
        signal_audio += (1 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    
    # Add amplitude envelope (words)
    envelope = np.abs(np.sin(2 * np.pi * 2 * t))  # ~2 "words" per second
    signal_audio *= envelope
    
    # Add noise
    signal_audio += 0.1 * np.random.randn(len(t))
    
    # Normalize
    signal_audio = signal_audio / np.max(np.abs(signal_audio)) * 0.9
    signal_audio = signal_audio.astype(np.float32)
    
    # Create source from buffer
    source = AudioSource.from_buffer(signal_audio, sample_rate)
    
    print(f"Audio duration: {source.duration:.2f}s")
    print(f"Sample rate: {source.sample_rate} Hz")
    
    # Extract features
    extractor = AudioFeatureExtractor()
    features = extractor.extract(source)
    
    print(f"\nExtracted Features:")
    print(f"  MFCCs shape: {features.mfccs.shape if features.mfccs is not None else 'N/A'}")
    print(f"  Mel spectrogram shape: {features.mel_spectrogram.shape if features.mel_spectrogram is not None else 'N/A'}")
    print(f"  Spectral centroid: {features.spectral_centroid:.1f} Hz")
    print(f"  Zero crossing rate: {features.zero_crossing_rate:.4f}")
    print(f"  Energy mean: {features.energy_mean:.4f}")
    print(f"  SNR: {features.snr:.1f} dB")
    print(f"  Silence ratio: {features.silence_ratio:.1%}")
    
    # Voice activity detection
    vad = VoiceActivityDetector(sample_rate=sample_rate)
    vad_segments = vad.detect(signal_audio)
    
    print(f"\nVAD detected {len(vad_segments)} segments")
    speech_segments = [(s, e) for s, e, is_speech in vad_segments if is_speech]
    print(f"  Speech segments: {len(speech_segments)}")
    
    # Test streaming chunks
    print(f"\nStreaming test:")
    chunk_count = 0
    for chunk in source.stream_chunks(chunk_duration=0.5):
        chunk_count += 1
    print(f"  Generated {chunk_count} chunks")
    
    source.close()
