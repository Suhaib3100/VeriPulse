import numpy as np
import cv2
from core.vision.video_reader import VideoReader
from core.vision.face_tracker import FaceTracker
from core.vision.roi_tracker import ROITracker
from core.rppg.signal_extractor import SignalExtractor
from core.rppg.filters import BandpassFilter
from core.rppg.features import FeatureExtractor
from core.rppg.quality_metrics import QualityAnalyzer

class RPPGProcessor:
    def __init__(self, fs=30, method='pos', buffer_size=300):
        self.fs = fs
        self.method = method
        self.buffer_size = buffer_size
        self.face_tracker = FaceTracker()
        self.roi_tracker = ROITracker()
        self.signal_extractor = SignalExtractor()
        self.filter = BandpassFilter(fs=fs)
        self.feature_extractor = FeatureExtractor(fs=fs)
        self.quality_analyzer = QualityAnalyzer()
        
        # State for real-time processing
        self.buffers = {
            "forehead": [],
            "left_cheek": [],
            "right_cheek": []
        }

    def process_frame(self, frame):
        """
        Process a single frame for real-time analysis.
        
        Args:
            frame: Input video frame.
            
        Returns:
            dict: Current analysis results (bbox, liveness, features).
        """
        # Detect and track face
        face_box = self.face_tracker.process_frame(frame)
        
        result = {
            "bbox": face_box,
            "liveness_score": 0.0,
            "label": "WAITING",
            "bpm": 0.0,
            "snr": 0.0
        }
        
        if not face_box:
            # Clear buffers or append None? 
            # For robustness, maybe clear if face lost for too long.
            # For now, just return empty result.
            return result
            
        # Extract ROIs
        rois = self.roi_tracker.extract_rois(frame, face_box)
        
        # Update buffers
        for key in self.buffers:
            if key in rois:
                self.buffers[key].append(rois[key])
            else:
                self.buffers[key].append(None)
                
            # Maintain buffer size
            if len(self.buffers[key]) > self.buffer_size:
                self.buffers[key].pop(0)
                
        # Check if we have enough data
        if len(self.buffers["forehead"]) < self.fs * 2: # Need at least 2 seconds
            return result
            
        # Process signals
        signals = {}
        roi_features = {}
        
        for roi_name, roi_frames in self.buffers.items():
            # Extract raw signal
            # Note: SignalExtractor.extract expects a list of frames
            raw_signal = self.signal_extractor.extract(roi_frames, method=self.method)
            
            # Filter signal
            filtered_signal = self.filter.apply(raw_signal)
            signals[roi_name] = filtered_signal
            
            # Extract features
            feats = self.feature_extractor.extract(filtered_signal)
            roi_features[f"{roi_name}_features"] = feats
            
        # Compute Consistency & Liveness
        consistency = self._compute_consistency(signals, roi_features)
        result["consistency"] = consistency
        
        liveness_score, label = self._classify_liveness({**roi_features, "consistency": consistency})
        result["liveness_score"] = liveness_score
        result["label"] = label
        
        # Aggregate BPM (mean of valid ROIs)
        bpms = [f["hr_bpm"] for f in roi_features.values() if f["hr_bpm"] > 0]
        if bpms:
            result["bpm"] = np.mean(bpms)
            
        result["snr"] = np.mean([f["snr"] for f in roi_features.values()])
        
        return result

    def process_video(self, source, duration=None):
        """
        Process a video source to extract rPPG features.
        
        Args:
            source: Webcam index or file path.
            duration: Max duration to process in seconds (optional).
            
        Returns:
            dict: Aggregated features and liveness score.
        """
        reader = VideoReader(source, target_fps=self.fs)
        
        frames = []
        rois_data = {
            "forehead": [],
            "left_cheek": [],
            "right_cheek": []
        }
        
        frame_count = 0
        max_frames = int(duration * self.fs) if duration else float('inf')
        
        try:
            for frame, timestamp in reader:
                if frame_count >= max_frames:
                    break
                    
                # Detect and track face
                face_box = self.face_tracker.process_frame(frame)
                
                if face_box:
                    # Extract ROIs
                    rois = self.roi_tracker.extract_rois(frame, face_box)
                    
                    for key in rois_data:
                        if key in rois:
                            rois_data[key].append(rois[key])
                        else:
                            # Handle missing ROI?
                            pass
                else:
                    # Face lost, maybe append None or skip?
                    # For signal continuity, we might want to append None or last valid?
                    # SignalExtractor handles None/empty frames by appending 0.
                    for key in rois_data:
                        rois_data[key].append(None)
                
                frame_count += 1
                
        except StopIteration:
            pass
        finally:
            reader.release()
            
        if frame_count == 0:
            return {"error": "No frames processed"}

        # Process signals for each ROI
        results = {}
        signals = {}
        
        for roi_name, roi_frames in rois_data.items():
            # Extract raw signal
            raw_signal = self.signal_extractor.extract(roi_frames, method=self.method)
            
            # Filter signal
            filtered_signal = self.filter.apply(raw_signal)
            signals[roi_name] = filtered_signal
            
            # Extract features
            feats = self.feature_extractor.extract(filtered_signal)
            results[f"{roi_name}_features"] = feats

        # Compute Cross-ROI Consistency
        consistency = self._compute_consistency(signals, results)
        results["consistency"] = consistency
        
        # Aggregate Score
        liveness_score, label = self._classify_liveness(results)
        results["liveness_score"] = liveness_score
        results["label"] = label
        
        return results

    def _compute_consistency(self, signals, roi_features):
        """
        Compute consistency metrics across ROIs.
        """
        roi_names = list(signals.keys())
        if len(roi_names) < 2:
            return {"mean_correlation": 0.0, "bpm_agreement": 0.0}
            
        correlations = []
        bpm_diffs = []
        
        for i in range(len(roi_names)):
            for j in range(i + 1, len(roi_names)):
                name1, name2 = roi_names[i], roi_names[j]
                s1 = signals[name1]
                s2 = signals[name2]
                
                # Correlation
                min_len = min(len(s1), len(s2))
                if min_len > 0:
                    # Normalize
                    n1 = (s1[:min_len] - np.mean(s1[:min_len]))
                    n2 = (s2[:min_len] - np.mean(s2[:min_len]))
                    if np.std(n1) > 0 and np.std(n2) > 0:
                        c = np.corrcoef(n1, n2)[0, 1]
                        correlations.append(c)
                
                # BPM Agreement
                bpm1 = roi_features.get(f"{name1}_features", {}).get("hr_bpm", 0)
                bpm2 = roi_features.get(f"{name2}_features", {}).get("hr_bpm", 0)
                if bpm1 > 0 and bpm2 > 0:
                    bpm_diffs.append(abs(bpm1 - bpm2))
                    
        mean_corr = np.mean(correlations) if correlations else 0.0
        
        agreement_rate = 0.0
        if bpm_diffs:
            agreement_count = sum(1 for d in bpm_diffs if d < 5.0)
            agreement_rate = agreement_count / len(bpm_diffs)
            
        return {
            "mean_correlation": float(mean_corr),
            "bpm_agreement": float(agreement_rate)
        }

    def _classify_liveness(self, results):
        """
        Heuristic liveness classification.
        """
        consistency = results.get("consistency", {})
        mean_corr = consistency.get("mean_correlation", 0.0)
        bpm_agreement = consistency.get("bpm_agreement", 0.0)
        
        # Average SNR
        snrs = []
        for key, val in results.items():
            if key.endswith("_features"):
                snrs.append(val.get("snr", 0))
        
        avg_snr = np.mean(snrs) if snrs else 0.0
        
        # Score calculation
        # Weights: Correlation 40%, Agreement 30%, SNR 30%
        s_corr = max(0, mean_corr)
        s_agree = bpm_agreement
        s_snr = np.clip(avg_snr / 5.0, 0, 1) # Assume SNR > 5 is good
        
        score = 0.4 * s_corr + 0.3 * s_agree + 0.3 * s_snr
        
        label = "LIVE" if score > 0.5 else "SUSPECT"
        
        return float(score), label
        # Use forehead as primary for now, or average?
        # Let's average the SNR and Periodicity
        avg_snr = np.mean([results[f"{k}_features"]["snr"] for k in rois_data])
        avg_periodicity = np.mean([results[f"{k}_features"]["periodicity"] for k in rois_data])
        
        # Final Liveness Score
        # Weighted sum of SNR, Periodicity, and Consistency
        # Tunable weights
        score = 0.4 * avg_snr/4.0 + 0.3 * avg_periodicity + 0.3 * consistency
        score = float(np.clip(score, 0, 1))
        
        results["liveness_score"] = score
        
        return results

    def _compute_consistency(self, signals):
        """Compute consistency (correlation) between ROI signals."""
        keys = list(signals.keys())
        if len(keys) < 2:
            return 0.0
            
        corrs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                s1 = signals[keys[i]]
                s2 = signals[keys[j]]
                
                # Normalize
                s1 = (s1 - np.mean(s1))
                s2 = (s2 - np.mean(s2))
                
                std1 = np.std(s1)
                std2 = np.std(s2)
                
                if std1 > 0 and std2 > 0:
                    s1 /= std1
                    s2 /= std2
                    corr = np.corrcoef(s1, s2)[0, 1]
                    corrs.append(corr)
                    
        if not corrs:
            return 0.0
            
        return float(max(0, np.mean(corrs)))
