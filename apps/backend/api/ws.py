"""WebSocket endpoint for video frames."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import base64
import json
import time
import mediapipe as mp
from typing import List, Dict, Optional

from core.vision.face_detector import FaceDetector
from core.rppg.signal_extractor import SignalExtractor
from core.rppg.features import FeatureExtractor
from core.rppg.filters import BandpassFilter
from core.liveness.liveness import PhysioFeatures, compute_liveness_result
from core.liveness.audio_liveness import AudioLivenessDetector
from core.liveness.blink_detector import BlinkDetector
from core.liveness.motion_validator import MotionValidator
from core.liveness.texture_validator import TextureValidator

router = APIRouter()

class LivenessSession:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.signal_extractor = SignalExtractor()
        self.feature_extractor = FeatureExtractor(fs=30)
        self.bandpass_filter = BandpassFilter(fs=30)
        self.audio_detector = AudioLivenessDetector()
        
        # Advanced Liveness Detectors
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.blink_detector = BlinkDetector()
        self.motion_validator = MotionValidator()
        self.texture_validator = TextureValidator()

        # Reduced buffer size to 60 frames (2 seconds @ 30fps) for faster analysis
        self.buffer_size = 60 
        self.roi_buffers: Dict[str, List[np.ndarray]] = {
            "forehead": [], 
            "left_cheek": [], 
            "right_cheek": []
        }
        self.frame_count = 0
        self.is_completed = False
        self.final_result = None

    def process_audio(self, audio_bytes: bytes):
        if not self.is_completed:
            self.audio_detector.process_chunk(audio_bytes)

    def process_frame(self, frame: np.ndarray):
        if self.is_completed:
            return self.final_result

        self.frame_count += 1
        
        # 1. Detect Face (Haar) for ROI
        face_bbox = self.face_detector.detect(frame)
        if face_bbox is None:
            # Clear buffers if face lost to avoid mixing signals
            self._reset_buffers()
            return {
                "status": "no_face",
                "bbox": None
            }
            
        x, y, w, h = face_bbox
        
        # 2. Advanced Liveness (MediaPipe + Texture)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h_frame, w_frame, _ = frame.shape
            self.blink_detector.process(landmarks, w_frame, h_frame)
            self.motion_validator.process(landmarks, w_frame, h_frame)
            
        # Analyze Full Frame Texture (Background + Face)
        # This helps detect if the entire video is generated/filtered, not just the face
        self.texture_validator.process(frame, face_bbox=None)

        # 3. Extract ROIs for rPPG
        rois = self._get_rois(frame, face_bbox)
        
        # 4. Accumulate Means
        for name, (rx, ry, rw, rh) in rois.items():
            roi_patch = frame[ry:ry+rh, rx:rx+rw]
            if roi_patch.size == 0: continue
            
            mean_color = np.mean(roi_patch, axis=(0, 1))
            self.roi_buffers[name].append(mean_color)
            
            if len(self.roi_buffers[name]) > self.buffer_size:
                self.roi_buffers[name].pop(0)
        
        # 5. Check for Completion
        if len(self.roi_buffers["forehead"]) == self.buffer_size:
            return self._finalize_analysis(face_bbox)
        
        return {
            "status": "collecting",
            "bbox": [int(x), int(y), int(w), int(h)],
            "progress": min(1.0, len(self.roi_buffers["forehead"]) / self.buffer_size)
        }

    def _finalize_analysis(self, bbox):
        liveness_result = self._compute_liveness()
        audio_score = self.audio_detector.analyze()
        
        # Get Advanced Liveness Scores
        blink_score = self.blink_detector.get_score()
        motion_score = self.motion_validator.get_score()
        texture_score = self.texture_validator.get_score()
        
        # Fusion Logic
        rppg_score = liveness_result.final_score
        reasons = liveness_result.reasons
        
        # SNR Check (Signal-to-Noise Ratio)
        # Real pulses have high SNR. Noise (photos) has low SNR.
        snr = liveness_result.debug.get("physio_snr", 0)
        if snr < 2.0:
            reasons.append("Low physiological signal quality")
            rppg_score = min(rppg_score, 0.3) # Cap rPPG score if noisy
        
        # Weighted Video Score
        # Adjusted for 2s scan: Higher weight on Texture/rPPG, lower on Blink (might not happen in 2s)
        # rPPG: 40%, Texture: 30%, Motion: 20%, Blink: 10%
        video_score = (rppg_score * 0.4) + (texture_score * 0.3) + (motion_score * 0.2) + (blink_score * 0.1)
        
        if blink_score < 0.5:
            reasons.append("No blinking detected (Short scan)")
            # Relaxed Veto for 2s scan: Allow passing if other metrics are high
        else:
            reasons.append("Normal blinking detected")
            
        if motion_score < 0.5:
            reasons.append("Unnatural head motion")
            # Veto: If static image (photo), cap score
            video_score = min(video_score, 0.5)
            
        if texture_score < 0.5:
            reasons.append("Unnatural video texture (Smooth/Blurry)")
            # Veto: Likely AI generated or filter
            video_score = min(video_score, 0.6)
        
        final_score = video_score
        
        # Audio Fusion (Veto-based)
        if audio_score != 0.5: # 0.5 is "uncertain"
            if audio_score < 0.4:
                reasons.append("Synthetic/Robotic audio detected")
                final_score = min(final_score, audio_score)
            elif audio_score > 0.7:
                reasons.append("Natural audio detected")
                # Boost slightly if both are good
                final_score = (final_score * 0.6) + (audio_score * 0.4)
        
        # Detailed Classification
        classification = "UNCERTAIN"
        if final_score >= 0.75: # Slightly lowered threshold for REAL HUMAN
            classification = "REAL HUMAN"
        elif audio_score < 0.4 and video_score > 0.6:
            classification = "AI VOICE DETECTED"
        elif texture_score < 0.3: # Lowered threshold for texture failure
            classification = "AI VIDEO / DEEPFAKE"
        elif video_score < 0.4:
            classification = "AI VIDEO / DEEPFAKE"
        elif final_score < 0.6:
            classification = "POTENTIAL SPOOF"

        # Recalculate Level
        if final_score >= 0.75:
            level = "HIGH"
        elif final_score >= 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"

        x, y, w, h = bbox
        self.final_result = {
            "status": "completed",
            "bbox": [int(x), int(y), int(w), int(h)],
            "liveness": level,
            "classification": classification,
            "score": float(final_score),
            "video_score": float(video_score),
            "audio_score": audio_score,
            "bpm": liveness_result.debug.get("physio_bpm", 0),
            "snr": liveness_result.debug.get("physio_snr", 0),
            "reasons": reasons
        }
        self.is_completed = True
        return self.final_result

    def _reset_buffers(self):
        for k in self.roi_buffers:
            self.roi_buffers[k] = []

    def _get_rois(self, frame, bbox):
        x, y, w, h = bbox
        # Simple heuristic ROIs
        return {
            "forehead": (x + int(w*0.3), y + int(h*0.1), int(w*0.4), int(h*0.15)),
            "left_cheek": (x + int(w*0.15), y + int(h*0.55), int(w*0.2), int(h*0.15)),
            "right_cheek": (x + int(w*0.65), y + int(h*0.55), int(w*0.2), int(h*0.15))
        }

    def _compute_liveness(self):
        signals = {}
        for name, data in self.roi_buffers.items():
            if not data: continue
            data_np = np.array(data)
            raw = self.signal_extractor._pos(data_np)
            filtered = self.bandpass_filter.apply(raw)
            signals[name] = filtered
            
        # Extract features
        roi_features = {}
        bpms = []
        snrs = []
        ibi_cvs = []
        
        for name, sig in signals.items():
            feats = self.feature_extractor.extract(sig)
            roi_features[name] = feats
            if feats['hr_bpm'] > 0: bpms.append(feats['hr_bpm'])
            snrs.append(feats['snr'])
            ibi_cvs.append(feats['ibi_cv'])
            
        # Cross-ROI Correlation
        correlations = []
        names = list(signals.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                s1 = signals[names[i]]
                s2 = signals[names[j]]
                # Normalize
                if np.std(s1) > 0 and np.std(s2) > 0:
                    s1 = (s1 - np.mean(s1)) / np.std(s1)
                    s2 = (s2 - np.mean(s2)) / np.std(s2)
                    corr = np.corrcoef(s1, s2)[0, 1]
                    correlations.append(corr)
                    
        mean_corr = np.mean(correlations) if correlations else 0.0
        
        physio = PhysioFeatures(
            bpm_mean=float(np.mean(bpms)) if bpms else 0.0,
            bpm_std=float(np.std(bpms)) if bpms else 0.0,
            snr_mean=float(np.mean(snrs)) if snrs else 0.0,
            snr_std=float(np.std(snrs)) if snrs else 0.0,
            cross_roi_corr_mean=float(mean_corr),
            ibi_cv=float(np.mean(ibi_cvs)) if ibi_cvs else 0.0,
            roi_features=roi_features
        )
        
        return compute_liveness_result(physio, [])

@router.websocket("/ws/liveness")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = LivenessSession()
    
    try:
        while True:
            # Expecting JSON with base64 image: {"image": "base64string...", "audio": "base64string..."}
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                
                # Handle Audio
                audio_b64 = payload.get("audio")
                if audio_b64:
                    try:
                        # Remove header if present
                        if "," in audio_b64:
                            audio_b64 = audio_b64.split(",")[1]
                        audio_bytes = base64.b64decode(audio_b64)
                        session.process_audio(audio_bytes)
                    except Exception as e:
                        print(f"Error processing audio: {e}")

                image_b64 = payload.get("image")
                
                if not image_b64:
                    continue
                    
                # Decode image
                # Remove header if present (e.g., "data:image/jpeg;base64,")
                if "," in image_b64:
                    image_b64 = image_b64.split(",")[1]
                    
                image_bytes = base64.b64decode(image_b64)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Process
                result = session.process_frame(frame)
                
                # Send back result
                await websocket.send_json(result)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("Client disconnected")
