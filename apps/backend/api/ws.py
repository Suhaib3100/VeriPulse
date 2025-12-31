"""WebSocket endpoint for video frames."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import base64
import json
import time
from typing import List, Dict

from core.vision.face_detector import FaceDetector
from core.rppg.signal_extractor import SignalExtractor
from core.rppg.features import FeatureExtractor
from core.rppg.filters import BandpassFilter
from core.liveness.liveness import PhysioFeatures, compute_liveness_result

router = APIRouter()

class LivenessSession:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.signal_extractor = SignalExtractor()
        self.feature_extractor = FeatureExtractor(fs=30)
        self.bandpass_filter = BandpassFilter(fs=30)
        
        self.buffer_size = 150 # 5 seconds @ 30fps
        self.roi_buffers: Dict[str, List[np.ndarray]] = {
            "forehead": [], 
            "left_cheek": [], 
            "right_cheek": []
        }
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray):
        self.frame_count += 1
        
        # 1. Detect Face
        face_bbox = self.face_detector.detect(frame)
        if not face_bbox:
            # Clear buffers if face lost to avoid mixing signals
            self._reset_buffers()
            return {
                "status": "no_face",
                "bbox": None
            }
            
        x, y, w, h = face_bbox
        
        # 2. Extract ROIs
        rois = self._get_rois(frame, face_bbox)
        
        # 3. Accumulate Means
        for name, (rx, ry, rw, rh) in rois.items():
            roi_patch = frame[ry:ry+rh, rx:rx+rw]
            if roi_patch.size == 0: continue
            
            mean_color = np.mean(roi_patch, axis=(0, 1))
            self.roi_buffers[name].append(mean_color)
            
            if len(self.roi_buffers[name]) > self.buffer_size:
                self.roi_buffers[name].pop(0)
        
        # 4. Process if buffer full
        result_data = {
            "status": "collecting",
            "bbox": [int(x), int(y), int(w), int(h)],
            "progress": min(1.0, len(self.roi_buffers["forehead"]) / self.buffer_size)
        }
        
        if len(self.roi_buffers["forehead"]) == self.buffer_size:
            liveness_result = self._compute_liveness()
            result_data.update({
                "status": "analyzed",
                "liveness": liveness_result.level,
                "score": liveness_result.final_score,
                "bpm": liveness_result.debug.get("physio_bpm", 0),
                "snr": liveness_result.debug.get("physio_snr", 0),
                "reasons": liveness_result.reasons
            })
            
        return result_data

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
            # Expecting JSON with base64 image: {"image": "base64string..."}
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
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
