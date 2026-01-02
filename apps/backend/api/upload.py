from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import cv2
import numpy as np
import tempfile
import librosa
import mediapipe as mp
from typing import Dict, List

from core.vision.face_detector import FaceDetector
from core.rppg.signal_extractor import SignalExtractor
from core.rppg.features import FeatureExtractor
from core.rppg.filters import BandpassFilter
from core.liveness.audio_liveness import AudioLivenessDetector
from core.liveness.blink_detector import BlinkDetector
from core.liveness.motion_validator import MotionValidator
from core.liveness.texture_validator import TextureValidator

router = APIRouter()

@router.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Initialize Detectors
        face_detector = FaceDetector()
        signal_extractor = SignalExtractor()
        bandpass_filter = BandpassFilter(fs=30)
        audio_detector = AudioLivenessDetector()
        
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        blink_detector = BlinkDetector()
        motion_validator = MotionValidator()
        texture_validator = TextureValidator()

        # Process Video
        cap = cv2.VideoCapture(tmp_path)
        
        roi_buffers = {"forehead": [], "left_cheek": [], "right_cheek": []}
        frame_count = 0
        faces_detected_count = 0
        max_frames = 150 # Analyze up to 5 seconds
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Face Detection
            face_bbox = face_detector.detect(frame)
            
            # Texture Analysis (Full Frame)
            texture_validator.process(frame, face_bbox=None)
            
            if face_bbox is not None:
                faces_detected_count += 1
                x, y, w, h = face_bbox
                
                # MediaPipe Liveness
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    h_frame, w_frame, _ = frame.shape
                    blink_detector.process(landmarks, w_frame, h_frame)
                    motion_validator.process(landmarks, w_frame, h_frame)
                
                # rPPG Extraction
                rois = {
                    "forehead": (x + int(w*0.3), y + int(h*0.1), int(w*0.4), int(h*0.15)),
                    "left_cheek": (x + int(w*0.15), y + int(h*0.55), int(w*0.2), int(h*0.15)),
                    "right_cheek": (x + int(w*0.65), y + int(h*0.55), int(w*0.2), int(h*0.15))
                }
                
                for name, (rx, ry, rw, rh) in rois.items():
                    roi_patch = frame[ry:ry+rh, rx:rx+rw]
                    if roi_patch.size > 0:
                        mean_color = np.mean(roi_patch, axis=(0, 1))
                        roi_buffers[name].append(mean_color)

        cap.release()

        # Process Audio
        try:
            # librosa can load audio directly from video file
            y, sr = librosa.load(tmp_path, sr=16000, duration=5.0)
            audio_detector.audio_buffer = y
            audio_score = audio_detector.analyze()
        except Exception as e:
            print(f"Audio load error: {e}")
            audio_score = 0.5 # Uncertain

        # Compute rPPG Score (Simplified for file upload)
        rppg_score = 0.5
        valid_signals = 0
        for name, data in roi_buffers.items():
            if len(data) > 30:
                data_np = np.array(data)
                raw = signal_extractor._pos(data_np)
                filtered = bandpass_filter.apply(raw)
                # Check signal variance/strength
                if np.std(filtered) > 0.1:
                    valid_signals += 1
        
        if valid_signals >= 2:
            rppg_score = 0.8
        elif valid_signals == 1:
            rppg_score = 0.6
            
        # Get Scores
        blink_score = blink_detector.get_score()
        motion_score = motion_validator.get_score()
        texture_score = texture_validator.get_score()
        
        # Fusion Logic
        reasons = []
        
        if faces_detected_count < 10:
            # If no face detected (or very few frames), rely primarily on texture analysis
            video_score = texture_score
            reasons.append("No consistent face detected - relying on texture analysis")
        else:
            video_score = (rppg_score * 0.4) + (texture_score * 0.3) + (motion_score * 0.2) + (blink_score * 0.1)
        
        if blink_score < 0.5 and faces_detected_count >= 10: reasons.append("Low blink rate")
        if motion_score < 0.5 and faces_detected_count >= 10: reasons.append("Unnatural motion")
        if texture_score < 0.5: reasons.append("Smooth/Blurry texture (Possible AI Video)")
        if audio_score < 0.4: reasons.append("Synthetic audio signatures detected")

        final_score = video_score
        if audio_score != 0.5:
             if audio_score < 0.4: final_score = min(final_score, audio_score)
             elif audio_score > 0.7: final_score = (final_score * 0.6) + (audio_score * 0.4)

        classification = "UNCERTAIN"
        if final_score >= 0.75: classification = "REAL HUMAN"
        elif audio_score < 0.4 and video_score > 0.6: classification = "AI VOICE DETECTED"
        elif texture_score < 0.3: classification = "AI VIDEO / DEEPFAKE"
        elif video_score < 0.4: classification = "AI VIDEO / DEEPFAKE"
        elif final_score < 0.6: classification = "POTENTIAL SPOOF"

        return {
            "classification": classification,
            "score": float(final_score),
            "video_score": float(video_score),
            "audio_score": float(audio_score),
            "texture_score": float(texture_score),
            "reasons": reasons
        }

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
