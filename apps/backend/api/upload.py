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
        has_video_track = False
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            has_video_track = True    
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
        audio_score = 0.5 # Default to uncertain
        has_audio_track = False
        try:
            # librosa can load audio directly from video file
            # Use a larger duration for better analysis
            y, sr = librosa.load(tmp_path, sr=16000, duration=10.0)
            if len(y) > 0:
                has_audio_track = True
                # Check for silence
                if np.max(np.abs(y)) < 0.01:
                    audio_score = 0.5 # Silent
                else:
                    audio_detector.audio_buffer = y
                    audio_score = audio_detector.analyze()
        except Exception as e:
            print(f"Audio load error: {e}")
            # If librosa fails, it might be a video without audio or format issue
            pass

        # Compute rPPG Score (Advanced)
        rppg_score = 0.0
        feature_extractor = FeatureExtractor(fs=30)
        
        valid_rois = 0
        total_snr = 0
        
        for name, data in roi_buffers.items():
            if len(data) > 30:
                data_np = np.array(data)
                # Extract signal using POS method
                raw = signal_extractor._pos(data_np)
                filtered = bandpass_filter.apply(raw)
                
                # Extract features (SNR, Periodicity)
                feats = feature_extractor.extract(filtered)
                
                # Score based on SNR and Periodicity
                # Real pulse usually has SNR > 2.0 and Periodicity > 0.6
                roi_score = 0.0
                if feats['snr'] > 1.5: roi_score += 0.4
                if feats['snr'] > 3.0: roi_score += 0.2
                if feats['periodicity'] > 0.5: roi_score += 0.4
                
                if roi_score > 0.5:
                    valid_rois += 1
                    total_snr += feats['snr']
        
        if valid_rois > 0:
            # High confidence if multiple ROIs show good signal
            rppg_score = min(1.0, (valid_rois / 3.0) * 0.8 + (total_snr / (valid_rois * 5.0)) * 0.2)
            # Boost score if we have strong signals
            if valid_rois >= 2 and total_snr/valid_rois > 2.5:
                rppg_score = max(rppg_score, 0.9)
        else:
            rppg_score = 0.2 # Low confidence/No pulse detected

        # Get Scores
        blink_score = blink_detector.get_score()
        motion_score = motion_validator.get_score()
        texture_score = texture_validator.get_score()
        
        # Fusion Logic
        reasons = []
        video_score = 0.5
        
        if not has_video_track:
            # Audio Only File
            final_score = audio_score
            if audio_score < 0.4: 
                classification = "AI AUDIO GENERATED"
                reasons.append("Synthetic audio signatures detected")
            elif audio_score > 0.7:
                classification = "REAL HUMAN AUDIO"
            else:
                classification = "UNCERTAIN"
                
            return {
                "classification": classification,
                "score": float(final_score),
                "video_score": 0.0,
                "audio_score": float(audio_score),
                "texture_score": 0.0,
                "reasons": reasons,
                "components": {
                    "rppg": 0.0,
                    "blink": 0.0,
                    "motion": 0.0,
                    "texture": 0.0,
                    "audio": float(audio_score)
                }
            }

        # Video Processing Logic
        if faces_detected_count < 10:
            # If no face detected (or very few frames), rely primarily on texture analysis
            video_score = texture_score
            reasons.append("No consistent face detected - relying on texture analysis")
            
            # If texture is high but no face, it's likely a real video of something else, or a face that wasn't detected.
            # We should NOT classify as "REAL HUMAN" if no face is found.
            if texture_score > 0.7:
                classification = "REAL VIDEO (NO FACE)"
                final_score = texture_score
            elif texture_score < 0.3:
                classification = "AI VIDEO GENERATED"
                final_score = texture_score
            else:
                classification = "UNCERTAIN"
                final_score = 0.5
                
            return {
                "classification": classification,
                "score": float(final_score),
                "video_score": float(video_score),
                "audio_score": float(audio_score),
                "texture_score": float(texture_score),
                "reasons": reasons,
                "components": {
                    "rppg": 0.0,
                    "blink": 0.0,
                    "motion": 0.0,
                    "texture": float(texture_score),
                    "audio": float(audio_score)
                }
            }
            
        else:
            # Weighted Fusion
            # rPPG is the strongest physiological indicator
            # Texture is the strongest artifact indicator
            video_score = (rppg_score * 0.35) + (texture_score * 0.35) + (motion_score * 0.15) + (blink_score * 0.15)
        
        # Generate Explanations
        if blink_score < 0.4 and faces_detected_count >= 30: reasons.append(f"Abnormal blink rate (Score: {blink_score:.2f})")
        if motion_score < 0.4 and faces_detected_count >= 30: reasons.append(f"Unnatural head motion (Score: {motion_score:.2f})")
        if texture_score < 0.4: reasons.append(f"Digital artifacts/Smoothness detected (Score: {texture_score:.2f})")
        if rppg_score < 0.4 and faces_detected_count >= 30: reasons.append(f"No natural pulse detected (Score: {rppg_score:.2f})")
        if has_audio_track and audio_score < 0.4: reasons.append(f"Synthetic audio signatures (Score: {audio_score:.2f})")

        # Final Verdict Calculation
        final_score = video_score
        if has_audio_track and audio_score != 0.5:
             # If audio is clearly fake, drag down the score heavily
             if audio_score < 0.4: final_score = min(final_score, audio_score)
             # If audio is clearly real, it can boost a bit, but video artifacts are more important
             elif audio_score > 0.8: final_score = (final_score * 0.7) + (audio_score * 0.3)

        classification = "UNCERTAIN"
        
        # Strict AI Detection Logic
        if texture_score < 0.2:
            classification = "AI VIDEO GENERATED"
            final_score = min(final_score, 0.2)
        elif has_audio_track and audio_score < 0.3:
            classification = "AI AUDIO GENERATED"
            final_score = min(final_score, 0.3)
        elif rppg_score < 0.3 and faces_detected_count > 50:
             # If we have a good face view but no pulse, it's likely a deepfake
            classification = "DEEPFAKE DETECTED"
            final_score = min(final_score, 0.35)
        elif video_score < 0.4:
            classification = "DEEPFAKE DETECTED"
        elif final_score >= 0.75:
            classification = "REAL HUMAN"
        elif final_score < 0.6:
            classification = "POTENTIAL SPOOF"
        else:
            classification = "UNCERTAIN"

        return {
            "classification": classification,
            "score": float(final_score),
            "video_score": float(video_score),
            "audio_score": float(audio_score),
            "texture_score": float(texture_score),
            "reasons": reasons,
            "components": {
                "rppg": float(rppg_score),
                "blink": float(blink_score),
                "motion": float(motion_score),
                "texture": float(texture_score),
                "audio": float(audio_score)
            }
        }

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
