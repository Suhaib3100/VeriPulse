"""
Video Pipeline - VideoSource, FaceTracker, and VideoFrameBatch abstractions.

This module provides:
- VideoSource: Unified interface for webcam, file, and screen capture
- FaceTracker: MediaPipe-based face detection and tracking with landmarks
- VideoFrameBatch: Temporal frame accumulation for rPPG and deepfake analysis
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Generator, Optional, Tuple, List, Dict, Any
from enum import Enum
import time
from collections import deque


class VideoSourceType(Enum):
    """Types of video sources supported."""
    WEBCAM = "webcam"
    FILE = "file"
    SCREEN_CAPTURE = "screen_capture"
    FRAME_STREAM = "frame_stream"  # For browser plugin feeding frames


@dataclass
class Frame:
    """A video frame with metadata."""
    data: np.ndarray  # BGR format
    timestamp: float  # seconds since start
    frame_number: int
    source_type: VideoSourceType
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape
    
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def bgr(self) -> np.ndarray:
        """Return frame data in BGR format."""
        return self.data


@dataclass
class FaceDetection:
    """Face detection result with bounding box and landmarks."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks: Optional[np.ndarray] = None  # 468 landmarks for MediaPipe face mesh
    confidence: float = 0.0
    face_id: int = 0  # For tracking across frames
    
    # Derived regions of interest
    forehead_roi: Optional[Tuple[int, int, int, int]] = None
    left_cheek_roi: Optional[Tuple[int, int, int, int]] = None
    right_cheek_roi: Optional[Tuple[int, int, int, int]] = None
    
    # Head pose (if available)
    pitch: float = 0.0  # Up/down
    yaw: float = 0.0    # Left/right
    roll: float = 0.0   # Tilt
    
    # Eye aspect ratios for blink detection
    left_ear: float = 0.0
    right_ear: float = 0.0
    
    # Face crop from frame (set by FaceTracker)
    face_crop: Optional[np.ndarray] = None
    
    @property
    def head_pose(self) -> Optional[Dict[str, float]]:
        """Return head pose as dictionary."""
        return {"yaw": self.yaw, "pitch": self.pitch, "roll": self.roll}
    
    @property
    def ear(self) -> float:
        """Return average eye aspect ratio for blink detection."""
        return (self.left_ear + self.right_ear) / 2.0


class VideoSource:
    """
    Unified video source abstraction.
    
    Supports:
    - Live webcam (device index)
    - Video file path
    - Screen capture placeholder
    - Frame stream (for browser plugins)
    
    Example:
        >>> source = VideoSource.from_webcam(0)
        >>> for frame in source.frames():
        ...     process(frame)
        >>> source.release()
    """
    
    def __init__(
        self,
        source_type: VideoSourceType,
        source: Any,
        target_fps: float = 30.0,
        target_size: Optional[Tuple[int, int]] = None
    ):
        self.source_type = source_type
        self._source = source
        self.target_fps = target_fps
        self.target_size = target_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_buffer: deque = deque(maxlen=300)  # For frame stream mode
        self._start_time: float = 0
        self._frame_count: int = 0
        self._is_open: bool = False
        
    @classmethod
    def from_webcam(cls, device_index: int = 0, **kwargs) -> "VideoSource":
        """Create a VideoSource from a webcam device."""
        return cls(VideoSourceType.WEBCAM, device_index, **kwargs)
    
    @classmethod
    def from_file(cls, file_path: str, **kwargs) -> "VideoSource":
        """Create a VideoSource from a video file."""
        return cls(VideoSourceType.FILE, file_path, **kwargs)
    
    @classmethod
    def from_screen_capture(cls, **kwargs) -> "VideoSource":
        """Create a VideoSource for screen capture (placeholder)."""
        return cls(VideoSourceType.SCREEN_CAPTURE, None, **kwargs)
    
    @classmethod
    def from_frame_stream(cls, **kwargs) -> "VideoSource":
        """Create a VideoSource for receiving frames from external source."""
        return cls(VideoSourceType.FRAME_STREAM, None, **kwargs)
    
    def open(self) -> bool:
        """Open the video source."""
        self._start_time = time.time()
        self._frame_count = 0
        
        if self.source_type == VideoSourceType.WEBCAM:
            self._cap = cv2.VideoCapture(self._source)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self._is_open = True
                return True
                
        elif self.source_type == VideoSourceType.FILE:
            self._cap = cv2.VideoCapture(self._source)
            self._is_open = self._cap.isOpened()
            return self._is_open
            
        elif self.source_type in (VideoSourceType.SCREEN_CAPTURE, VideoSourceType.FRAME_STREAM):
            self._is_open = True
            return True
            
        return False
    
    def push_frame(self, frame_data: np.ndarray) -> None:
        """Push a frame to the buffer (for FRAME_STREAM mode)."""
        if self.source_type == VideoSourceType.FRAME_STREAM:
            timestamp = time.time() - self._start_time
            frame = Frame(
                data=frame_data,
                timestamp=timestamp,
                frame_number=self._frame_count,
                source_type=self.source_type
            )
            self._frame_buffer.append(frame)
            self._frame_count += 1
    
    def frames(self) -> Generator[Frame, None, None]:
        """
        Yield frames from the video source.
        
        Yields:
            Frame objects with BGR data and metadata.
        """
        if not self._is_open:
            if not self.open():
                raise RuntimeError(f"Failed to open video source: {self._source}")
        
        while self._is_open:
            frame_data = None
            
            if self.source_type in (VideoSourceType.WEBCAM, VideoSourceType.FILE):
                if self._cap is None:
                    break
                ret, frame_data = self._cap.read()
                if not ret:
                    break
                    
            elif self.source_type == VideoSourceType.FRAME_STREAM:
                if self._frame_buffer:
                    yield self._frame_buffer.popleft()
                    continue
                else:
                    time.sleep(0.01)  # Wait for frames
                    continue
                    
            elif self.source_type == VideoSourceType.SCREEN_CAPTURE:
                # TODO: Implement actual screen capture (e.g., pyautogui, mss)
                break
            
            if frame_data is not None:
                # Resize if target size specified
                if self.target_size is not None:
                    frame_data = cv2.resize(frame_data, self.target_size)
                
                timestamp = time.time() - self._start_time
                frame = Frame(
                    data=frame_data,
                    timestamp=timestamp,
                    frame_number=self._frame_count,
                    source_type=self.source_type
                )
                self._frame_count += 1
                yield frame
    
    def read(self) -> Optional[Frame]:
        """Read a single frame (non-generator interface)."""
        try:
            return next(self.frames())
        except StopIteration:
            return None
    
    def release(self) -> None:
        """Release the video source."""
        self._is_open = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    @property
    def fps(self) -> float:
        """Get the actual FPS of the source."""
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        return self.target_fps
    
    @property
    def frame_count(self) -> int:
        """Get total frame count (for files) or frames read so far."""
        if self._cap is not None and self.source_type == VideoSourceType.FILE:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self._frame_count
    
    def __enter__(self) -> "VideoSource":
        self.open()
        return self
    
    def __exit__(self, *args) -> None:
        self.release()


class FaceTracker:
    """
    Face detection and tracking using MediaPipe.
    
    Provides:
    - Face bounding box detection
    - 468-point face mesh landmarks
    - Head pose estimation
    - ROI extraction for rPPG
    - Blink detection via eye aspect ratio
    
    Example:
        >>> tracker = FaceTracker()
        >>> for frame in video_source.frames():
        ...     detection = tracker.process(frame)
        ...     if detection:
        ...         print(f"Face at {detection.bbox}")
    """
    
    # MediaPipe face mesh landmark indices
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    FOREHEAD_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    LEFT_CHEEK_INDICES = [116, 123, 147, 187, 207, 213, 192, 214, 212, 138, 135, 169, 170, 140]
    RIGHT_CHEEK_INDICES = [345, 352, 376, 411, 427, 433, 416, 434, 432, 367, 364, 394, 395, 369]
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
        refine_landmarks: bool = True
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self._prev_detection: Optional[FaceDetection] = None
        self._face_id_counter: int = 0
        self._tracking_threshold: float = 0.3  # IoU threshold for same face
        
    def process(self, frame: Frame) -> Optional[FaceDetection]:
        """
        Process a frame and detect/track faces.
        
        Args:
            frame: Input Frame object with BGR data, or raw numpy array (BGR).
            
        Returns:
            FaceDetection with bbox, landmarks, pose, and ROIs, or None if no face.
        """
        # Handle both Frame objects and raw numpy arrays
        if isinstance(frame, np.ndarray):
            frame_data = frame
            h, w = frame.shape[:2]
        else:
            frame_data = frame.data
            h, w = frame.height, frame.width
            
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            self._prev_detection = None
            return None
        
        # Get first face (we set max_num_faces=1)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face_landmarks.landmark
        ])
        
        # Compute bounding box from landmarks
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # Add padding
        padding = int(0.1 * (x_max - x_min))
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Extract face crop
        face_crop = frame_data[y_min:y_max, x_min:x_max].copy() if (y_max > y_min and x_max > x_min) else None
        
        # Determine face ID (tracking)
        face_id = self._get_face_id(bbox)
        
        # Compute head pose
        pitch, yaw, roll = self._estimate_head_pose(landmarks, w, h)
        
        # Compute eye aspect ratios
        left_ear = self._compute_ear(landmarks, self.LEFT_EYE_INDICES)
        right_ear = self._compute_ear(landmarks, self.RIGHT_EYE_INDICES)
        
        # Extract ROIs for rPPG
        forehead_roi = self._compute_roi(landmarks, self.FOREHEAD_INDICES, w, h)
        left_cheek_roi = self._compute_roi(landmarks, self.LEFT_CHEEK_INDICES, w, h)
        right_cheek_roi = self._compute_roi(landmarks, self.RIGHT_CHEEK_INDICES, w, h)
        
        detection = FaceDetection(
            bbox=bbox,
            landmarks=landmarks,
            confidence=1.0,  # MediaPipe doesn't provide confidence per face
            face_id=face_id,
            forehead_roi=forehead_roi,
            left_cheek_roi=left_cheek_roi,
            right_cheek_roi=right_cheek_roi,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            left_ear=left_ear,
            right_ear=right_ear,
            face_crop=face_crop
        )
        
        self._prev_detection = detection
        return detection
    
    def _get_face_id(self, bbox: Tuple[int, int, int, int]) -> int:
        """Assign face ID based on tracking."""
        if self._prev_detection is None:
            self._face_id_counter += 1
            return self._face_id_counter
        
        # Compute IoU with previous detection
        iou = self._compute_iou(bbox, self._prev_detection.bbox)
        if iou > self._tracking_threshold:
            return self._prev_detection.face_id
        
        self._face_id_counter += 1
        return self._face_id_counter
    
    @staticmethod
    def _compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _estimate_head_pose(self, landmarks: np.ndarray, w: int, h: int) -> Tuple[float, float, float]:
        """
        Estimate head pose (pitch, yaw, roll) from landmarks.
        
        Uses a simple approach with key facial points.
        """
        # Key points for pose estimation
        nose_tip = landmarks[4]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # 2D image points
        image_points = np.array([
            nose_tip[:2],
            chin[:2],
            left_eye[:2],
            right_eye[:2],
            left_mouth[:2],
            right_mouth[:2]
        ], dtype=np.float64)
        
        # 3D model points (generic face model)
        model_points = np.array([
            [0.0, 0.0, 0.0],          # Nose tip
            [0.0, -63.6, -12.5],      # Chin
            [-43.3, 32.7, -26.0],     # Left eye
            [43.3, 32.7, -26.0],      # Right eye
            [-28.9, -28.9, -24.1],    # Left mouth
            [28.9, -28.9, -24.1]      # Right mouth
        ], dtype=np.float64)
        
        # Camera matrix (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if success:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Extract Euler angles
                sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
                singular = sy < 1e-6
                
                if not singular:
                    pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                    yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                    roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                else:
                    pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                    yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                    roll = 0
                
                return (
                    float(np.degrees(pitch)),
                    float(np.degrees(yaw)),
                    float(np.degrees(roll))
                )
        except Exception:
            pass
        
        return (0.0, 0.0, 0.0)
    
    def _compute_ear(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """
        Compute Eye Aspect Ratio for blink detection.
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        pts = landmarks[eye_indices]
        
        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        
        # Horizontal distance
        h = np.linalg.norm(pts[0] - pts[3])
        
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return float(ear)
    
    def _compute_roi(
        self,
        landmarks: np.ndarray,
        indices: List[int],
        w: int,
        h: int
    ) -> Tuple[int, int, int, int]:
        """Compute bounding box ROI from landmark indices."""
        pts = landmarks[indices]
        x_min = int(max(0, pts[:, 0].min()))
        y_min = int(max(0, pts[:, 1].min()))
        x_max = int(min(w, pts[:, 0].max()))
        y_max = int(min(h, pts[:, 1].max()))
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def release(self) -> None:
        """Release resources."""
        self.face_mesh.close()


@dataclass
class VideoFrameBatch:
    """
    Accumulates frames over a time window for temporal analysis.
    
    Used for:
    - rPPG signal extraction (needs 8-10 seconds of data)
    - Temporal deepfake cues
    - Motion analysis
    
    Example:
        >>> batch = VideoFrameBatch(window_seconds=10, fps=30)
        >>> for frame in video_source.frames():
        ...     batch.add(frame, face_detection)
        ...     if batch.is_ready:
        ...         features = extract_rppg(batch)
    """
    
    window_seconds: float = 10.0
    fps: float = 30.0
    
    frames: List[Frame] = field(default_factory=list)
    detections: List[Optional[FaceDetection]] = field(default_factory=list)
    
    # Precomputed ROI signals (updated on each add)
    forehead_signal: List[np.ndarray] = field(default_factory=list)
    left_cheek_signal: List[np.ndarray] = field(default_factory=list)
    right_cheek_signal: List[np.ndarray] = field(default_factory=list)
    
    timestamps: List[float] = field(default_factory=list)
    
    @property
    def max_frames(self) -> int:
        """Maximum number of frames to store."""
        return int(self.window_seconds * self.fps)
    
    @property
    def is_ready(self) -> bool:
        """Check if we have enough frames for analysis."""
        return len(self.frames) >= self.max_frames * 0.8  # 80% of window
    
    @property
    def duration(self) -> float:
        """Current duration of accumulated frames in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
    
    def add(self, frame: Frame, detection: Optional[FaceDetection]) -> None:
        """Add a frame and its detection to the batch."""
        self.frames.append(frame)
        self.detections.append(detection)
        self.timestamps.append(frame.timestamp)
        
        # Extract ROI mean colors if face detected
        if detection is not None:
            self.forehead_signal.append(
                self._extract_roi_signal(frame.data, detection.forehead_roi)
            )
            self.left_cheek_signal.append(
                self._extract_roi_signal(frame.data, detection.left_cheek_roi)
            )
            self.right_cheek_signal.append(
                self._extract_roi_signal(frame.data, detection.right_cheek_roi)
            )
        else:
            # Append zeros if no face
            self.forehead_signal.append(np.zeros(3))
            self.left_cheek_signal.append(np.zeros(3))
            self.right_cheek_signal.append(np.zeros(3))
        
        # Maintain window size
        while len(self.frames) > self.max_frames:
            self.frames.pop(0)
            self.detections.pop(0)
            self.timestamps.pop(0)
            self.forehead_signal.pop(0)
            self.left_cheek_signal.pop(0)
            self.right_cheek_signal.pop(0)
    
    def _extract_roi_signal(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Extract mean RGB values from ROI."""
        if roi is None:
            return np.zeros(3)
        
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            return np.zeros(3)
        
        roi_patch = frame[y:y+h, x:x+w]
        if roi_patch.size == 0:
            return np.zeros(3)
        
        # Mean BGR -> RGB
        mean_bgr = np.mean(roi_patch, axis=(0, 1))
        return mean_bgr[::-1]  # Convert to RGB
    
    def get_roi_signals(self) -> Dict[str, np.ndarray]:
        """Get all ROI signals as numpy arrays."""
        return {
            "forehead": np.array(self.forehead_signal),
            "left_cheek": np.array(self.left_cheek_signal),
            "right_cheek": np.array(self.right_cheek_signal)
        }
    
    def get_valid_detections(self) -> List[FaceDetection]:
        """Get list of non-None detections."""
        return [d for d in self.detections if d is not None]
    
    def get_face_coverage(self) -> float:
        """Get fraction of frames with detected face."""
        if not self.detections:
            return 0.0
        valid = sum(1 for d in self.detections if d is not None)
        return valid / len(self.detections)
    
    def clear(self) -> None:
        """Clear all accumulated data."""
        self.frames.clear()
        self.detections.clear()
        self.timestamps.clear()
        self.forehead_signal.clear()
        self.left_cheek_signal.clear()
        self.right_cheek_signal.clear()


if __name__ == "__main__":
    # Demo usage
    print("Testing VideoSource with webcam...")
    
    tracker = FaceTracker()
    batch = VideoFrameBatch(window_seconds=5, fps=30)
    
    with VideoSource.from_webcam(0) as source:
        frame_count = 0
        for frame in source.frames():
            detection = tracker.process(frame)
            batch.add(frame, detection)
            
            # Display
            display_frame = frame.data.copy()
            if detection:
                x, y, w, h = detection.bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    display_frame,
                    f"Yaw: {detection.yaw:.1f} Pitch: {detection.pitch:.1f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
            
            cv2.putText(
                display_frame,
                f"Batch: {len(batch.frames)}/{batch.max_frames} Ready: {batch.is_ready}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.imshow("VeriPulse Video Pipeline", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count > 300:  # 10 seconds
                break
    
    cv2.destroyAllWindows()
    tracker.release()
    print(f"Processed {frame_count} frames, batch coverage: {batch.get_face_coverage():.2%}")
