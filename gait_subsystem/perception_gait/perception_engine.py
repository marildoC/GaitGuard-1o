"""
Main Perception Engine.
Orchestrates Object Detection, Pose Estimation, Tracking, and Feature Extraction.
"""

from __future__ import annotations
import logging
import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple
from collections import deque
from ultralytics import YOLO

from schemas import Frame, Tracklet
from core.interfaces import PerceptionEngine
from perception.detector import Detection
from .tracker_ocsort import OCSortTracker, Track
from .appearance import AppearanceExtractor
from .ring_buffer import RingBuffer, RingBufferConfig
from gait.config import GaitConfig, default_gait_config

logger = logging.getLogger(__name__)

class TrackState:
    """Maintains transient state for an active track, including skeleton history and EMA smoothing."""
    def __init__(self, track_id: int, camera_id: str):
        self.tracklet = Tracklet(
            track_id=track_id, camera_id=camera_id,
            last_frame_id=0, last_box=(0, 0, 0, 0),
            confidence=0.0, age_frames=0, lost_frames=0, history_boxes=[]
        )
        self.appearance_feature: Optional[np.ndarray] = None
        self.kp_ema: Optional[np.ndarray] = None
        self.kp_history: deque[np.ndarray] = deque(maxlen=30) 


def extract_yolo_boxes_keypoints(yolo_result) -> Tuple[List[List[float]], List[float], List[np.ndarray]]:
    """
    Parses YOLOv8-pose output into standard formats.
    Returns:
        - Bounding boxes (xyxy)
        - Confidence scores
        - Normalized keypoints (K, 3) where columns are [x, y, confidence]
    """
    boxes, scores, keypoints_list = [], [], []

    try:
        if hasattr(yolo_result, "boxes") and yolo_result.boxes is not None:
            xyxy = yolo_result.boxes.xyxy.cpu().numpy()
            confs = yolo_result.boxes.conf.cpu().numpy()
            for i in range(xyxy.shape[0]):
                boxes.append([float(xyxy[i,0]), float(xyxy[i,1]), float(xyxy[i,2]), float(xyxy[i,3])])
                scores.append(float(confs[i]))
        
        if hasattr(yolo_result, "keypoints") and yolo_result.keypoints is not None:
            if hasattr(yolo_result.keypoints, 'xyn'):
                kp_data = yolo_result.keypoints.xyn.cpu().numpy() 
                confs = yolo_result.keypoints.conf.cpu().numpy() if yolo_result.keypoints.conf is not None else np.ones((kp_data.shape[0], kp_data.shape[1]))
                for i in range(kp_data.shape[0]):
                    keypoints_list.append(np.hstack([kp_data[i], confs[i][:, None]]))
    except Exception as e:
        logger.error(f"YOLO Extraction error: {e}")

    return boxes, scores, keypoints_list


class Phase1PerceptionEngine(PerceptionEngine):
    """
    Primary engine for real-time person tracking and pose stream generation.
    Handles multi-stage inference: YOLO (Pose) -> Appearance (ReID) -> OC-SORT (Tracking).
    """
    def __init__(self, tracker=None, appearance=None, ring_buffer=None, max_lost_frames=30, 
                 keypoint_ema_alpha=0.65, keypoint_history_length=30, gait_config=None):
        super().__init__()
        self.gait_config = gait_config or default_gait_config()
        self.yolo_pose_model = YOLO(self.gait_config.models.pose_model_name)
        
        is_pytorch = self.gait_config.models.pose_model_name.endswith(".pt")
        if is_pytorch:
            try: self.yolo_pose_model.fuse()
            except: pass
            self.yolo_pose_model.to("cuda" if (self.gait_config.device.use_half and self.gait_config.device.device == "cuda") else "cpu")
        
        self.tracker = tracker or OCSortTracker()
        self.appearance = appearance or AppearanceExtractor()
        self.ring_buffer = ring_buffer or RingBuffer(RingBufferConfig())
        self._states: Dict[int, TrackState] = {}
        self.max_lost_frames = max_lost_frames
        self.keypoint_ema_alpha = keypoint_ema_alpha
        self.keypoint_history_length = keypoint_history_length

    def process_frame(self, frame: Frame) -> List[Tracklet]:
        """
        Executes the full perception loop for a single frame.
        1. YOLO-Pose detection
        2. Appearance embedding extraction
        3. OC-SORT track association
        4. Keypoint EMA smoothing and sequence update
        5. Ring Buffer persistence
        """
        if frame.image is None: return []

        yolo_results = self.yolo_pose_model.predict(
            source=frame.image, device=self.gait_config.device.device,
            imgsz=self.gait_config.route.img_size, conf=self.gait_config.thresholds.min_visibility,
            iou=0.45, classes=[0], half=self.gait_config.device.use_half, verbose=False, task="pose"
        )
        
        boxes, scores, keypoints_data = extract_yolo_boxes_keypoints(yolo_results[0])
        detections = [Detection(x1=b[0], y1=b[1], x2=b[2], y2=b[3], score=s, class_id=0, class_name="person")
                      for b, s in zip(boxes, scores)]

        features = self.appearance.compute_features_for_detections(frame, detections)
        tracks: List[Track] = self.tracker.update(detections, features)

        active_ids = set()
        current_tracklets = [] 

        for tr in tracks:
            tid = tr.track_id
            active_ids.add(tid)
            if tid not in self._states:
                self._states[tid] = TrackState(tid, frame.camera_id)
            
            state = self._states[tid]
            t = state.tracklet
            t.last_frame_id, t.last_box, t.confidence = frame.frame_id, tuple(tr.bbox.tolist()), tr.score
            t.age_frames, t.lost_frames = t.age_frames + 1, 0
            
            best_iou = 0.0
            associated_kp = None
            for idx, y_box in enumerate(boxes):
                iou_val = self._calculate_iou(t.last_box, y_box)
                if iou_val > best_iou:
                    best_iou, associated_kp = iou_val, keypoints_data[idx]
            
            if associated_kp is not None:
                if state.kp_ema is None: state.kp_ema = associated_kp.copy()
                else: state.kp_ema = self.keypoint_ema_alpha * associated_kp + (1.0 - self.keypoint_ema_alpha) * state.kp_ema
                state.kp_history.append(state.kp_ema.copy())
            
            t.gait_sequence_data = list(state.kp_history)
            current_tracklets.append(t)

        self._increment_lost_and_prune(active_ids)
        for t in current_tracklets:
            self.ring_buffer.add(t.track_id, frame.ts, frame.frame_id, t.last_box, 
                                 pose=t.gait_sequence_data[-1] if t.gait_sequence_data else None)
        
        return current_tracklets

    def _calculate_iou(self, boxA, boxB) -> float:
        """Computes Intersection over Union between two bounding boxes."""
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = float(boxA_area + boxB_area - inter_area)
        return inter_area / union_area if union_area > 0 else 0.0

    def _increment_lost_and_prune(self, active_ids: set) -> None:
        """Updates lost counters and removes tracks that have been inactive too long."""
        to_remove = []
        for tid, state in self._states.items():
            if tid not in active_ids:
                state.tracklet.lost_frames += 1
                if state.tracklet.lost_frames > self.max_lost_frames:
                    to_remove.append(tid)
        for tid in to_remove:
            self._states.pop(tid, None)
            self.ring_buffer.remove_track(tid)