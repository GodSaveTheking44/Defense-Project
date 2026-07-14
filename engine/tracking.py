"""
Defense COP v2.0 - Multi-Target Tracking
Production-grade object tracking with persistent IDs.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment  # type: ignore


class KalmanFilter:
    """
    A simple 2D linear Kalman Filter for tracking bounding boxes.
    State vector x: [x_c, y_c, w, h, vx, vy, vw, vh]^T
    Measurement vector z: [x_c, y_c, w, h]^T
    """
    def __init__(self):
        # 8 state variables, 4 measurement variables
        self.x = np.zeros((8, 1))
        
        # State transition matrix F
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # constant velocity model: x_t = x_{t-1} + v_{t-1}
            
        # Measurement matrix H
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0
            
        # State covariance P
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0  # High initial uncertainty for velocities
        
        # Process noise covariance Q
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 0.01
        
        # Measurement noise covariance R
        self.R = np.eye(4) * 1.0
        self.R[2, 2] *= 10.0  # Scale (width) uncertainty is higher
        self.R[3, 3] *= 10.0  # Aspect ratio (height) uncertainty is higher

    def predict(self) -> None:
        """Predict step."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z: np.ndarray) -> None:
        """Update step with new measurement z."""
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        identity_matrix = np.eye(8)
        self.P = np.dot(identity_matrix - np.dot(K, self.H), self.P)


@dataclass
class Target:
    """Tracked target with persistent ID."""
    id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    center: Tuple[float, float] = field(init=False)
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    trajectory: deque = field(default_factory=lambda: deque(maxlen=50))
    use_kalman: bool = False
    
    def __post_init__(self):
        """Calculate center point."""
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.trajectory.append(self.center)
        
        self.kf = None
        if self.use_kalman:
            self.kf = KalmanFilter()
            w = x2 - x1
            h = y2 - y1
            self.kf.x[:4] = np.array([[self.center[0]], [self.center[1]], [w], [h]])
    
    def update(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: float
    ) -> None:
        """Update target with new detection."""
        self.confidence = confidence
        
        if self.use_kalman and self.kf is not None:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            x_c = x1 + w / 2.0
            y_c = y1 + h / 2.0
            z = np.array([[x_c], [y_c], [w], [h]])
            self.kf.update(z)
            
            # Extract updated bbox and center from Kalman state
            x_c_u, y_c_u, w_u, h_u = self.kf.x[:4, 0]
            w_u = max(1.0, w_u)
            h_u = max(1.0, h_u)
            self.bbox = (x_c_u - w_u / 2.0, y_c_u - h_u / 2.0, x_c_u + w_u / 2.0, y_c_u + h_u / 2.0)
            self.center = (x_c_u, y_c_u)
            self.velocity = (float(self.kf.x[4, 0]), float(self.kf.x[5, 0]))
        else:
            old_center = self.center
            self.bbox = bbox
            x1, y1, x2, y2 = bbox
            self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Calculate velocity (pixels per frame)
            self.velocity = (
                self.center[0] - old_center[0],
                self.center[1] - old_center[1]
            )
            
        self.trajectory.append(self.center)
        self.hits += 1
        self.time_since_update = 0
    
    def predict(self) -> None:
        """Predict next position."""
        self.age += 1
        self.time_since_update += 1
        
        if self.use_kalman and self.kf is not None:
            self.kf.predict()
            # Extract predicted bbox and center from Kalman state
            x_c, y_c, w, h = self.kf.x[:4, 0]
            w = max(1.0, w)
            h = max(1.0, h)
            self.bbox = (x_c - w / 2.0, y_c - h / 2.0, x_c + w / 2.0, y_c + h / 2.0)
            self.center = (x_c, y_c)
            self.velocity = (float(self.kf.x[4, 0]), float(self.kf.x[5, 0]))
        else:
            # Apply constant velocity offset to bounding box
            vx, vy = self.velocity
            x1, y1, x2, y2 = self.bbox
            self.bbox = (x1 + vx, y1 + vy, x2 + vx, y2 + vy)
            
            # Recalculate center
            self.center = ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


class SimpleTracker:
    """
    Simple multi-target tracker using IoU matching.
    Production implementation would use ByteTrack or SORT.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        use_kalman: bool = False
    ):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU for matching
            max_age: Maximum frames to keep target without detection
            min_hits: Minimum hits before target is confirmed
            use_kalman: Enable Kalman filter state estimation (SORT)
        """
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._min_hits = min_hits
        self._next_id = 1
        self._targets: Dict[int, Target] = {}
        self._use_kalman = use_kalman
    
    def update(
        self,
        detections: List[Tuple[Tuple[float, float, float, float], str, float]]
    ) -> List[Target]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (bbox, class_name, confidence)
        
        Returns:
            List of active confirmed targets
        """
        # Predict existing targets
        for target in self._targets.values():
            target.predict()
        
        # Match detections to existing targets
        matched_tracks, unmatched_detections = self._match_detections(detections)
        
        # Update matched targets
        for track_id, detection_idx in matched_tracks:
            bbox, class_name, confidence = detections[detection_idx]
            self._targets[track_id].update(bbox, confidence)
        
        # Create new targets for unmatched detections
        for det_idx in unmatched_detections:
            bbox, class_name, confidence = detections[det_idx]
            new_target = Target(
                id=self._next_id,
                bbox=bbox,
                class_name=class_name,
                confidence=confidence,
                use_kalman=self._use_kalman
            )
            self._targets[self._next_id] = new_target
            self._next_id += 1
        
        # Remove old targets
        targets_to_remove = [
            tid for tid, target in self._targets.items()
            if target.time_since_update > self._max_age
        ]
        for tid in targets_to_remove:
            del self._targets[tid]
        
        # Return confirmed targets only
        return [
            target for target in self._targets.values()
            if target.hits >= self._min_hits
        ]
    
    def _match_detections(
        self,
        detections: List[Tuple[Tuple[float, float, float, float], str, float]]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Match detections to existing tracks using IoU.
        
        Returns:
            (matched_pairs, unmatched_detection_indices)
        """
        if len(self._targets) == 0:
            return [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], []
        
        # Compute IoU matrix
        track_ids = list(self._targets.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for t_idx, track_id in enumerate(track_ids):
            track_bbox = self._targets[track_id].bbox
            for d_idx, (det_bbox, _, _) in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._compute_iou(track_bbox, det_bbox)
        
        # Hungarian matching using Scipy
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_tracks = []
        matched_det_indices = set()
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self._iou_threshold:
                matched_tracks.append((track_ids[r], c))
                matched_det_indices.add(c)
        
        unmatched_detections = [
            i for i in range(len(detections))
            if i not in matched_det_indices
        ]
        
        return matched_tracks, unmatched_detections
    
    @staticmethod
    def _compute_iou(
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Compute Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_all_targets(self) -> List[Target]:
        """Get all active targets (including unconfirmed)."""
        return list(self._targets.values())
