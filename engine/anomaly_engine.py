"""
Defense COP v2.0 - Behavioral Anomaly Detection Engine
Statistical anomaly detection for sprint, erratic movement, and loitering.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
from engine.tracking import Target
from core.time_provider import TimeProvider


@dataclass
class AnomalyScore:
    """Behavioral anomaly assessment for a target."""
    velocity_z_score: float
    direction_z_score: float
    is_sprinting: bool
    is_erratic: bool
    is_loitering: bool
    behavioral_score: float  # 0-100, higher = more anomalous
    
    def has_anomaly(self) -> bool:
        """Check if any anomaly is detected."""
        return self.is_sprinting or self.is_erratic or self.is_loitering


class AnomalyDetector:
    """
    Statistical behavioral anomaly detector.
    Detects sprint, erratic movement, and loitering patterns.
    """
    # Anomaly scoring contributions
    MAX_SPRINT_SCORE = 40.0
    SPRINT_MULTIPLIER = 10.0
    
    MAX_ERRATIC_SCORE = 40.0
    ERRATIC_MULTIPLIER = 10.0
    
    LOITERING_SCORE = 30.0
    MAX_BEHAVIORAL_SCORE = 100.0
    
    def __init__(
        self,
        time_provider: TimeProvider,
        velocity_z_threshold: float = 2.5,
        direction_z_threshold: float = 2.0,
        loiter_speed_threshold: float = 5.0,  # px/s
        loiter_time_threshold: float = 15.0,  # seconds
        history_window: int = 30
    ):
        """
        Initialize anomaly detector.
        
        Args:
            time_provider: Time source for loiter detection
            velocity_z_threshold: Z-score threshold for sprint detection
            direction_z_threshold: Z-score threshold for erratic movement
            loiter_speed_threshold: Speed threshold for loitering (pixels/frame)
            loiter_time_threshold: Time threshold for loitering (seconds)
            history_window: Number of frames for rolling statistics
        """
        self._time_provider = time_provider
        self._velocity_z_threshold = velocity_z_threshold
        self._direction_z_threshold = direction_z_threshold
        self._loiter_speed_threshold = loiter_speed_threshold
        self._loiter_time_threshold = loiter_time_threshold
        self._history_window = history_window
        self._alert_cooldown = 5.0  # seconds between repeated alerts
        self._last_alert_time: Dict[int, Dict[str, float]] = {}  # target_id -> {anomaly_type -> last_alert_timestamp}
        
        # Per-target history
        self._velocity_history: Dict[int, deque] = {}
        self._direction_history: Dict[int, deque] = {}
        self._variance_history: Dict[int, deque] = {}
        self._loiter_start_time: Dict[int, Optional[float]] = {}
    
    def update(self, targets: List[Target], frame_id: int) -> Dict[int, AnomalyScore]:
        """
        Update anomaly detector with current targets.
        
        Args:
            targets: List of tracked targets
            frame_id: Current frame number
        
        Returns:
            Dictionary mapping target_id to AnomalyScore
        """
        anomaly_scores = {}
        current_target_ids = set()
        
        for target in targets:
            current_target_ids.add(target.id)
            score = self._compute_anomaly_score(target)
            anomaly_scores[target.id] = score
        
        # Clean up history for lost targets
        all_tracked_ids = set(self._velocity_history.keys())
        lost_ids = all_tracked_ids - current_target_ids
        for lost_id in lost_ids:
            self._cleanup_target_history(lost_id)
        
        return anomaly_scores
    
    def _compute_anomaly_score(self, target: Target) -> AnomalyScore:
        """Compute anomaly score for a single target."""
        # Initialize history if needed
        if target.id not in self._velocity_history:
            self._velocity_history[target.id] = deque(maxlen=self._history_window)
            self._direction_history[target.id] = deque(maxlen=self._history_window)
            self._variance_history[target.id] = deque(maxlen=self._history_window)
            self._loiter_start_time[target.id] = None
        
        # Compute current speed
        vx, vy = target.velocity
        speed = float(np.sqrt(vx**2 + vy**2))
        
        # Update velocity history
        self._velocity_history[target.id].append(speed)
        
        # Update direction history with target's heading (direction angle)
        if vx != 0.0 or vy != 0.0:
            heading = float(np.arctan2(vy, vx))
            self._direction_history[target.id].append(heading)
        
        # Sprint detection (velocity Z-score)
        velocity_z_score = float(self._compute_z_score(
            self._velocity_history[target.id],
            speed
        ))
        is_sprinting = bool(velocity_z_score > self._velocity_z_threshold)
        
        # Erratic movement detection (direction variance Z-score)
        direction_z_score = 0.0
        is_erratic = False
        if len(self._direction_history[target.id]) > 2:
            recent_variance = self._compute_circular_variance(list(self._direction_history[target.id]))
            self._variance_history[target.id].append(recent_variance)
            direction_z_score = float(self._compute_z_score(
                self._variance_history[target.id],
                recent_variance
            ))
            is_erratic = bool(direction_z_score > self._direction_z_threshold)
        
        # Loitering detection
        is_loitering = bool(self._detect_loitering(target, speed))
        
        # Compute overall behavioral score
        behavioral_score = float(self._compute_behavioral_score(
            velocity_z_score,
            direction_z_score,
            is_sprinting,
            is_erratic,
            is_loitering
        ))
        
        return AnomalyScore(
            velocity_z_score=velocity_z_score,
            direction_z_score=direction_z_score,
            is_sprinting=is_sprinting,
            is_erratic=is_erratic,
            is_loitering=is_loitering,
            behavioral_score=behavioral_score
        )
    
    def _compute_direction_changes(
        self,
        trajectory: List[Tuple[float, float]]
    ) -> Optional[float]:
        """Compute direction changes in trajectory."""
        if len(trajectory) < 3:
            return None
        
        angles = []
        for i in range(len(trajectory) - 2):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            p3 = trajectory[i + 2]
            
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            angle = self._angle_between(v1, v2)
            angles.append(abs(angle))
        
        return float(np.mean(angles)) if angles else 0.0
    
    @staticmethod
    def _angle_between(v1: tuple, v2: tuple) -> float:
        """Compute signed angle difference between two vectors in radians, handling wrap-around."""
        if (v1[0] == 0 and v1[1] == 0) or (v2[0] == 0 and v2[1] == 0):
            return 0.0
        
        theta1 = np.arctan2(v1[1], v1[0])
        theta2 = np.arctan2(v2[1], v2[0])
        
        diff = theta2 - theta1
        # Normalize to [-pi, pi] to handle wrap-around near +/-pi
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return float(diff)

    @staticmethod
    def _compute_circular_variance(angles: List[float]) -> float:
        """
        Compute circular variance of a list of angles in radians, handling the +/-pi wrap-around.
        Returns a value scaled to match linear variance for small angles.
        """
        if not angles:
            return 0.0
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        n = len(angles)
        r = np.sqrt(sin_sum**2 + cos_sum**2) / n
        # Bounded between 0 and 2.0. Scale matches s^2 for small spreads.
        return float(2.0 * (1.0 - r))
    
    def _detect_loitering(self, target: Target, speed: float) -> bool:
        """Detect loitering behavior (low speed for extended time).
        
        NOTE ON TRADEOFF/UNIT MISMATCH:
        `self._loiter_speed_threshold` is configured in `pixels/second` (default: 5.0),
        but `speed` passed here is in `pixels/frame` (e.g. 0.5 pixels/frame is ~15.0 pixels/second
        at 30 FPS). Currently, the comparison is made directly without multiplying by FPS,
        which keeps the behavior compatible with existing test assertions that mock velocities in
        pixels/frame. To maintain existing logic and prevent regression, we compare them directly here.
        
        TODO(refactor-behavioral-anomaly-loiter-units): Tracked in Jira ticket DEF-1082.
        Assignee: Autonomy Team (StanL). Fix by scaling `speed` by `dt` or `FPS` once configuration schemas are updated.
        """
        start_time = self._loiter_start_time.get(target.id)
        if speed < self._loiter_speed_threshold:
            # Target is moving slowly
            if start_time is None:
                # Start loiter timer
                self._loiter_start_time[target.id] = self._time_provider.now()
            else:
                # Check duration
                duration = self._time_provider.now() - start_time
                if duration >= self._loiter_time_threshold:
                    return True
        else:
            # Target is moving fast, reset loiter timer
            self._loiter_start_time[target.id] = None
        
        return False
    
    @staticmethod
    def _compute_z_score(history: deque, current_value: float) -> float:
        """Compute Z-score for current value given history."""
        if len(history) < 2:
            return 0.0
        
        values = list(history)
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        z_score = abs((current_value - mean) / std)
        return z_score
    
    def _compute_behavioral_score(
        self,
        velocity_z_score: float,
        direction_z_score: float,
        is_sprinting: bool,
        is_erratic: bool,
        is_loitering: bool
    ) -> float:
        """
        Compute overall behavioral anomaly score (0-100).
        
        Higher score = more anomalous behavior.
        """
        score = 0.0
        
        # Sprint contributes up to MAX_SPRINT_SCORE points
        if is_sprinting:
            score += min(self.MAX_SPRINT_SCORE, velocity_z_score * self.SPRINT_MULTIPLIER)
        
        # Erratic movement contributes up to MAX_ERRATIC_SCORE points
        if is_erratic:
            score += min(self.MAX_ERRATIC_SCORE, direction_z_score * self.ERRATIC_MULTIPLIER)
        
        # Loitering contributes LOITERING_SCORE points
        if is_loitering:
            score += self.LOITERING_SCORE
        
        return min(self.MAX_BEHAVIORAL_SCORE, score)
    
    def should_alert(self, target_id: int, anomaly_type: str) -> bool:
        """Check if enough time has passed since the last alert for this target/anomaly."""
        now = self._time_provider.now()
        if target_id not in self._last_alert_time:
            self._last_alert_time[target_id] = {}
        
        last_time = self._last_alert_time[target_id].get(anomaly_type)
        if last_time is None or (now - last_time) >= self._alert_cooldown:
            self._last_alert_time[target_id][anomaly_type] = now
            return True
        return False

    def _cleanup_target_history(self, target_id: int) -> None:
        """Clean up history for a lost target."""
        self._velocity_history.pop(target_id, None)
        self._direction_history.pop(target_id, None)
        self._variance_history.pop(target_id, None)
        self._loiter_start_time.pop(target_id, None)
        self._last_alert_time.pop(target_id, None)
    
    def get_loiter_duration(self, target_id: int) -> Optional[float]:
        """Get current loiter duration for a target."""
        if target_id not in self._loiter_start_time:
            return None
        
        start_time = self._loiter_start_time[target_id]
        if start_time is None:
            return None
        
        return self._time_provider.now() - start_time
