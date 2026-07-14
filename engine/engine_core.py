"""
Defense COP v2.0 - Engine Core
Orchestration layer with dependency injection and deterministic execution.
"""
from typing import List, Tuple, Optional, Dict
import numpy as np

from core.config import COPConfig
from core.telemetry import TelemetryLogger
from core.time_provider import TimeProvider, FrameTimeProvider
from engine.tracking import SimpleTracker, Target
from engine.spatial import BEVMapper
from engine.anomaly_engine import AnomalyDetector, AnomalyScore
from engine.threat_engine import ThreatEngine, ThreatLevel
from core.protocols import DetectorProtocol
from dataclasses import dataclass


@dataclass
class PipelineOutput:
    """Typed output from the COP processing pipeline."""
    frame_id: int
    anomaly_scores: Dict[int, AnomalyScore]
    threat_levels: Dict[int, ThreatLevel]
    bev_canvas: Optional[np.ndarray]
    target_count: int



class COPEngine:
    """
    Core orchestration engine for Defense COP v2.0.
    Zero global state, dependency injection only.
    """
    
    def __init__(
        self,
        config: COPConfig,
        time_provider: TimeProvider,
        telemetry_logger: TelemetryLogger,
        detector_model: Optional[DetectorProtocol] = None
    ):
        """
        Initialize COP engine with injected dependencies.
        
        Args:
            config: System configuration
            time_provider: Deterministic time source
            telemetry_logger: Telemetry logger
            detector_model: Object detection model (YOLO, etc.)
        """
        self.config = config
        self._time_provider = time_provider
        self._telemetry = telemetry_logger
        self._detector = detector_model
        
        # Initialize subsystems
        self._tracker = SimpleTracker(
            iou_threshold=config.tracking.iou_threshold,
            max_age=config.tracking.max_age,
            min_hits=config.tracking.min_hits,
            use_kalman=(config.tracking.tracker_type == "sort")
        )
        
        self._bev_mapper: Optional[BEVMapper] = None
        if config.bev.enabled:
            self._bev_mapper = BEVMapper(
                calibration_points=config.bev.calibration_points,
                map_width=config.bev.map_width,
                map_height=config.bev.map_height,
                meters_per_pixel=config.bev.meters_per_pixel,
                reference_latitude=config.bev.reference_latitude,
                reference_longitude=config.bev.reference_longitude
            )
            self._telemetry.log_bev_calibration(config.bev.calibration_points)
        
        self._anomaly_detector = AnomalyDetector(
            time_provider=time_provider,
            velocity_z_threshold=config.anomaly.velocity_z_threshold,
            direction_z_threshold=config.anomaly.direction_z_threshold,
            loiter_speed_threshold=config.anomaly.loiter_speed_threshold,
            loiter_time_threshold=config.anomaly.loiter_time_threshold,
            history_window=config.anomaly.history_window
        )
        
        self._threat_engine = ThreatEngine()
        
        # State
        self._frame_count = 0
    
    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[Tuple[Tuple[float, float, float, float], str, float]]
    ) -> Tuple[List[Target], PipelineOutput]:
        """
        Process a single frame through the COP pipeline.
        
        Args:
            frame: Input frame (H x W x 3)
            detections: List of (bbox, class_name, confidence)
        
        Returns:
            (targets, {anomaly_scores, threat_levels, bev_canvas})
        """
        # Increment frame counter first for consistent frame_id
        frame_id = self._frame_count
        self._frame_count += 1
        
        # Update tracker
        targets = self._tracker.update(detections)
        
        # Anomaly detection
        anomaly_scores = self._anomaly_detector.update(targets, frame_id)
        
        # Threat assessment
        threat_levels = self._threat_engine.assess_threats(targets, anomaly_scores)
        
        # Generate telemetry for anomalies
        self._log_anomaly_events(targets, anomaly_scores, frame_id)
        
        # Stream Cursor-on-Target (CoT) over UDP for all georeferenced targets
        if self._bev_mapper and self._telemetry._cot_enabled:
            for target in targets:
                bev_point = self._bev_mapper.pixel_to_bev(target.center)
                if bev_point is not None:
                    lat, lon = self._bev_mapper.bev_to_gps(bev_point)
                    threat = threat_levels.get(target.id)
                    threat_class = threat.threat_class if threat else "LOW"
                    self._telemetry.log_cot(
                        target_id=target.id,
                        lat=lat,
                        lon=lon,
                        threat_class=threat_class,
                        callsign=f"Target #{target.id}"
                    )
        
        # Create BEV canvas
        bev_canvas = None
        if self._bev_mapper:
            bev_canvas = self._create_bev_visualization(targets, anomaly_scores)
        
        # Advance time if using frame-based provider
        if isinstance(self._time_provider, FrameTimeProvider):
            self._time_provider.next_frame()
        
        return targets, PipelineOutput(
            frame_id=frame_id,
            anomaly_scores=anomaly_scores,
            threat_levels=threat_levels,
            bev_canvas=bev_canvas,
            target_count=len(targets)
        )
    
    def _create_bev_visualization(
        self,
        targets: List[Target],
        anomaly_scores: dict
    ) -> np.ndarray:
        """Create BEV mini-map with targets and trajectories."""
        assert self._bev_mapper is not None
        canvas = self._bev_mapper.create_bev_canvas()
        
        for target in targets:
            is_anomalous = (
                target.id in anomaly_scores and
                anomaly_scores[target.id].has_anomaly()
            )
            
            # Draw trajectory
            if len(target.trajectory) > 1:
                trajectory_list = list(target.trajectory)
                self._bev_mapper.draw_trajectory_on_bev(canvas, trajectory_list)
            
            # Draw target
            self._bev_mapper.draw_target_on_bev(
                canvas,
                target.center,
                target.id,
                is_anomalous=is_anomalous
            )
        
        return canvas
    
    def _log_anomaly_events(
        self,
        targets: List[Target],
        anomaly_scores: dict,
        frame_id: int
    ) -> None:
        """Log telemetry events for detected anomalies."""
        for target in targets:
            if target.id not in anomaly_scores:
                continue
            
            score = anomaly_scores[target.id]
            
            if score.is_sprinting and self._anomaly_detector.should_alert(target.id, "sprint"):
                speed = np.sqrt(target.velocity[0]**2 + target.velocity[1]**2)
                self._telemetry.log_anomaly_sprint(
                    target.id,
                    speed,
                    score.velocity_z_score,
                    frame_id
                )
            
            if score.is_erratic and self._anomaly_detector.should_alert(target.id, "erratic"):
                direction_history = self._anomaly_detector._variance_history.get(target.id)
                direction_variance = float(list(direction_history)[-1]) if direction_history and len(direction_history) > 0 else 0.0
                self._telemetry.log_anomaly_erratic(
                    target.id,
                    direction_variance,
                    score.direction_z_score,
                    frame_id
                )
            
            if score.is_loitering and self._anomaly_detector.should_alert(target.id, "loiter"):
                loiter_duration = self._anomaly_detector.get_loiter_duration(target.id)
                if loiter_duration:
                    speed = np.sqrt(target.velocity[0]**2 + target.velocity[1]**2)
                    self._telemetry.log_anomaly_loiter(
                        target.id,
                        loiter_duration,
                        speed,
                        frame_id
                    )
    
    def shutdown(self) -> None:
        """Clean shutdown of engine."""
        self._telemetry.close()
    
    def get_frame_count(self) -> int:
        """Get current frame count."""
        return self._frame_count
