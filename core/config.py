"""
Defense COP v2.0 - Configuration System
Production-grade configuration with typed dataclasses.
"""
from dataclasses import dataclass, field
import logging
from typing import List, Tuple
import yaml  # type: ignore
from pathlib import Path

logger = logging.getLogger("DefenseCOP.Config")


@dataclass
class BEVConfig:
    """Bird's Eye View configuration."""
    enabled: bool = True
    calibration_points: List[Tuple[float, float, float, float]] = field(default_factory=lambda: [
        (100, 400, 0, 0),
        (540, 400, 10, 0),
        (200, 150, 0, 10),
        (440, 150, 10, 10)
    ])
    map_width: int = 400
    map_height: int = 400
    meters_per_pixel: float = 0.025
    reference_latitude: float = 34.0522
    reference_longitude: float = -118.2437


@dataclass
class AnomalyConfig:
    """Anomaly detection configuration."""
    velocity_z_threshold: float = 2.5
    direction_z_threshold: float = 2.0
    loiter_speed_threshold: float = 5.0  # pixels/second
    loiter_time_threshold: float = 15.0  # seconds
    history_window: int = 30  # frames


@dataclass
class TrackingConfig:
    """Object tracking configuration."""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_age: int = 30
    min_hits: int = 3
    tracker_type: str = "sort"


@dataclass
class VideoConfig:
    """Video input configuration."""
    source: str = "0"  # 0 for webcam, or path to video file
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class UIConfig:
    """UI rendering configuration."""
    show_bev: bool = True
    show_trajectories: bool = True
    show_anomaly_score: bool = True
    show_fps: bool = True
    bev_position: Tuple[int, int] = (10, 10)  # top-left corner


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""
    enabled: bool = True
    output_dir: str = "telemetry"
    format: str = "jsonl"
    log_all_frames: bool = False  # Only log events if False
    cot_enabled: bool = False
    cot_host: str = "239.2.3.1"
    cot_port: int = 24910


@dataclass
class DetectorConfig:
    """Detector configuration."""
    enabled: bool = True
    model: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    acceleration: str = "none"  # "none", "onnx", "tensorrt"
    device: str = "cpu"        # "cpu", "cuda", "0", etc.


@dataclass
class COPConfig:
    """Master configuration for Defense COP v2.0."""
    bev: BEVConfig = field(default_factory=BEVConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    headless: bool = False
    deterministic_seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> "COPConfig":
        """Load configuration from YAML file.
        
        Handles empty YAML files gracefully and warns on unknown keys.
        """
        logger = logging.getLogger("DefenseCOP.Config")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        # Warn about unknown top-level keys
        known_keys = {"bev", "anomaly", "tracking", "video", "ui", "telemetry", "detector", "headless", "deterministic_seed"}
        unknown_keys = set(data.keys()) - known_keys
        for key in unknown_keys:
            logger.warning(f"Unknown config key '{key}' will be ignored")
        
        # Build config with safe unpacking (filter unknown sub-keys)
        def safe_construct(target_cls, data_dict):
            """Construct dataclass filtering out unknown fields."""
            import dataclasses
            field_names = {f.name for f in dataclasses.fields(target_cls)}
            filtered = {k: v for k, v in data_dict.items() if k in field_names}
            unknown = set(data_dict.keys()) - field_names
            for k in unknown:
                logger.warning(f"Unknown config field '{k}' will be ignored")
            
            # Convert lists to tuples where type hints expect tuples
            if target_cls is BEVConfig and "calibration_points" in filtered:
                pts = filtered["calibration_points"]
                if isinstance(pts, list):
                    filtered["calibration_points"] = [tuple(p) if isinstance(p, list) else p for p in pts]
            elif target_cls is UIConfig and "bev_position" in filtered:
                pos = filtered["bev_position"]
                if isinstance(pos, list):
                    filtered["bev_position"] = tuple(pos)
            
            return target_cls(**filtered)
        
        return cls(
            bev=safe_construct(BEVConfig, data.get("bev", {})),
            anomaly=safe_construct(AnomalyConfig, data.get("anomaly", {})),
            tracking=safe_construct(TrackingConfig, data.get("tracking", {})),
            video=safe_construct(VideoConfig, data.get("video", {})),
            ui=safe_construct(UIConfig, data.get("ui", {})),
            telemetry=safe_construct(TelemetryConfig, data.get("telemetry", {})),
            detector=safe_construct(DetectorConfig, data.get("detector", {})),
            headless=data.get("headless", False),
            deterministic_seed=data.get("deterministic_seed", 42)
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        from dataclasses import asdict
        data = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def validate(self) -> List[str]:
        """Validate configuration values. Returns list of error messages (empty if valid)."""
        errors = []
        
        if self.video.fps <= 0:
            errors.append(f"video.fps must be positive, got {self.video.fps}")
        if self.video.width <= 0 or self.video.height <= 0:
            errors.append(f"video dimensions must be positive, got {self.video.width}x{self.video.height}")
        if self.anomaly.velocity_z_threshold <= 0:
            errors.append(f"anomaly.velocity_z_threshold must be positive, got {self.anomaly.velocity_z_threshold}")
        if self.anomaly.direction_z_threshold <= 0:
            errors.append(f"anomaly.direction_z_threshold must be positive, got {self.anomaly.direction_z_threshold}")
        if self.anomaly.loiter_speed_threshold < 0:
            errors.append(f"anomaly.loiter_speed_threshold must be non-negative, got {self.anomaly.loiter_speed_threshold}")
        if self.anomaly.loiter_time_threshold <= 0:
            errors.append(f"anomaly.loiter_time_threshold must be positive, got {self.anomaly.loiter_time_threshold}")
        if self.anomaly.history_window < 2:
            errors.append(f"anomaly.history_window must be >= 2, got {self.anomaly.history_window}")
        if self.bev.map_width <= 0 or self.bev.map_height <= 0:
            errors.append(f"bev map dimensions must be positive, got {self.bev.map_width}x{self.bev.map_height}")
        if self.bev.meters_per_pixel <= 0:
            errors.append(f"bev.meters_per_pixel must be positive, got {self.bev.meters_per_pixel}")
        if len(self.bev.calibration_points) != 4:
            errors.append(f"bev.calibration_points must have exactly 4 points, got {len(self.bev.calibration_points)}")
        if self.tracking.iou_threshold < 0 or self.tracking.iou_threshold > 1:
            errors.append(f"tracking.iou_threshold must be in [0, 1], got {self.tracking.iou_threshold}")
        if self.tracking.max_age < 1:
            errors.append(f"tracking.max_age must be >= 1, got {self.tracking.max_age}")
        if self.tracking.min_hits < 1:
            errors.append(f"tracking.min_hits must be >= 1, got {self.tracking.min_hits}")
        if self.tracking.tracker_type not in {"simple", "sort"}:
            errors.append(f"tracking.tracker_type must be 'simple' or 'sort', got '{self.tracking.tracker_type}'")
        if self.detector.acceleration not in {"none", "onnx", "tensorrt"}:
            errors.append(f"detector.acceleration must be 'none', 'onnx', or 'tensorrt', got '{self.detector.acceleration}'")
        
        return errors
