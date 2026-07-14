"""
Defense COP v2.0 - Telemetry System
JSON structured logging with deterministic timestamps for audit and replay.
"""
import json
import socket
import datetime
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional, List
from pathlib import Path
from core.time_provider import TimeProvider

logger = logging.getLogger("DefenseCOP.Telemetry")


class TelemetrySeverity(Enum):
    """Telemetry event severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ALERT = "ALERT"
    CRITICAL = "CRITICAL"


@dataclass
class TelemetryEvent:
    """Structured telemetry event."""
    timestamp: float
    event_type: str
    severity: str
    data: Dict[str, Any]
    frame_id: Optional[int] = None
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), sort_keys=True)


class TelemetryLogger:
    """Structured telemetry logger with deterministic timestamps."""
    
    # Constants for default configurations
    DEFAULT_COT_HOST = "239.2.3.1"
    DEFAULT_COT_PORT = 24910
    DEFAULT_MAX_EVENTS = 100_000
    COT_STALE_DURATION_SECONDS = 15.0
    COT_MULTICAST_TTL = 2
    
    def __init__(
        self,
        time_provider: TimeProvider,
        output_dir: Optional[Path] = None,
        enabled: bool = True,
        max_events: int = DEFAULT_MAX_EVENTS,
        cot_enabled: bool = False,
        cot_host: str = DEFAULT_COT_HOST,
        cot_port: int = DEFAULT_COT_PORT
    ):
        """
        Initialize telemetry logger.
        
        Args:
            time_provider: Deterministic time source
            output_dir: Directory for telemetry files
            enabled: Whether logging is enabled
            max_events: Maximum number of events to keep in memory
            cot_enabled: Whether Cursor-on-Target streaming is enabled
            cot_host: Target UDP host for CoT packets
            cot_port: Target UDP port for CoT packets
        """
        self._time_provider = time_provider
        self._enabled = enabled
        self._events: List[TelemetryEvent] = []
        self._output_dir = output_dir
        self._log_file = None
        self._max_events = max_events
        self._cot_enabled = cot_enabled
        self._cot_host = cot_host
        self._cot_port = cot_port
        self._cot_socket = None
        
        if self._enabled and self._output_dir:
            try:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                log_path = self._output_dir / f"telemetry_{int(self._time_provider.now())}.jsonl"
                self._log_file = open(log_path, "w")
            except OSError as e:
                logger.warning(
                    f"Failed to open telemetry file: {e}. Logging to memory only."
                )
        
        # Setup UDP socket for ATAK streaming if enabled
        if self._enabled and self._cot_enabled:
            try:
                self._cot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # If multicast address, configure TTL
                if self._cot_host.startswith("239.") or self._cot_host.startswith("224."):
                    self._cot_socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.COT_MULTICAST_TTL)
            except Exception as e:
                logger.warning(
                    f"Failed to initialize CoT UDP socket: {e}. Streaming disabled."
                )
    
    def log(
        self,
        event_type: str,
        severity: TelemetrySeverity,
        data: Dict[str, Any],
        frame_id: Optional[int] = None
    ) -> None:
        """
        Log a telemetry event.
        
        Args:
            event_type: Event type identifier (e.g., "bev.calibrated")
            severity: Event severity level
            data: Event data payload
            frame_id: Optional frame ID for video correlation
        """
        if not self._enabled:
            return
        
        event = TelemetryEvent(
            timestamp=self._time_provider.now(),
            event_type=event_type,
            severity=severity.value,
            data=data,
            frame_id=frame_id
        )
        
        self._events.append(event)
        
        # Cap events to prevent unbounded memory growth
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        
        if self._log_file:
            self._log_file.write(event.to_json() + "\n")
            self._log_file.flush()
            
    def log_cot(
        self,
        target_id: int,
        lat: float,
        lon: float,
        threat_class: str,
        callsign: str
    ) -> None:
        """Send Cursor-on-Target (CoT) XML over UDP socket."""
        if not self._enabled or not self._cot_enabled or not self._cot_socket:
            return
            
        # Determine military symbol and CoT type based on threat class
        if threat_class in {"CRITICAL", "HIGH"}:
            cot_type = "a-h-G-p-i"  # Hostile Ground Infantry
        elif threat_class == "MEDIUM":
            cot_type = "a-u-G-p-i"  # Unknown Ground Infantry
        else:
            cot_type = "a-f-G-p-i"  # Friendly Ground Infantry
            
        now = datetime.datetime.now(datetime.timezone.utc)
        time_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        stale_time = now + datetime.timedelta(seconds=self.COT_STALE_DURATION_SECONDS)
        stale_str = stale_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        uid = f"DefenseCOP.Target.{target_id}"
        
        xml = (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<event version="2.0" uid="{uid}" type="{cot_type}" '
            f'time="{time_str}" start="{time_str}" stale="{stale_str}" how="m-g">\n'
            f'  <point lat="{lat:.7f}" lon="{lon:.7f}" hae="0.0" ce="10.0" le="10.0" />\n'
            f'  <detail>\n'
            f'    <contact callsign="{callsign}" />\n'
            f'    <status status="alert" />\n'
            f'  </detail>\n'
            f'</event>'
        )
        
        try:
            self._cot_socket.sendto(xml.encode("utf-8"), (self._cot_host, self._cot_port))
        except OSError as e:
            # Catch network/socket errors to maintain pipeline stability
            logger.warning(f"Failed to stream CoT packet over UDP to {(self._cot_host, self._cot_port)}: {e}")
    
    def log_bev_calibration(self, calibration_points: List[tuple]) -> None:
        """Log BEV calibration event."""
        self.log(
            event_type="bev.calibrated",
            severity=TelemetrySeverity.INFO,
            data={"calibration_points": calibration_points}
        )
    
    def log_anomaly_sprint(
        self,
        target_id: int,
        velocity: float,
        z_score: float,
        frame_id: int
    ) -> None:
        """Log sprint anomaly detection."""
        self.log(
            event_type="alert.anomaly.sprint",
            severity=TelemetrySeverity.ALERT,
            data={
                "target_id": target_id,
                "velocity": velocity,
                "z_score": z_score
            },
            frame_id=frame_id
        )
    
    def log_anomaly_erratic(
        self,
        target_id: int,
        direction_variance: float,
        z_score: float,
        frame_id: int
    ) -> None:
        """Log erratic movement anomaly."""
        self.log(
            event_type="alert.anomaly.erratic",
            severity=TelemetrySeverity.ALERT,
            data={
                "target_id": target_id,
                "direction_variance": direction_variance,
                "z_score": z_score
            },
            frame_id=frame_id
        )
    
    def log_anomaly_loiter(
        self,
        target_id: int,
        duration: float,
        avg_speed: float,
        frame_id: int
    ) -> None:
        """Log loitering anomaly."""
        self.log(
            event_type="alert.anomaly.loiter",
            severity=TelemetrySeverity.WARNING,
            data={
                "target_id": target_id,
                "duration_seconds": duration,
                "avg_speed": avg_speed
            },
            frame_id=frame_id
        )
    
    def log_target_detected(
        self,
        target_id: int,
        class_name: str,
        confidence: float,
        bbox: List[float],
        frame_id: int
    ) -> None:
        """Log new target detection."""
        self.log(
            event_type="tracking.target.detected",
            severity=TelemetrySeverity.INFO,
            data={
                "target_id": target_id,
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox
            },
            frame_id=frame_id
        )
    
    def log_target_lost(self, target_id: int, frame_id: int) -> None:
        """Log target lost event."""
        self.log(
            event_type="tracking.target.lost",
            severity=TelemetrySeverity.INFO,
            data={"target_id": target_id},
            frame_id=frame_id
        )
    
    def get_events(self) -> List[TelemetryEvent]:
        """Retrieve all logged events."""
        return self._events.copy()
    
    def close(self) -> None:
        """Close telemetry logger and flush buffers."""
        if self._log_file:
            try:
                self._log_file.close()
            except Exception as e:
                logger.debug(f"Error closing log file: {e}")
            self._log_file = None
        if self._cot_socket:
            try:
                self._cot_socket.close()
            except Exception as e:
                logger.debug(f"Error closing CoT UDP socket: {e}")
            self._cot_socket = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
 
    def __del__(self):
        """Destructor fallback to ensure file handle is closed."""
        try:
            self.close()
        except Exception as e:
            logger.debug(f"Error in destructor close(): {e}")
