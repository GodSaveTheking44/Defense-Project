"""
Defense COP v2.0 - Telemetry Unit Tests
Tests for TelemetryLogger structured event logging.
"""
import json
import pytest

from core.telemetry import TelemetryLogger, TelemetrySeverity, TelemetryEvent
from core.time_provider import DeterministicTimeProvider


class TestTelemetryEvent:
    """Test TelemetryEvent dataclass."""
    
    def test_to_json(self):
        event = TelemetryEvent(
            timestamp=1000.0,
            event_type="test.event",
            severity="INFO",
            data={"key": "value"},
            frame_id=42
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["timestamp"] == 1000.0
        assert parsed["event_type"] == "test.event"
        assert parsed["severity"] == "INFO"
        assert parsed["data"]["key"] == "value"
        assert parsed["frame_id"] == 42
    
    def test_to_json_no_frame(self):
        event = TelemetryEvent(
            timestamp=1000.0,
            event_type="test.event",
            severity="DEBUG",
            data={}
        )
        parsed = json.loads(event.to_json())
        assert parsed["frame_id"] is None


class TestTelemetryLogger:
    """Test TelemetryLogger."""
    
    @pytest.fixture
    def time_provider(self):
        return DeterministicTimeProvider(start_time=1000.0)
    
    @pytest.fixture
    def logger_memory_only(self, time_provider):
        """Logger that only logs to memory (no file)."""
        return TelemetryLogger(time_provider=time_provider, enabled=True)
    
    @pytest.fixture
    def logger_with_file(self, time_provider, tmp_path):
        """Logger that writes to a file."""
        return TelemetryLogger(
            time_provider=time_provider,
            output_dir=tmp_path,
            enabled=True
        )
    
    def test_log_event_to_memory(self, logger_memory_only):
        """Test events are stored in memory."""
        logger_memory_only.log(
            event_type="test.event",
            severity=TelemetrySeverity.INFO,
            data={"key": "value"}
        )
        
        events = logger_memory_only.get_events()
        assert len(events) == 1
        assert events[0].event_type == "test.event"
    
    def test_log_event_to_file(self, logger_with_file, tmp_path):
        """Test events are written to JSONL file."""
        logger_with_file.log(
            event_type="test.event",
            severity=TelemetrySeverity.INFO,
            data={"key": "value"},
            frame_id=1
        )
        logger_with_file.close()
        
        # Find the JSONL file
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        
        with open(jsonl_files[0]) as f:
            lines = f.readlines()
        
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event_type"] == "test.event"
        assert parsed["frame_id"] == 1
    
    def test_disabled_logger(self, time_provider):
        """Test disabled logger produces no output."""
        logger = TelemetryLogger(time_provider=time_provider, enabled=False)
        logger.log(
            event_type="test.event",
            severity=TelemetrySeverity.INFO,
            data={}
        )
        assert len(logger.get_events()) == 0
    
    def test_log_anomaly_sprint(self, logger_memory_only):
        """Test sprint anomaly convenience method."""
        logger_memory_only.log_anomaly_sprint(
            target_id=5, velocity=45.2, z_score=3.1, frame_id=100
        )
        
        events = logger_memory_only.get_events()
        assert len(events) == 1
        assert events[0].event_type == "alert.anomaly.sprint"
        assert events[0].severity == "ALERT"
        assert events[0].data["target_id"] == 5
    
    def test_log_anomaly_erratic(self, logger_memory_only):
        """Test erratic anomaly convenience method."""
        logger_memory_only.log_anomaly_erratic(
            target_id=3, direction_variance=0.45, z_score=2.8, frame_id=50
        )
        
        events = logger_memory_only.get_events()
        assert events[0].event_type == "alert.anomaly.erratic"
        assert events[0].data["direction_variance"] == 0.45
    
    def test_log_anomaly_loiter(self, logger_memory_only):
        """Test loiter anomaly convenience method."""
        logger_memory_only.log_anomaly_loiter(
            target_id=7, duration=18.5, avg_speed=2.1, frame_id=445
        )
        
        events = logger_memory_only.get_events()
        assert events[0].event_type == "alert.anomaly.loiter"
        assert events[0].severity == "WARNING"
    
    def test_context_manager(self, time_provider, tmp_path):
        """Test context manager closes file."""
        with TelemetryLogger(
            time_provider=time_provider,
            output_dir=tmp_path,
            enabled=True
        ) as logger:
            logger.log(
                event_type="test",
                severity=TelemetrySeverity.INFO,
                data={}
            )
        
        # File should be closed after context manager exits
        assert logger._log_file is None
        
        # File should contain the event
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
    
    def test_deterministic_timestamps(self, time_provider, logger_memory_only):
        """Test timestamps use the deterministic time provider."""
        logger_memory_only.log(
            event_type="event1",
            severity=TelemetrySeverity.INFO,
            data={}
        )
        time_provider.advance(5.0)
        logger_memory_only.log(
            event_type="event2",
            severity=TelemetrySeverity.INFO,
            data={}
        )
        
        events = logger_memory_only.get_events()
        assert events[0].timestamp == 1000.0
        assert events[1].timestamp == 1005.0
    
    def test_multiple_events(self, logger_memory_only):
        """Test logging multiple events."""
        for i in range(10):
            logger_memory_only.log(
                event_type=f"event.{i}",
                severity=TelemetrySeverity.DEBUG,
                data={"index": i}
            )
        
        events = logger_memory_only.get_events()
        assert len(events) == 10
    
    def test_close_idempotent(self, logger_with_file):
        """Test that calling close multiple times is safe."""
        logger_with_file.close()
        logger_with_file.close()  # Should not raise
    
    def test_bev_calibration_log(self, logger_memory_only):
        """Test BEV calibration event logging."""
        points = [(100, 400, 0, 0), (540, 400, 10, 0)]
        logger_memory_only.log_bev_calibration(points)
        
        events = logger_memory_only.get_events()
        assert len(events) == 1
        assert events[0].event_type == "bev.calibrated"
        assert events[0].severity == "INFO"

    def test_log_cot_broadcasting(self, time_provider):
        """Test log_cot streams correct Cursor-on-Target XML formatted packets."""
        from unittest.mock import MagicMock
        
        logger = TelemetryLogger(
            time_provider=time_provider,
            enabled=True,
            cot_enabled=True,
            cot_host="127.0.0.1",
            cot_port=4242
        )
        
        mock_socket = MagicMock()
        logger._cot_socket = mock_socket
        
        logger.log_cot(
            target_id=5,
            lat=34.0522,
            lon=-118.2437,
            threat_class="CRITICAL",
            callsign="Target #5"
        )
        
        assert mock_socket.sendto.called
        args, kwargs = mock_socket.sendto.call_args
        
        payload_bytes = args[0]
        payload = payload_bytes.decode("utf-8")
        
        assert "uid=\"DefenseCOP.Target.5\"" in payload
        assert "type=\"a-h-G-p-i\"" in payload
        assert "lat=\"34.0522000\"" in payload
        assert "lon=\"-118.2437000\"" in payload
        assert "Target #5" in payload
        
        # Test low-threat mapping
        logger.log_cot(
            target_id=6,
            lat=34.0522,
            lon=-118.2437,
            threat_class="LOW",
            callsign="Target #6"
        )
        args_low = mock_socket.sendto.call_args_list[1][0]
        payload_low = args_low[0].decode("utf-8")
        assert "type=\"a-f-G-p-i\"" in payload_low

    def test_log_cot_socket_failure_warning(self, time_provider):
        """Test that OSError during CoT streaming is logged as a warning."""
        from unittest.mock import MagicMock, patch
        
        telemetry_logger = TelemetryLogger(
            time_provider=time_provider,
            enabled=True,
            cot_enabled=True,
            cot_host="127.0.0.1",
            cot_port=4242
        )
        
        mock_socket = MagicMock()
        mock_socket.sendto.side_effect = OSError("Socket error")
        telemetry_logger._cot_socket = mock_socket
        
        with patch("core.telemetry.logger.warning") as mock_warning:
            # Should not raise exception
            telemetry_logger.log_cot(
                target_id=5,
                lat=34.0522,
                lon=-118.2437,
                threat_class="CRITICAL",
                callsign="Target #5"
            )
            mock_warning.assert_called_once()
            assert "Failed to stream CoT packet over UDP" in mock_warning.call_args[0][0]

    def test_close_socket_error_handling(self, time_provider):
        """Test that socket close errors during cleanup are caught and logged as debug."""
        from unittest.mock import MagicMock, patch
        
        telemetry_logger = TelemetryLogger(
            time_provider=time_provider,
            enabled=True,
            cot_enabled=True,
            cot_host="127.0.0.1",
            cot_port=4242
        )
        
        mock_socket = MagicMock()
        mock_socket.close.side_effect = Exception("Close error")
        telemetry_logger._cot_socket = mock_socket
        
        with patch("core.telemetry.logger.debug") as mock_debug:
            telemetry_logger.close()
            # Assert logger.debug was called with closing error details
            mock_debug.assert_any_call("Error closing CoT UDP socket: Close error")