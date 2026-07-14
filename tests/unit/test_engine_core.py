"""
Defense COP v2.0 - Engine Core Unit Tests
Tests for COPEngine pipeline orchestration.
"""
import pytest
import numpy as np

from core.config import COPConfig
from core.time_provider import DeterministicTimeProvider
from core.telemetry import TelemetryLogger
from engine.engine_core import COPEngine, PipelineOutput
from engine.anomaly_engine import AnomalyScore


class TestCOPEngine:
    """Test COPEngine orchestration."""
    
    @pytest.fixture
    def time_provider(self):
        return DeterministicTimeProvider(start_time=1000.0)
    
    @pytest.fixture
    def telemetry(self, time_provider):
        return TelemetryLogger(time_provider=time_provider, enabled=True)
    
    @pytest.fixture
    def config(self):
        config = COPConfig()
        config.bev.enabled = False  # Disable BEV for simpler tests
        config.tracking.min_hits = 1  # Confirm targets immediately
        return config
    
    @pytest.fixture
    def config_with_bev(self):
        config = COPConfig()
        config.tracking.min_hits = 1
        return config
    
    @pytest.fixture
    def engine(self, config, time_provider, telemetry):
        return COPEngine(
            config=config,
            time_provider=time_provider,
            telemetry_logger=telemetry
        )
    
    @pytest.fixture
    def engine_with_bev(self, config_with_bev, time_provider, telemetry):
        return COPEngine(
            config=config_with_bev,
            time_provider=time_provider,
            telemetry_logger=telemetry
        )
    
    def test_process_frame_returns_pipeline_output(self, engine):
        """Test that process_frame returns typed PipelineOutput."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [((100.0, 100.0, 200.0, 200.0), "person", 0.9)]
        
        targets, output = engine.process_frame(frame, detections)
        
        assert isinstance(output, PipelineOutput)
        assert isinstance(output.frame_id, int)
        assert isinstance(output.anomaly_scores, dict)
        assert isinstance(output.threat_levels, dict)
        assert isinstance(output.target_count, int)
    
    def test_process_frame_empty_detections(self, engine):
        """Test processing with no detections."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        targets, output = engine.process_frame(frame, [])
        
        assert len(targets) == 0
        assert output.target_count == 0
        assert len(output.anomaly_scores) == 0
        assert len(output.threat_levels) == 0
    
    def test_frame_counter_increments(self, engine):
        """Test frame counter increments correctly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        _, output1 = engine.process_frame(frame, [])
        _, output2 = engine.process_frame(frame, [])
        _, output3 = engine.process_frame(frame, [])
        
        assert output1.frame_id == 0
        assert output2.frame_id == 1
        assert output3.frame_id == 2
        assert engine.get_frame_count() == 3
    
    def test_bev_canvas_none_when_disabled(self, engine):
        """Test BEV canvas is None when BEV is disabled."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        _, output = engine.process_frame(frame, [])
        
        assert output.bev_canvas is None
    
    def test_bev_canvas_present_when_enabled(self, engine_with_bev):
        """Test BEV canvas is present when BEV is enabled."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [((100.0, 100.0, 200.0, 200.0), "person", 0.9)]
        
        targets, output = engine_with_bev.process_frame(frame, detections)
        
        assert output.bev_canvas is not None
        assert output.bev_canvas.shape[2] == 3  # 3-channel image
    
    def test_anomaly_scores_per_target(self, engine):
        """Test anomaly scores are computed for each target."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            ((100.0, 100.0, 200.0, 200.0), "person", 0.9),
            ((300.0, 300.0, 400.0, 400.0), "person", 0.8),
        ]
        
        targets, output = engine.process_frame(frame, detections)
        
        for target in targets:
            assert target.id in output.anomaly_scores
            score = output.anomaly_scores[target.id]
            assert isinstance(score, AnomalyScore)
    
    def test_threat_levels_per_target(self, engine):
        """Test threat levels are computed for each target."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [((100.0, 100.0, 200.0, 200.0), "person", 0.9)]
        
        targets, output = engine.process_frame(frame, detections)
        
        for target in targets:
            assert target.id in output.threat_levels
    
    def test_shutdown_closes_telemetry(self, engine, telemetry):
        """Test that shutdown closes the telemetry logger."""
        engine.shutdown()
        # After shutdown, the telemetry file should be closed
        assert telemetry._log_file is None
    
    def test_target_count_matches_targets(self, engine):
        """Test that target_count matches number of returned targets."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            ((100.0, 100.0, 200.0, 200.0), "person", 0.9),
            ((300.0, 300.0, 400.0, 400.0), "person", 0.85),
        ]
        
        targets, output = engine.process_frame(frame, detections)
        
        assert output.target_count == len(targets)