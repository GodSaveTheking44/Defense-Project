"""
Defense COP v2.0 - Renderer Unit Tests
Tests for TacticalRenderer UI rendering logic.
"""
import pytest
import numpy as np
from engine.tracking import Target
from engine.anomaly_engine import AnomalyScore
from engine.threat_engine import ThreatLevel
from ui.renderer import TacticalRenderer


class TestTacticalRenderer:
    """Test TacticalRenderer."""
    
    @pytest.fixture
    def renderer(self):
        return TacticalRenderer(
            show_bev=True,
            show_trajectories=True,
            show_anomaly_score=True,
            show_fps=True
        )
    
    def _make_target(self, target_id, center=(150, 150)):
        bbox = (center[0] - 50.0, center[1] - 50.0, center[0] + 50.0, center[1] + 50.0)
        return Target(id=target_id, bbox=bbox, class_name="person", confidence=0.9)
    
    def _make_score(self, behavioral=10.0):
        return AnomalyScore(
            velocity_z_score=0.5,
            direction_z_score=0.3,
            is_sprinting=False,
            is_erratic=False,
            is_loitering=False,
            behavioral_score=behavioral
        )
    
    def _make_threat(self, target_id, threat_class="LOW"):
        return ThreatLevel(
            target_id=target_id,
            threat_score=10.0,
            threat_class=threat_class,
            contributing_factors=[]
        )
    
    def test_render_frame_same_dimensions(self, renderer):
        """Output should match input frame dimensions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = renderer.render_frame(frame, [], {}, {})
        assert output.shape == frame.shape
    
    def test_render_frame_with_targets(self, renderer):
        """Frame with targets should differ from empty frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        target = self._make_target(1)
        scores = {1: self._make_score()}
        threats = {1: self._make_threat(1)}
        
        empty_output = renderer.render_frame(frame, [], {}, {})
        target_output = renderer.render_frame(frame, [target], scores, threats)
        
        assert not np.array_equal(empty_output, target_output)
    
    def test_bev_overlay_applied(self, renderer):
        """BEV overlay should modify the frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bev_canvas = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        output_no_bev = renderer.render_frame(frame, [], {}, {}, bev_canvas=None)
        output_with_bev = renderer.render_frame(frame, [], {}, {}, bev_canvas=bev_canvas)
        
        assert not np.array_equal(output_no_bev, output_with_bev)
    
    def test_bev_overlay_skipped_when_disabled(self):
        """BEV overlay should be skipped when show_bev=False."""
        renderer = TacticalRenderer(show_bev=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bev_canvas = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Even with a canvas provided, it shouldn't be overlaid
        output = renderer.render_frame(frame, [], {}, {}, bev_canvas=bev_canvas)
        # The output should be the same as no BEV
        output_no_bev = renderer.render_frame(frame, [], {}, {}, bev_canvas=None)
        np.testing.assert_array_equal(output, output_no_bev)
    
    def test_fps_displayed(self, renderer):
        """FPS counter should render on frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output_no_fps = renderer.render_frame(frame, [], {}, {}, fps=None)
        output_with_fps = renderer.render_frame(frame, [], {}, {}, fps=30.0)
        
        assert not np.array_equal(output_no_fps, output_with_fps)
    
    def test_fps_hidden_when_disabled(self):
        """FPS counter should not render when show_fps=False."""
        renderer = TacticalRenderer(show_fps=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        output = renderer.render_frame(frame, [], {}, {}, fps=30.0)
        output_no_fps = renderer.render_frame(frame, [], {}, {}, fps=None)
        np.testing.assert_array_equal(output, output_no_fps)
    
    def test_scale_at_reference_resolution(self, renderer):
        """Scale should be ~1.0 at 640x480."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert renderer._scale(10, frame) == 10
    
    def test_scale_at_double_resolution(self, renderer):
        """Scale should double at 1280x960."""
        frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        assert renderer._scale(10, frame) == 20
    
    def test_scale_at_half_resolution(self, renderer):
        """Scale should halve at 320x240."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        assert renderer._scale(10, frame) == 5
    
    def test_scale_minimum_one(self, renderer):
        """Scale should never return 0."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        assert renderer._scale(1, frame) >= 1
    
    def test_scale_font(self, renderer):
        """Font scale should scale proportionally."""
        frame_640 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_1280 = np.zeros((960, 1280, 3), dtype=np.uint8)
        
        font_640 = renderer._scale_font(0.5, frame_640)
        font_1280 = renderer._scale_font(0.5, frame_1280)
        
        assert font_1280 == pytest.approx(font_640 * 2.0)
    
    def test_render_different_resolutions(self, renderer):
        """Renderer should not crash at various resolutions."""
        for h, w in [(240, 320), (480, 640), (720, 1280), (1080, 1920)]:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            target = self._make_target(1, center=(w // 2, h // 2))
            scores = {1: self._make_score()}
            threats = {1: self._make_threat(1)}
            
            output = renderer.render_frame(frame, [target], scores, threats, fps=30.0)
            assert output.shape == (h, w, 3)