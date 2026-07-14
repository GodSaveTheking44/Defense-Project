"""
Defense COP v2.0 - SORT Tracker Unit Tests
"""
import pytest
import numpy as np
from engine.tracking import KalmanFilter, Target, SimpleTracker


class TestKalmanFilter:
    """Test the custom 2D linear Kalman Filter."""

    def test_initialization(self):
        kf = KalmanFilter()
        assert kf.x.shape == (8, 1)
        assert kf.P.shape == (8, 8)
        assert kf.F.shape == (8, 8)
        assert kf.H.shape == (4, 8)
        assert kf.R.shape == (4, 4)
        assert kf.Q.shape == (8, 8)
        assert np.allclose(kf.x, 0)

    def test_predict(self):
        kf = KalmanFilter()
        # Set state: center=(10, 20), size=(30, 40), velocity=(1, 2, 0, 0)
        kf.x[:4] = np.array([[10.0], [20.0], [30.0], [40.0]])
        kf.x[4:6] = np.array([[1.0], [2.0]])
        
        kf.predict()
        
        # Expected predicted state: center=(11, 22), size=(30, 40)
        assert kf.x[0, 0] == 11.0
        assert kf.x[1, 0] == 22.0
        assert kf.x[2, 0] == 30.0
        assert kf.x[3, 0] == 40.0

    def test_update(self):
        kf = KalmanFilter()
        kf.x[:4] = np.array([[10.0], [20.0], [30.0], [40.0]])
        
        # Update with measurement z = [12.0, 18.0, 32.0, 38.0]
        z = np.array([[12.0], [18.0], [32.0], [38.0]])
        kf.update(z)
        
        # Check that the filter pulls state toward measurement
        assert abs(kf.x[0, 0] - 12.0) < 2.0
        assert abs(kf.x[1, 0] - 18.0) < 2.0


class TestSORTTarget:
    """Test Target class Kalman integration."""

    def test_target_kalman_post_init(self):
        bbox = (10.0, 20.0, 50.0, 80.0)  # w=40, h=60, center=(30, 50)
        target = Target(id=1, bbox=bbox, class_name="person", confidence=0.8, use_kalman=True)
        
        assert target.kf is not None
        assert target.kf.x[0, 0] == 30.0
        assert target.kf.x[1, 0] == 50.0
        assert target.kf.x[2, 0] == 40.0
        assert target.kf.x[3, 0] == 60.0

    def test_target_kalman_predict_and_update(self):
        bbox = (10.0, 20.0, 50.0, 80.0)  # center=(30, 50)
        target = Target(id=1, bbox=bbox, class_name="person", confidence=0.8, use_kalman=True)
        
        # Update target with a measurement indicating movement
        new_bbox = (12.0, 22.0, 52.0, 82.0)  # center=(32, 52)
        target.update(new_bbox, 0.95)
        
        # Check that center is updated (Kalman filter weights the prediction and measurement)
        assert target.center[0] == pytest.approx(31.8, abs=0.1)
        assert target.center[1] == pytest.approx(51.8, abs=0.1)
        
        # Predict next step
        target.predict()
        
        # Center should move in the direction of velocity (which is now positive)
        assert target.center[0] > 31.81
        assert target.center[1] > 51.81


class TestSORTTracker:
    """Test SimpleTracker in SORT mode (using Kalman)."""

    def test_sort_tracking_association(self):
        tracker = SimpleTracker(iou_threshold=0.3, max_age=5, min_hits=1, use_kalman=True)
        
        # Frame 1: target birth
        detections1 = [((10.0, 10.0, 30.0, 30.0), "person", 0.9)]
        targets1 = tracker.update(detections1)
        assert len(targets1) == 1
        assert targets1[0].id == 1
        assert targets1[0].use_kalman is True
        
        # Frame 2: track update
        detections2 = [((11.0, 11.0, 31.0, 31.0), "person", 0.9)]
        targets2 = tracker.update(detections2)
        assert len(targets2) == 1
        assert targets2[0].id == 1

    def test_tracking_continuity_through_occlusion(self):
        # min_hits=1 so they are confirmed immediately
        tracker = SimpleTracker(iou_threshold=0.3, max_age=5, min_hits=1, use_kalman=True)
        
        # Frame 1: detect target
        targets = tracker.update([((10.0, 10.0, 30.0, 30.0), "person", 0.9)])
        assert len(targets) == 1
        assert targets[0].id == 1
        
        # Frame 2: target occluded (no detections)
        tracker.update([])
        all_targets = tracker.get_all_targets()
        assert len(all_targets) == 1
        assert all_targets[0].id == 1
        assert all_targets[0].time_since_update == 1
        
        # Bounding box is predicted using Kalman velocity
        x1, y1, x2, y2 = all_targets[0].bbox
        assert x1 is not None
        
        # Frame 3: target reappears and matches
        targets_reappear = tracker.update([((12.0, 12.0, 32.0, 32.0), "person", 0.9)])
        assert len(targets_reappear) == 1
        assert targets_reappear[0].id == 1
        assert targets_reappear[0].time_since_update == 0
