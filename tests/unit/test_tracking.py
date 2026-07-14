"""
Defense COP v2.0 - Tracking Unit Tests
Deterministic tests for SimpleTracker and Target classes.
"""
from engine.tracking import Target, SimpleTracker


class TestTarget:
    """Test Target class logic."""

    def test_target_initialization(self):
        """Test Target center calculation and initial state."""
        bbox = (100.0, 100.0, 200.0, 200.0)
        target = Target(id=1, bbox=bbox, class_name="person", confidence=0.85)

        assert target.id == 1
        assert target.bbox == bbox
        assert target.class_name == "person"
        assert target.confidence == 0.85
        assert target.center == (150.0, 150.0)
        assert target.velocity == (0.0, 0.0)
        assert target.age == 0
        assert target.hits == 1
        assert target.time_since_update == 0
        assert len(target.trajectory) == 1
        assert target.trajectory[0] == (150.0, 150.0)

    def test_target_update(self):
        """Test Target updates coordinates, velocity, and trajectory history."""
        bbox1 = (100.0, 100.0, 200.0, 200.0)
        target = Target(id=1, bbox=bbox1, class_name="person", confidence=0.85)

        bbox2 = (110.0, 105.0, 210.0, 205.0)
        target.update(bbox2, confidence=0.90)

        assert target.bbox == bbox2
        assert target.confidence == 0.90
        assert target.center == (160.0, 155.0)
        assert target.velocity == (10.0, 5.0)  # Difference in centers
        assert target.hits == 2
        assert target.time_since_update == 0
        assert len(target.trajectory) == 2
        assert target.trajectory[-1] == (160.0, 155.0)

    def test_target_predict_constant_velocity(self):
        """Test target coordinate projection based on constant velocity."""
        bbox1 = (100.0, 100.0, 200.0, 200.0)
        target = Target(id=1, bbox=bbox1, class_name="person", confidence=0.85)

        # Update to establish velocity (10.0, 5.0)
        bbox2 = (110.0, 105.0, 210.0, 205.0)
        target.update(bbox2, confidence=0.90)

        # Run predict step
        target.predict()

        assert target.age == 1
        assert target.time_since_update == 1
        # Predicted box should shift by (10.0, 5.0)
        assert target.bbox == (120.0, 110.0, 220.0, 210.0)
        assert target.center == (170.0, 160.0)


class TestSimpleTracker:
    """Test SimpleTracker class logic."""

    def test_tracker_target_birth(self):
        """Test that new detections spawn unconfirmed targets."""
        tracker = SimpleTracker(iou_threshold=0.3, max_age=5, min_hits=2)
        detections = [((100.0, 100.0, 200.0, 200.0), "person", 0.9)]

        # First update: Target born, but not confirmed because hits (1) < min_hits (2)
        active_targets = tracker.update(detections)
        assert len(active_targets) == 0

        # Retrieve unconfirmed targets
        all_targets = tracker.get_all_targets()
        assert len(all_targets) == 1
        assert all_targets[0].id == 1
        assert all_targets[0].hits == 1

    def test_tracker_target_confirmation(self):
        """Test target confirmation when hits reach min_hits."""
        tracker = SimpleTracker(iou_threshold=0.3, max_age=5, min_hits=2)
        det = ((100.0, 100.0, 200.0, 200.0), "person", 0.9)

        # Frame 1: Birth (hits=1)
        active = tracker.update([det])
        assert len(active) == 0

        # Frame 2: Match (hits=2)
        active = tracker.update([det])
        assert len(active) == 1
        assert active[0].id == 1
        assert active[0].hits == 2

    def test_tracker_expiry(self):
        """Test target deletion after time_since_update exceeds max_age."""
        tracker = SimpleTracker(iou_threshold=0.3, max_age=2, min_hits=1)
        det = ((100.0, 100.0, 200.0, 200.0), "person", 0.9)

        # Frame 1: Active
        active = tracker.update([det])
        assert len(active) == 1

        # Frame 2: No detection (age increments)
        active = tracker.update([])
        assert len(active) == 1
        assert tracker.get_all_targets()[0].time_since_update == 1

        # Frame 3: No detection (age = 2)
        active = tracker.update([])
        assert len(active) == 1
        assert tracker.get_all_targets()[0].time_since_update == 2

        # Frame 4: No detection (age = 3 > max_age of 2) -> Removed
        active = tracker.update([])
        assert len(active) == 0
        assert len(tracker.get_all_targets()) == 0

    def test_tracker_hungarian_association(self):
        """Test that tracker correctly matches multiple targets in the same frame."""
        tracker = SimpleTracker(iou_threshold=0.1, max_age=5, min_hits=1)
        
        # Frame 1: Birth of Target 1 and Target 2
        tracker.update([
            ((100.0, 100.0, 150.0, 150.0), "person", 0.9),
            ((200.0, 200.0, 250.0, 250.0), "person", 0.9)
        ])
        
        # Frame 2: Detections shifted slightly
        active = tracker.update([
            ((202.0, 202.0, 252.0, 252.0), "person", 0.9), # matches Target 2
            ((102.0, 102.0, 152.0, 152.0), "person", 0.9)  # matches Target 1
        ])
        
        assert len(active) == 2
        ids = {t.id for t in active}
        assert ids == {1, 2}
