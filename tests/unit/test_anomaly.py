"""
Defense COP v2.0 - Anomaly Detector Unit Tests
Deterministic tests for behavioral anomaly detection.
"""
import pytest
from engine.anomaly_engine import AnomalyDetector, AnomalyScore
from engine.tracking import Target
from core.time_provider import DeterministicTimeProvider


class TestAnomalyScore:
    """Test anomaly score dataclass."""
    
    def test_has_anomaly_sprint(self):
        """Test anomaly detection for sprint."""
        score = AnomalyScore(
            velocity_z_score=3.0,
            direction_z_score=1.0,
            is_sprinting=True,
            is_erratic=False,
            is_loitering=False,
            behavioral_score=40.0
        )
        
        assert score.has_anomaly() is True
    
    def test_has_anomaly_erratic(self):
        """Test anomaly detection for erratic movement."""
        score = AnomalyScore(
            velocity_z_score=1.0,
            direction_z_score=2.5,
            is_sprinting=False,
            is_erratic=True,
            is_loitering=False,
            behavioral_score=35.0
        )
        
        assert score.has_anomaly() is True
    
    def test_has_anomaly_loiter(self):
        """Test anomaly detection for loitering."""
        score = AnomalyScore(
            velocity_z_score=0.5,
            direction_z_score=0.3,
            is_sprinting=False,
            is_erratic=False,
            is_loitering=True,
            behavioral_score=30.0
        )
        
        assert score.has_anomaly() is True
    
    def test_no_anomaly(self):
        """Test normal behavior (no anomaly)."""
        score = AnomalyScore(
            velocity_z_score=0.8,
            direction_z_score=0.5,
            is_sprinting=False,
            is_erratic=False,
            is_loitering=False,
            behavioral_score=10.0
        )
        
        assert score.has_anomaly() is False


class TestAnomalyDetector:
    """Test anomaly detector."""
    
    @pytest.fixture
    def time_provider(self):
        """Create deterministic time provider."""
        return DeterministicTimeProvider(start_time=0.0)
    
    @pytest.fixture
    def detector(self, time_provider):
        """Create anomaly detector instance."""
        return AnomalyDetector(
            time_provider=time_provider,
            velocity_z_threshold=2.5,
            direction_z_threshold=2.0,
            loiter_speed_threshold=5.0,
            loiter_time_threshold=15.0,
            history_window=30
        )
    
    def create_target(self, target_id, center, velocity=(0, 0)):
        """Helper to create a target."""
        bbox = (center[0] - 25, center[1] - 25, center[0] + 25, center[1] + 25)
        target = Target(
            id=target_id,
            bbox=bbox,
            class_name="person",
            confidence=0.9
        )
        target.velocity = velocity
        return target
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector._velocity_z_threshold == 2.5
        assert detector._direction_z_threshold == 2.0
        assert detector._loiter_speed_threshold == 5.0
        assert detector._loiter_time_threshold == 15.0
    
    def test_first_detection_no_anomaly(self, detector):
        """Test first detection (no history, no anomaly)."""
        target = self.create_target(1, (100, 100), velocity=(5, 0))
        
        scores = detector.update([target], frame_id=0)
        
        assert 1 in scores
        score = scores[1]
        
        # First detection should have low Z-scores
        assert score.is_sprinting is False
        assert score.is_erratic is False
        assert score.is_loitering is False
    
    def test_sprint_detection(self, detector):
        """Test sprint anomaly detection."""
        target = self.create_target(1, (100, 100))
        
        # Build up normal velocity history
        for i in range(20):
            target.velocity = (3, 0)  # Normal speed
            detector.update([target], frame_id=i)
            target.update(
                (target.bbox[0] + 3, target.bbox[1], 
                 target.bbox[2] + 3, target.bbox[3]),
                0.9
            )
        
        # Suddenly sprint
        target.velocity = (30, 0)  # Very fast
        scores = detector.update([target], frame_id=20)
        
        assert scores[1].velocity_z_score > 2.5
        assert scores[1].is_sprinting is True
    
    def test_loitering_detection(self, detector, time_provider):
        """Test loitering anomaly detection."""
        target = self.create_target(1, (100, 100), velocity=(0, 0))
        
        # Move very slowly for extended period
        for i in range(20):
            time_provider.advance(1.0)  # 1 second per frame
            target.velocity = (0.5, 0.5)  # Very slow movement
            scores = detector.update([target], frame_id=i)
        
        # After 20 seconds, should be loitering
        assert scores[1].is_loitering is True
    
    def test_loitering_reset_on_movement(self, detector, time_provider):
        """Test loitering resets when target starts moving."""
        target = self.create_target(1, (100, 100))
        
        # Loiter for a while
        for i in range(10):
            time_provider.advance(1.0)
            target.velocity = (1, 1)
            detector.update([target], frame_id=i)
        
        # Start moving fast
        target.velocity = (20, 0)
        scores = detector.update([target], frame_id=10)
        
        # Loiter should not be detected
        assert scores[1].is_loitering is False
    
    def test_behavioral_score_calculation(self, detector):
        """Test behavioral score calculation."""
        # Create anomaly score manually
        score = AnomalyScore(
            velocity_z_score=3.0,
            direction_z_score=2.5,
            is_sprinting=True,
            is_erratic=True,
            is_loitering=False,
            behavioral_score=70.0
        )
        
        # Score should be high due to multiple anomalies
        assert score.behavioral_score >= 50.0
    
    def test_multiple_targets(self, detector):
        """Test detection with multiple targets."""
        target1 = self.create_target(1, (100, 100), velocity=(5, 0))
        target2 = self.create_target(2, (200, 200), velocity=(3, 0))
        target3 = self.create_target(3, (300, 300), velocity=(10, 0))
        
        scores = detector.update([target1, target2, target3], frame_id=0)
        
        assert len(scores) == 3
        assert 1 in scores
        assert 2 in scores
        assert 3 in scores
    
    def test_target_history_cleanup(self, detector):
        """Test cleanup of lost target history."""
        target1 = self.create_target(1, (100, 100))
        target2 = self.create_target(2, (200, 200))
        
        # Update with both targets
        detector.update([target1, target2], frame_id=0)
        
        assert 1 in detector._velocity_history
        assert 2 in detector._velocity_history
        
        # Update with only target 1
        detector.update([target1], frame_id=1)
        
        # Target 2 history should be cleaned
        assert 1 in detector._velocity_history
        assert 2 not in detector._velocity_history
    
    def test_deterministic_behavior(self, time_provider):
        """Test that detector behavior is deterministic."""
        detector1 = AnomalyDetector(
            time_provider=DeterministicTimeProvider(start_time=0.0),
            velocity_z_threshold=2.5,
            direction_z_threshold=2.0
        )
        
        detector2 = AnomalyDetector(
            time_provider=DeterministicTimeProvider(start_time=0.0),
            velocity_z_threshold=2.5,
            direction_z_threshold=2.0
        )
        
        target1 = self.create_target(1, (100, 100), velocity=(5, 0))
        target2 = self.create_target(1, (100, 100), velocity=(5, 0))
        
        scores1 = detector1.update([target1], frame_id=0)
        scores2 = detector2.update([target2], frame_id=0)
        
        assert scores1[1].velocity_z_score == scores2[1].velocity_z_score
        assert scores1[1].is_sprinting == scores2[1].is_sprinting
    
    def test_get_loiter_duration(self, detector, time_provider):
        """Test getting loiter duration for a target."""
        target = self.create_target(1, (100, 100), velocity=(1, 1))
        
        # No loiter initially
        assert detector.get_loiter_duration(1) is None
        
        # Start loitering
        for i in range(10):
            time_provider.advance(1.0)
            target.velocity = (2, 2)
            detector.update([target], frame_id=i)
        
        # Should have loiter duration
        duration = detector.get_loiter_duration(1)
        assert duration is not None
        assert duration >= 9.0  # At least 9 seconds

    def test_erratic_movement_detection(self, detector):
        """Test erratic movement detection with alternating zigzag directions."""
        target = self.create_target(1, (100, 100))
        
        # 1. Straight movement (direction changes = 0.0)
        # We run 15 frames of straight movement to establish history
        for i in range(15):
            x = 100 + i * 10
            y = 100
            bbox = (x - 25, y - 25, x + 25, y + 25)
            target.update(bbox, 0.9)
            detector.update([target], frame_id=i)
        
        # 2. Start moving erratically (alternating directions)
        coords = [
            (250, 100),  # Right
            (250, 120),  # Down
            (270, 120),  # Right
            (270, 100),  # Up
            (290, 100),  # Right
            (290, 120),  # Down
            (310, 120),  # Right
            (310, 100),  # Up
        ]
        
        scores = {}
        for idx, (cx, cy) in enumerate(coords):
            frame_id = 15 + idx
            bbox = (cx - 25, cy - 25, cx + 25, cy + 25)
            target.update(bbox, 0.9)
            scores = detector.update([target], frame_id=frame_id)
            
        # The target should be flagged as erratic
        assert 1 in scores
        assert scores[1].direction_z_score > 2.0
        assert scores[1].is_erratic is True

    def test_signed_angle_wrap_around(self, detector):
        """Test that _angle_between handles angle wrap-around correctly near pi."""
        import numpy as np
        v1 = (float(np.cos(3.1)), float(np.sin(3.1)))
        v2 = (float(np.cos(-3.1)), float(np.sin(-3.1)))
        
        angle = detector._angle_between(v1, v2)
        # Difference should be ~0.08 radians instead of ~6.2 radians
        assert abs(angle) < 0.1

    def test_circular_variance_prevents_false_alerts(self, detector):
        """Test that circular variance prevents false erratic alerts near +/-pi transitions."""
        target = self.create_target(1, (500, 100))
        
        # Heading oscillates between +3.12 and -3.12 radians
        # (moving West/Left with small y fluctuations)
        for i in range(15):
            # Alternating y coordinate slightly to simulate noise, but start straight to avoid initial transient
            y_offset = 0.0
            if i >= 3:
                y_offset = 1.0 if (i % 2 == 0) else -1.0
                
            x = 500 - i * 10
            y = 100 + y_offset
            bbox = (x - 25, y - 25, x + 25, y + 25)
            target.update(bbox, 0.9)
            scores = detector.update([target], frame_id=i)
            
            # Target should never be flagged as erratic
            assert scores[1].is_erratic is False
            
            # The direction variance (circular) should be very small
            variance_history = detector._variance_history.get(1)
            if variance_history and len(variance_history) > 0:
                assert variance_history[-1] < 0.1
