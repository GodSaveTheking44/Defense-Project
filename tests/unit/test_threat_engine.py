"""
Defense COP v2.0 - Threat Engine Unit Tests
Tests for ThreatEngine threat classification.
"""
import pytest
from engine.threat_engine import ThreatEngine, ThreatLevel
from engine.anomaly_engine import AnomalyScore
from engine.tracking import Target


class TestThreatLevel:
    """Test ThreatLevel dataclass."""
    
    def test_threat_level_creation(self):
        level = ThreatLevel(
            target_id=1,
            threat_score=75.0,
            threat_class="CRITICAL",
            contributing_factors=["SPRINT_DETECTED"]
        )
        assert level.target_id == 1
        assert level.threat_score == 75.0
        assert level.threat_class == "CRITICAL"
        assert len(level.contributing_factors) == 1


class TestThreatEngine:
    """Test ThreatEngine classification."""
    
    @pytest.fixture
    def engine(self):
        return ThreatEngine()
    
    def _make_target(self, target_id, center=(100, 100)):
        bbox = (center[0] - 25, center[1] - 25, center[0] + 25, center[1] + 25)
        return Target(id=target_id, bbox=bbox, class_name="person", confidence=0.9)
    
    def _make_score(self, behavioral=0.0, sprint=False, erratic=False, loiter=False):
        return AnomalyScore(
            velocity_z_score=3.0 if sprint else 0.5,
            direction_z_score=2.5 if erratic else 0.3,
            is_sprinting=sprint,
            is_erratic=erratic,
            is_loitering=loiter,
            behavioral_score=behavioral
        )
    
    def test_low_threat(self, engine):
        """Score < 30 -> LOW."""
        target = self._make_target(1)
        score = self._make_score(behavioral=10.0)
        
        result = engine._compute_threat_level(target, score)
        
        assert result.threat_class == "LOW"
        assert result.threat_score == 10.0
        assert len(result.contributing_factors) == 0
    
    def test_medium_threat(self, engine):
        """Score 30-49 -> MEDIUM."""
        target = self._make_target(1)
        score = self._make_score(behavioral=40.0, sprint=True)
        
        result = engine._compute_threat_level(target, score)
        
        assert result.threat_class == "MEDIUM"
        assert "SPRINT_DETECTED" in result.contributing_factors
    
    def test_high_threat(self, engine):
        """Score 50-69 -> HIGH."""
        target = self._make_target(1)
        score = self._make_score(behavioral=60.0, sprint=True, erratic=True)
        
        result = engine._compute_threat_level(target, score)
        
        assert result.threat_class == "HIGH"
        assert "SPRINT_DETECTED" in result.contributing_factors
        assert "ERRATIC_MOVEMENT" in result.contributing_factors
    
    def test_critical_threat(self, engine):
        """Score >= 70 -> CRITICAL."""
        target = self._make_target(1)
        score = self._make_score(behavioral=80.0, sprint=True, erratic=True, loiter=True)
        
        result = engine._compute_threat_level(target, score)
        
        assert result.threat_class == "CRITICAL"
        assert "SPRINT_DETECTED" in result.contributing_factors
        assert "ERRATIC_MOVEMENT" in result.contributing_factors
        assert "LOITERING" in result.contributing_factors
    
    def test_boundary_30(self, engine):
        """Score exactly 30 -> MEDIUM."""
        target = self._make_target(1)
        score = self._make_score(behavioral=30.0)
        result = engine._compute_threat_level(target, score)
        assert result.threat_class == "MEDIUM"
    
    def test_boundary_50(self, engine):
        """Score exactly 50 -> HIGH."""
        target = self._make_target(1)
        score = self._make_score(behavioral=50.0)
        result = engine._compute_threat_level(target, score)
        assert result.threat_class == "HIGH"
    
    def test_boundary_70(self, engine):
        """Score exactly 70 -> CRITICAL."""
        target = self._make_target(1)
        score = self._make_score(behavioral=70.0)
        result = engine._compute_threat_level(target, score)
        assert result.threat_class == "CRITICAL"
    
    def test_zero_score(self, engine):
        """Score 0 -> LOW with no factors."""
        target = self._make_target(1)
        score = self._make_score(behavioral=0.0)
        result = engine._compute_threat_level(target, score)
        assert result.threat_class == "LOW"
        assert result.threat_score == 0.0
        assert len(result.contributing_factors) == 0
    
    def test_max_score(self, engine):
        """Score 100 -> CRITICAL."""
        target = self._make_target(1)
        score = self._make_score(behavioral=100.0, sprint=True, erratic=True, loiter=True)
        result = engine._compute_threat_level(target, score)
        assert result.threat_class == "CRITICAL"
    
    def test_assess_threats_multiple_targets(self, engine):
        """Test batch assessment of multiple targets."""
        t1 = self._make_target(1)
        t2 = self._make_target(2, center=(200, 200))
        
        scores = {
            1: self._make_score(behavioral=10.0),
            2: self._make_score(behavioral=80.0, sprint=True),
        }
        
        result = engine.assess_threats([t1, t2], scores)
        
        assert len(result) == 2
        assert result[1].threat_class == "LOW"
        assert result[2].threat_class == "CRITICAL"
    
    def test_assess_threats_missing_score(self, engine):
        """Test that targets without scores are skipped."""
        t1 = self._make_target(1)
        t2 = self._make_target(2, center=(200, 200))
        
        scores = {1: self._make_score(behavioral=10.0)}  # No score for target 2
        
        result = engine.assess_threats([t1, t2], scores)
        
        assert 1 in result
        assert 2 not in result