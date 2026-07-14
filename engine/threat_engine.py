"""
Defense COP v2.0 - Threat Assessment Engine
Placeholder for threat scoring and classification logic.
"""
from dataclasses import dataclass
from typing import List, Dict
from engine.tracking import Target
from engine.anomaly_engine import AnomalyScore


@dataclass
class ThreatLevel:
    """Threat assessment for a target."""
    target_id: int
    threat_score: float  # 0-100
    threat_class: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    contributing_factors: List[str]


class ThreatEngine:
    """
    Threat assessment engine.
    Combines anomaly scores with other factors for threat classification.
    """
    
    def __init__(self):
        """Initialize threat engine."""
        pass
    
    def assess_threats(
        self,
        targets: List[Target],
        anomaly_scores: Dict[int, AnomalyScore]
    ) -> Dict[int, ThreatLevel]:
        """
        Assess threat level for all targets.
        
        Args:
            targets: List of tracked targets
            anomaly_scores: Anomaly scores per target
        
        Returns:
            Dictionary mapping target_id to ThreatLevel
        """
        threat_levels = {}
        
        for target in targets:
            if target.id not in anomaly_scores:
                continue
            
            anomaly_score = anomaly_scores[target.id]
            threat_level = self._compute_threat_level(target, anomaly_score)
            threat_levels[target.id] = threat_level
        
        return threat_levels
    
    def _compute_threat_level(
        self,
        target: Target,
        anomaly_score: AnomalyScore
    ) -> ThreatLevel:
        """Compute threat level for a single target."""
        # Start with behavioral score
        threat_score = anomaly_score.behavioral_score
        contributing_factors = []
        
        # Add factors
        if anomaly_score.is_sprinting:
            contributing_factors.append("SPRINT_DETECTED")
        
        if anomaly_score.is_erratic:
            contributing_factors.append("ERRATIC_MOVEMENT")
        
        if anomaly_score.is_loitering:
            contributing_factors.append("LOITERING")
        
        # Classify threat level
        if threat_score >= 70:
            threat_class = "CRITICAL"
        elif threat_score >= 50:
            threat_class = "HIGH"
        elif threat_score >= 30:
            threat_class = "MEDIUM"
        else:
            threat_class = "LOW"
        
        return ThreatLevel(
            target_id=target.id,
            threat_score=threat_score,
            threat_class=threat_class,
            contributing_factors=contributing_factors
        )
