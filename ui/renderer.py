"""
Defense COP v2.0 - UI Renderer
Real-time visualization with BEV overlay, bounding boxes, and alerts.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from engine.tracking import Target
from engine.anomaly_engine import AnomalyScore
from engine.threat_engine import ThreatLevel


class TacticalRenderer:
    """
    Production-grade UI renderer for Defense COP.
    Renders bounding boxes, IDs, scores, BEV mini-map, and alerts.
    """
    # Color scheme (BGR format)
    COLOR_NORMAL = (0, 255, 0)          # Green
    COLOR_ANOMALY = (0, 0, 255)         # Red
    COLOR_WARNING = (0, 165, 255)       # Orange
    COLOR_TRAJECTORY = (100, 200, 100)  # Light green
    
    # Threat level colors
    COLOR_CRITICAL = (0, 0, 255)        # Red
    COLOR_HIGH = (0, 100, 255)          # Orange
    COLOR_MEDIUM = (0, 255, 255)        # Yellow
    COLOR_LOW = (0, 255, 0)             # Green
    
    # UI Panel colors
    COLOR_PANEL_BG = (40, 40, 40)
    COLOR_PANEL_BORDER = (150, 150, 150)
    COLOR_TEXT_WHITE = (255, 255, 255)
    
    # Rendering thicknesses
    THICKNESS_ANOMALOUS_BOX = 3
    THICKNESS_NORMAL_BOX = 2
    THICKNESS_TRAJECTORY = 2
    THICKNESS_BORDER = 2
    
    # Reference resolution for scaling
    REF_WIDTH = 640
    REF_HEIGHT = 480
    
    # Reference layout dimensions
    PANEL_WIDTH_REF = 240
    PANEL_HEIGHT_REF = 120
    PANEL_Y_REF = 50
    PANEL_X_OFFSET_REF = 250
    
    def __init__(
        self,
        show_bev: bool = True,
        show_trajectories: bool = True,
        show_anomaly_score: bool = True,
        show_fps: bool = True,
        bev_position: Tuple[int, int] = (10, 10)
    ):
        """
        Initialize renderer.
        
        Args:
            show_bev: Display BEV mini-map overlay
            show_trajectories: Display target trajectories on main view
            show_anomaly_score: Display behavioral scores
            show_fps: Display FPS counter
            bev_position: Top-left corner for BEV overlay
        """
        self.show_bev = show_bev
        self.show_trajectories = show_trajectories
        self.show_anomaly_score = show_anomaly_score
        self.show_fps = show_fps
        self.bev_position = bev_position
    
    def _scale(self, value: int, frame: np.ndarray) -> int:
        """Scale a pixel value based on frame resolution relative to 640x480 reference."""
        scale_factor = frame.shape[1] / self.REF_WIDTH
        return max(1, int(value * scale_factor))
    
    def _scale_font(self, value: float, frame: np.ndarray) -> float:
        """Scale a font size based on frame resolution."""
        scale_factor = frame.shape[1] / self.REF_WIDTH
        return value * scale_factor

    def render_frame(
        self,
        frame: np.ndarray,
        targets: List[Target],
        anomaly_scores: Dict[int, AnomalyScore],
        threat_levels: Dict[int, ThreatLevel],
        bev_canvas: Optional[np.ndarray] = None,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Render complete UI overlay on frame.
        
        Args:
            frame: Input frame
            targets: List of tracked targets
            anomaly_scores: Anomaly scores per target
            threat_levels: Threat levels per target
            bev_canvas: BEV mini-map (optional)
            fps: Current FPS (optional)
        
        Returns:
            Rendered frame
        """
        output = frame.copy()
        
        # Draw trajectories first (background layer)
        if self.show_trajectories:
            for target in targets:
                self._draw_trajectory(output, target)
        
        # Draw bounding boxes and labels
        for target in targets:
            is_anomalous = (
                target.id in anomaly_scores and
                anomaly_scores[target.id].has_anomaly()
            )
            
            anomaly_score = anomaly_scores.get(target.id)
            threat_level = threat_levels.get(target.id)
            
            self._draw_target_bbox(
                output,
                target,
                is_anomalous,
                anomaly_score,
                threat_level
            )
        
        # Overlay BEV mini-map
        if self.show_bev and bev_canvas is not None:
            output = self._overlay_bev(output, bev_canvas)
        
        # Draw FPS counter
        if self.show_fps and fps is not None:
            self._draw_fps(output, fps)
        
        # Draw alert panel
        self._draw_alert_panel(output, targets, anomaly_scores, threat_levels)
        
        return output
    
    def _draw_target_bbox(
        self,
        frame: np.ndarray,
        target: Target,
        is_anomalous: bool,
        anomaly_score: Optional[AnomalyScore],
        threat_level: Optional[ThreatLevel]
    ) -> None:
        """Draw bounding box and labels for a target."""
        x1, y1, x2, y2 = map(int, target.bbox)
        
        # Choose color based on threat level
        if threat_level:
            if threat_level.threat_class == "CRITICAL":
                color = self.COLOR_CRITICAL
            elif threat_level.threat_class == "HIGH":
                color = self.COLOR_HIGH
            elif threat_level.threat_class == "MEDIUM":
                color = self.COLOR_MEDIUM
            else:
                color = self.COLOR_NORMAL
        else:
            color = self.COLOR_ANOMALY if is_anomalous else self.COLOR_NORMAL
        
        # Draw bounding box
        thickness = self.THICKNESS_ANOMALOUS_BOX if is_anomalous else self.THICKNESS_NORMAL_BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw target ID
        label = f"T{target.id}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - self._scale(10, frame)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._scale_font(0.6, frame),
            color,
            2
        )
        
        # Draw behavioral score
        if self.show_anomaly_score and anomaly_score:
            score_text = f"Score: {anomaly_score.behavioral_score:.1f}"
            cv2.putText(
                frame,
                score_text,
                (x1, y2 + self._scale(20, frame)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._scale_font(0.5, frame),
                color,
                1
            )
            
            # Draw anomaly flags
            flags = []
            if anomaly_score.is_sprinting:
                flags.append("SPRINT")
            if anomaly_score.is_erratic:
                flags.append("ERRATIC")
            if anomaly_score.is_loitering:
                flags.append("LOITER")
            
            if flags:
                flag_text = " | ".join(flags)
                cv2.putText(
                    frame,
                    flag_text,
                    (x1, y2 + self._scale(40, frame)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self._scale_font(0.4, frame),
                    self.COLOR_ANOMALY,
                    1
                )
    
    def _draw_trajectory(self, frame: np.ndarray, target: Target) -> None:
        """Draw trajectory trail for a target."""
        if len(target.trajectory) < 2:
            return
        
        points = list(target.trajectory)
        pts = np.array([(int(x), int(y)) for x, y in points], dtype=np.int32)
        
        cv2.polylines(
            frame,
            [pts],
            isClosed=False,
            color=self.COLOR_TRAJECTORY,
            thickness=self.THICKNESS_TRAJECTORY
        )
    
    def _overlay_bev(
        self,
        frame: np.ndarray,
        bev_canvas: np.ndarray
    ) -> np.ndarray:
        """Overlay BEV mini-map on frame."""
        x, y = self.bev_position
        h, w = bev_canvas.shape[:2]
        
        # Ensure overlay fits
        frame_h, frame_w = frame.shape[:2]
        if y + h > frame_h or x + w > frame_w:
            return frame
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        overlay[y:y+h, x:x+w] = bev_canvas
        
        # Blend
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw border
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            self.COLOR_TEXT_WHITE,
            self.THICKNESS_BORDER
        )
        
        return frame
    
    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter."""
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            text,
            (frame.shape[1] - self._scale(150, frame), self._scale(30, frame)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._scale_font(0.7, frame),
            self.COLOR_NORMAL,
            2
        )
    
    def _draw_alert_panel(
        self,
        frame: np.ndarray,
        targets: List[Target],
        anomaly_scores: Dict[int, AnomalyScore],
        threat_levels: Dict[int, ThreatLevel]
    ) -> None:
        """Draw alert summary panel."""
        # Count anomalies
        sprint_count = sum(
            1 for tid, score in anomaly_scores.items()
            if score.is_sprinting
        )
        erratic_count = sum(
            1 for tid, score in anomaly_scores.items()
            if score.is_erratic
        )
        loiter_count = sum(
            1 for tid, score in anomaly_scores.items()
            if score.is_loitering
        )
        
        # Count critical threats
        critical_count = sum(
            1 for tid, level in threat_levels.items()
            if level.threat_class == "CRITICAL"
        )
        
        # Draw panel background
        panel_x = frame.shape[1] - self._scale(self.PANEL_X_OFFSET_REF, frame)
        panel_y = self._scale(self.PANEL_Y_REF, frame)
        panel_width = self._scale(self.PANEL_WIDTH_REF, frame)
        panel_height = self._scale(self.PANEL_HEIGHT_REF, frame)
        
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            self.COLOR_PANEL_BG,
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            self.COLOR_PANEL_BORDER,
            self.THICKNESS_BORDER
        )
        
        # Draw alert counts
        y_offset = panel_y + self._scale(25, frame)
        cv2.putText(
            frame,
            f"Targets: {len(targets)}",
            (panel_x + self._scale(10, frame), y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._scale_font(0.5, frame),
            self.COLOR_TEXT_WHITE,
            1
        )
        
        y_offset += self._scale(25, frame)
        color = self.COLOR_CRITICAL if critical_count > 0 else self.COLOR_TEXT_WHITE
        cv2.putText(
            frame,
            f"Critical: {critical_count}",
            (panel_x + self._scale(10, frame), y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._scale_font(0.5, frame),
            color,
            1
        )
        
        y_offset += self._scale(20, frame)
        cv2.putText(
            frame,
            f"Sprint: {sprint_count} | Erratic: {erratic_count}",
            (panel_x + self._scale(10, frame), y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._scale_font(0.4, frame),
            self.COLOR_TEXT_WHITE,
            1
        )
        
        y_offset += self._scale(20, frame)
        cv2.putText(
            frame,
            f"Loitering: {loiter_count}",
            (panel_x + self._scale(10, frame), y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._scale_font(0.4, frame),
            self.COLOR_TEXT_WHITE,
            1
        )
