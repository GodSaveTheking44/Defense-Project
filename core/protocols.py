"""
Defense COP v2.0 - Protocol Definitions
Type-safe interfaces for dependency injection.
"""
from typing import List, Tuple, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class DetectorProtocol(Protocol):
    """Protocol for object detectors.
    
    Any detector (YOLO, mock, etc.) must implement this interface.
    """
    
    def detect(
        self,
        frame: np.ndarray
    ) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
        """Detect objects in a frame.
        
        Args:
            frame: Input image (H x W x 3, BGR format)
        
        Returns:
            List of (bbox, class_name, confidence) tuples where
            bbox is (x1, y1, x2, y2) in pixel coordinates
        """
        ...
