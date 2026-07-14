"""
Defense COP v2.0 - Detector Unit Tests
Unit tests for MockDetector and YOLODetector.
"""
import sys
import numpy as np
from unittest.mock import MagicMock, patch
from engine.detector import MockDetector, YOLODetector


class TestMockDetector:
    """Test MockDetector logic."""

    def test_mock_detector_no_motion(self):
        """Test that a sequence of identical frames produces no detections."""
        detector = MockDetector(confidence_threshold=0.25)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Feed identical frames (no motion)
        for _ in range(5):
            detections = detector.detect(frame)
        
        assert len(detections) == 0

    def test_mock_detector_motion(self):
        """Test that motion (moving rectangle) produces detections."""
        detector = MockDetector(confidence_threshold=0.25)
        
        # Frame 1: Background (black)
        frame1 = np.zeros((400, 400, 3), dtype=np.uint8)
        for _ in range(10):
            detector.detect(frame1)
        
        # Frame 2: Moving foreground (white square in center)
        frame2 = np.zeros((400, 400, 3), dtype=np.uint8)
        frame2[150:250, 150:250] = 255
        
        # Run once to catch the motion event
        detections = detector.detect(frame2)
            
        # Should detect a target due to the white rectangle movement
        assert len(detections) > 0
        bbox, class_name, conf = detections[0]
        assert class_name == "person"
        assert conf > 0.25
        # Coordinates should be close to [150, 150, 250, 250]
        assert 140 <= bbox[0] <= 160
        assert 140 <= bbox[1] <= 160
        assert 240 <= bbox[2] <= 260
        assert 240 <= bbox[3] <= 260


class TestYOLODetector:
    """Test YOLODetector logic."""

    def test_yolo_detector_fallback_on_import_error(self):
        """Test that YOLODetector falls back to MockDetector when ultralytics is missing."""
        # Hide ultralytics if it is installed
        with patch.dict(sys.modules, {"ultralytics": None}):
            detector = YOLODetector(model_path="nonexistent.pt")
            
            # Assert fallback detector was instantiated
            assert detector._fallback_detector is not None
            assert isinstance(detector._fallback_detector, MockDetector)
            
            # Run detection (should run MockDetector logic)
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            detector.detect(frame)  # First frame to initialize background model
            detections = detector.detect(frame)  # Second frame (no motion)
            assert len(detections) == 0

    def test_yolo_detector_mocked_inference(self):
        """Test YOLO extraction of bounding boxes and class names with mocked YOLO."""
        mock_yolo_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_yolo_class.return_value = mock_model_instance
        
        # Mock results returned by YOLO model
        mock_results = MagicMock()
        
        # Mock box detections
        mock_box1 = MagicMock()
        mock_box1.conf = [0.85]
        mock_box1.cls = [0.0]  # person
        mock_box1.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 50.0, 100.0])]
        
        mock_box2 = MagicMock()
        mock_box2.conf = [0.15]  # below threshold (0.25)
        mock_box2.cls = [2.0]  # car
        mock_box2.xyxy = [MagicMock(tolist=lambda: [100.0, 100.0, 150.0, 150.0])]
        
        mock_results.boxes = [mock_box1, mock_box2]
        mock_results.names = {0: "person", 2: "car"}
        mock_model_instance.return_value = [mock_results]
        
        with patch.dict(sys.modules, {"ultralytics": MagicMock()}):
            with patch("ultralytics.YOLO", mock_yolo_class, create=True):
                detector = YOLODetector(model_path="yolov8n.pt", confidence_threshold=0.25)
                
                assert detector.model is not None
                assert detector._fallback_detector is None
                
                frame = np.zeros((200, 200, 3), dtype=np.uint8)
                detections = detector.detect(frame)
                
                # Should filter out low confidence box2, returning only box1
                assert len(detections) == 1
                bbox, class_name, conf = detections[0]
                
                assert bbox == (10.0, 20.0, 50.0, 100.0)
                assert class_name == "person"
                assert conf == 0.85
