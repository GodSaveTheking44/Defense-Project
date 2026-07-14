"""
Defense COP v2.0 - Object Detection Layer
Support for real-time YOLOv8 object detection with Mock fallback.
"""
import logging
import cv2
import numpy as np
from typing import List, Tuple

logger = logging.getLogger("DefenseCOP.Detector")


class MockDetector:
    """
    Mock object detector for demonstration.
    Uses background subtraction for motion detection.
    """
    
    def __init__(self, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        logger.info("Initialized MockDetector (Background Subtraction)")
    
    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
        """Mock detection using background subtraction."""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (float(x), float(y), float(x + w), float(y + h))
            class_name = "person"
            confidence = min(0.95, 0.5 + (area / 10000))
            
            detections.append((bbox, class_name, confidence))
            
        return detections


class YOLODetector:
    """
    Real-time object detector using YOLOv8 via Ultralytics.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        acceleration: str = "none",
        device: str = "cpu"
    ):
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.acceleration = acceleration
        self.device = device
        self._fallback_detector = None
        self.model = None
        
        try:
            from ultralytics import YOLO
            import os
            
            base_path, ext = os.path.splitext(self.model_path)
            
            # Perform exports if acceleration is requested and not already exported
            if self.acceleration == "onnx" and ext == ".pt":
                onnx_path = base_path + ".onnx"
                if not os.path.exists(onnx_path):
                    logger.info(f"Exporting model {self.model_path} to ONNX format...")
                    export_model = YOLO(self.model_path)
                    export_model.export(format="onnx", half=True, dynamic=True)
                self.model_path = onnx_path
                
                # Verify ONNX Runtime execution providers
                try:
                    import onnxruntime as ort  # type: ignore
                    providers = ort.get_available_providers()
                    logger.info(f"ONNX Runtime available providers: {providers}")
                    if "CUDAExecutionProvider" in providers:
                        logger.info("ONNX Runtime: Hardware GPU acceleration (CUDA) is available.")
                    else:
                        logger.warning("ONNX Runtime: Only CPU provider is available. GPU acceleration may not be active.")
                except ImportError:
                    logger.warning("ONNX Runtime (onnxruntime) package not found. Standard CPU fallback will be used.")
                    
            elif self.acceleration == "tensorrt" and ext == ".pt":
                engine_path = base_path + ".engine"
                if not os.path.exists(engine_path):
                    logger.info(f"Exporting model {self.model_path} to TensorRT format...")
                    export_model = YOLO(self.model_path)
                    export_model.export(format="engine", half=True, device=self.device)
                self.model_path = engine_path
                logger.info("TensorRT model (.engine) ready for hardware-accelerated GPU inference.")
            
            logger.info(f"Loading YOLOv8 model: {self.model_path} on device {self.device}...")
            self.model = YOLO(self.model_path)
            logger.info("YOLOv8 model loaded successfully.")
        except ImportError:
            logger.warning("ultralytics package not found. Falling back to MockDetector.")
            self._fallback_detector = MockDetector(confidence_threshold)
        except (RuntimeError, OSError, FileNotFoundError) as e:
            logger.error(f"Failed to load YOLO model: {e}. Falling back to MockDetector.")
            self._fallback_detector = MockDetector(confidence_threshold)

    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
        """
        Detect targets using YOLOv8, or fallback if unavailable.
        """
        if self._fallback_detector is not None:
            return self._fallback_detector.detect(frame)
            
        if self.model is None:
            self._fallback_detector = MockDetector(self.confidence_threshold)
            return self._fallback_detector.detect(frame)
            
        try:
            results = self.model(frame, verbose=False, device=self.device)[0]
            detections = []
            
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                
                cls_idx = int(box.cls[0])
                class_name = results.names[cls_idx]
                
                # Get xyxy coordinates as floats
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                bbox = (x1, y1, x2, y2)
                
                detections.append((bbox, class_name, conf))
                
            return detections
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error(f"Error during YOLO inference: {e}. Falling back to MockDetector.")
            self._fallback_detector = MockDetector(self.confidence_threshold)
            return self._fallback_detector.detect(frame)
