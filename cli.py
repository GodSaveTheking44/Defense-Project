"""
Defense COP v2.0 - Command Line Interface
Production CLI with webcam/video support and feature toggles.
"""
import argparse
import sys
import cv2
import time
from pathlib import Path

import threading
from core.config import COPConfig
from core.telemetry import TelemetryLogger
from core.time_provider import SystemTimeProvider, FrameTimeProvider
from engine.engine_core import COPEngine
from engine.detector import YOLODetector
from ui.renderer import TacticalRenderer

try:
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    from ui.dashboard import start_dashboard_server, stream_pipeline_output
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Defense COP v2.0 - Common Operational Picture System"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="Video source: 0 for webcam, or path to video file"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no UI display)"
    )
    
    parser.add_argument(
        "--no-bev",
        action="store_true",
        help="Disable BEV mapping"
    )
    
    parser.add_argument(
        "--telemetry-dir",
        type=str,
        default="telemetry",
        help="Directory for telemetry output"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name or path to weights"
    )
    
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run web-based dashboard and stream over WebSockets"
    )
    
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Enable Cursor-on-Target (CoT) XML UDP multicast streaming to ATAK"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS for video processing"
    )
    
    return parser.parse_args()


def open_video_source(source: str):
    """Open video capture source."""
    # Try as webcam index
    if source.isdigit():
        video_capture = cv2.VideoCapture(int(source))
    else:
        # Try as file path
        video_capture = cv2.VideoCapture(source)
    
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    
    return video_capture


def main():
    """Main entry point for Defense COP v2.0."""
    parsed_args = parse_args()
    
    # Load or create configuration
    if parsed_args.config:
        config = COPConfig.from_yaml(Path(parsed_args.config))
    else:
        config = COPConfig()
    
    # Override config with CLI args
    config.headless = parsed_args.headless
    config.bev.enabled = not parsed_args.no_bev
    config.video.source = parsed_args.video
    config.video.fps = parsed_args.fps
    config.telemetry.output_dir = parsed_args.telemetry_dir
    config.detector.model = parsed_args.model
    config.telemetry.cot_enabled = parsed_args.cot or config.telemetry.cot_enabled
    
    print("=" * 60)
    print("Defense COP v2.0 - Common Operational Picture System")
    print("=" * 60)
    print(f"Video Source: {config.video.source}")
    print(f"Detector Model: {config.detector.model}")
    print(f"BEV Enabled: {config.bev.enabled}")
    print(f"Headless Mode: {config.headless}")
    print(f"Telemetry Dir: {config.telemetry.output_dir}")
    print("=" * 60)
    
    # Open video source
    try:
        video_capture = open_video_source(config.video.source)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = config.video.fps
    
    # Initialize time provider
    if config.video.source.isdigit():
        # Live webcam - use system time
        time_provider = SystemTimeProvider()
    else:
        # Video file - use frame-based time
        time_provider = FrameTimeProvider(fps=fps)
    
    # Initialize telemetry
    telemetry_path = Path(config.telemetry.output_dir)
    telemetry = TelemetryLogger(
        time_provider=time_provider,
        output_dir=telemetry_path,
        enabled=config.telemetry.enabled,
        cot_enabled=config.telemetry.cot_enabled,
        cot_host=config.telemetry.cot_host,
        cot_port=config.telemetry.cot_port
    )
    
    # Start web dashboard if requested
    if parsed_args.web:
        if not DASHBOARD_AVAILABLE:
            print("ERROR: fastapi and uvicorn are required for the web dashboard.")
            print("Please run: pip install fastapi uvicorn")
            return 1
        print("Starting WebSocket dashboard server on http://127.0.0.1:8000 ...")
        threading.Thread(
            target=start_dashboard_server,
            args=("127.0.0.1", 8000),
            daemon=True
        ).start()

    # Initialize detector (YOLOv8 with Mock fallback)
    detector = YOLODetector(
        model_path=config.detector.model,
        confidence_threshold=config.tracking.confidence_threshold,
        acceleration=config.detector.acceleration,
        device=config.detector.device
    )
    
    # Initialize COP engine
    engine = COPEngine(
        config=config,
        time_provider=time_provider,
        telemetry_logger=telemetry,
        detector_model=detector
    )
    
    # Initialize renderer
    renderer = TacticalRenderer(
        show_bev=config.ui.show_bev,
        show_trajectories=config.ui.show_trajectories,
        show_anomaly_score=config.ui.show_anomaly_score,
        show_fps=config.ui.show_fps,
        bev_position=config.ui.bev_position
    )
    
    print("\nStarting video processing...")
    print("Press 'Q' to quit\n")
    
    # Main processing loop
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame_read_success, frame = video_capture.read()
            
            if not frame_read_success:
                print("End of video stream")
                break
            
            # Run detection
            detections = detector.detect(frame)
            
            # Process through COP engine
            targets, pipeline_output = engine.process_frame(frame, detections)
            
            # Render UI
            if not config.headless or parsed_args.web:
                rendered_frame = renderer.render_frame(
                    frame,
                    targets,
                    pipeline_output.anomaly_scores,
                    pipeline_output.threat_levels,
                    pipeline_output.bev_canvas,
                    fps=(frame_count + 1) / (time.time() - start_time + 1e-6)
                )
                
                if not config.headless:
                    cv2.imshow("Defense COP v2.0", rendered_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("\nShutdown requested by user")
                        break
                else:
                    # In headless mode, wait to maintain target FPS
                    target_delay = 1.0 / fps
                    elapsed = time.time() - start_time
                    expected_time = (frame_count + 1) * target_delay
                    sleep_time = expected_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Stream to web dashboard
                if parsed_args.web and DASHBOARD_AVAILABLE:
                    stream_pipeline_output(targets, pipeline_output, rendered_frame)
            
            frame_count += 1
            
            # Print status every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(
                    f"Frame {frame_count} | "
                    f"Targets: {len(targets)} | "
                    f"FPS: {current_fps:.1f}"
                )
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean shutdown
        print("\nShutting down...")
        engine.shutdown()
        video_capture.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessed {frame_count} frames")
        print(f"Telemetry saved to: {telemetry_path}")
        print("\nDefense COP v2.0 shutdown complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
