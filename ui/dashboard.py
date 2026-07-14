"""
Defense COP v2.0 - Web Dashboard Server
High-performance WebSocket streaming server for remote tactical command.
"""
import queue
import asyncio
import base64
import json
import logging
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

logger = logging.getLogger("DefenseCOP.Dashboard")

# Active WebSocket connections and update queue
active_connections: List[WebSocket] = []
update_queue: queue.Queue = queue.Queue(maxsize=5)

# Load template dynamically
TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"
try:
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        HTML_TEMPLATE = f.read()
except Exception as e:
    logger.error(f"Failed to load dashboard HTML template from {TEMPLATE_PATH}: {e}")
    HTML_TEMPLATE = "<h1>Error: Dashboard HTML template could not be loaded.</h1>"


# Async task to poll queue and broadcast to all WebSockets
async def queue_listener():
    logger.info("Starting WebSocket queue listener loop...")
    while True:
        try:
            # Check if there are active connections before doing anything
            if active_connections:
                # Retrieve update from queue without blocking the asyncio loop indefinitely
                update = await asyncio.to_thread(update_queue.get, timeout=0.1)
                message = json.dumps(update)
                # Broadcast
                await asyncio.gather(*[
                    connection.send_text(message) for connection in active_connections
                ], return_exceptions=True)
            else:
                # If no connections, just clear/drain the queue periodically
                if not update_queue.empty():
                    try:
                        update_queue.get_nowait()
                    except queue.Empty:
                        pass
                await asyncio.sleep(0.1)
        except queue.Empty:
            # Yield control if queue is empty
            await asyncio.sleep(0.01)
        except asyncio.TimeoutError:
            # Timeout is expected if queue is empty, just yield
            await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in queue listener: {e}", exc_info=True)
            await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the queue listener loop as a background task
    task = asyncio.create_task(queue_listener())
    yield
    # Shutdown: Clean up background task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Defense COP v2.0 - Tactical Control Panel", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return HTML_TEMPLATE


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket client connected. Active connections: {len(active_connections)}")
    try:
        while True:
            # Keep connection open and listen for disconnection
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally.")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
            logger.info(f"WebSocket connection cleaned up. Active connections: {len(active_connections)}")


# Thread-safe helper to push updates to the broadcast queue
def broadcast_update(frame_jpeg: bytes, bev_jpeg: Optional[bytes], alerts: list):
    frame_b64 = base64.b64encode(frame_jpeg).decode("utf-8")
    bev_b64 = base64.b64encode(bev_jpeg).decode("utf-8") if bev_jpeg else ""
    
    update = {
        "frame": frame_b64,
        "bev": bev_b64,
        "alerts": alerts
    }
    
    try:
        # If queue is full, clear the oldest to ensure real-time visual frames
        if update_queue.full():
            try:
                update_queue.get_nowait()
            except queue.Empty:
                pass
        update_queue.put_nowait(update)
    except Exception as e:
        logger.error(f"Failed to push update to broadcast queue: {e}")


def stream_pipeline_output(targets, pipeline_output, rendered_frame) -> None:
    """
    Format and stream pipeline output (rendered frames, BEV canvas, threat alerts)
    to the active dashboard WebSocket connections.
    """
    try:
        import cv2
        _, frame_jpeg = cv2.imencode('.jpg', rendered_frame)
        
        bev_jpeg = None
        if pipeline_output.bev_canvas is not None:
            _, bev_jpeg = cv2.imencode('.jpg', pipeline_output.bev_canvas)
        
        alerts = []
        target_map = {t.id: t.class_name for t in targets}
        for tid, threat in pipeline_output.threat_levels.items():
            alerts.append({
                "target_id": threat.target_id,
                "class_name": target_map.get(threat.target_id, "unknown"),
                "threat_score": threat.threat_score,
                "threat_class": threat.threat_class,
                "contributing_factors": threat.contributing_factors
            })
        
        broadcast_update(
            frame_jpeg.tobytes(),
            bev_jpeg.tobytes() if bev_jpeg is not None else None,
            alerts
        )
    except Exception as e:
        logger.error(f"Failed to stream pipeline output to dashboard: {e}", exc_info=True)


def start_dashboard_server(host: str = "127.0.0.1", port: int = 8000):
    """Run uvicorn server in a blocking call (designed for background threads)."""
    # Suppress verbose uvicorn logs to not pollute tactical output
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.error"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"
    
    uvicorn.run(app, host=host, port=port, log_config=log_config)
