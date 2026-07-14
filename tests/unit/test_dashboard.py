"""
Defense COP v2.0 - Web Dashboard server WebSockets Unit Tests
"""
import asyncio
import base64
import json
from starlette.websockets import WebSocketDisconnect
from ui.dashboard import active_connections, broadcast_update, websocket_endpoint, queue_listener, update_queue

class MockWebSocket:
    """Mock WebSocket client for testing route handlers without httpx/TestClient."""
    def __init__(self):
        self.accepted = False
        self.sent_messages = []
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, message):
        self.sent_messages.append(message)

    async def receive_text(self):
        # Sleep for a longer time so the test can check the active connection list
        if not self.closed:
            await asyncio.sleep(1.0)
            return "keepalive"
        else:
            raise WebSocketDisconnect()


def test_websocket_lifecycle():
    """Verify WebSocket endpoint lifecycle: accept connection, register connection, unregister on close."""
    async def run():
        ws = MockWebSocket()
        active_connections.clear()
        
        # Start endpoint handler as a background task
        task = asyncio.create_task(websocket_endpoint(ws))
        await asyncio.sleep(0.02)
        
        # Verify accepted and registered
        assert ws.accepted is True
        assert ws in active_connections
        
        # Disconnect by setting closed flag and cancelling the sleep/task
        ws.closed = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Verify connection cleaned up
        assert ws not in active_connections

    asyncio.run(run())


def test_broadcast_and_queue_listener():
    """Verify uvicorn background queue listener broadcasts updates to active connections."""
    async def run():
        ws = MockWebSocket()
        active_connections.clear()
        active_connections.append(ws)
        
        # Flush update queue
        while not update_queue.empty():
            try:
                update_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Start queue listener in background
        listener_task = asyncio.create_task(queue_listener())
        await asyncio.sleep(0.01)
        
        # Broadcast visual frames & alerts
        test_frame = b"frame_payload"
        test_bev = b"bev_payload"
        test_alerts = [{
            "target_id": 99,
            "class_name": "person",
            "threat_score": 90.0,
            "threat_class": "CRITICAL",
            "contributing_factors": ["sprinting"]
        }]
        
        broadcast_update(test_frame, test_bev, test_alerts)
        await asyncio.sleep(0.15)  # Let queue listener process
        
        # Clean shutdown of listener
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass
            
        # Verify message sent to MockWebSocket
        assert len(ws.sent_messages) == 1
        data = json.loads(ws.sent_messages[0])
        
        assert base64.b64decode(data["frame"]) == test_frame
        assert base64.b64decode(data["bev"]) == test_bev
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["target_id"] == 99
        assert data["alerts"][0]["threat_class"] == "CRITICAL"

    asyncio.run(run())


def test_stream_pipeline_output():
    """Verify that stream_pipeline_output correctly formats and enqueues data."""
    import numpy as np
    from ui.dashboard import stream_pipeline_output, update_queue
    
    # Clear the queue first
    while not update_queue.empty():
        try:
            update_queue.get_nowait()
        except Exception:
            break
            
    class MockTarget:
        def __init__(self, target_id, class_name):
            self.id = target_id
            self.class_name = class_name
            
    class MockThreatLevel:
        def __init__(self, target_id, threat_score, threat_class, contributing_factors):
            self.target_id = target_id
            self.threat_score = threat_score
            self.threat_class = threat_class
            self.contributing_factors = contributing_factors
            
    class MockPipelineOutput:
        def __init__(self, threat_levels, bev_canvas):
            self.threat_levels = threat_levels
            self.bev_canvas = bev_canvas

    targets = [MockTarget(1, "person")]
    threat_levels = {1: MockThreatLevel(1, 85.5, "HIGH", ["sprint"])}
    bev_canvas = np.zeros((50, 50, 3), dtype=np.uint8)
    pipeline_output = MockPipelineOutput(threat_levels, bev_canvas)
    rendered_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    stream_pipeline_output(targets, pipeline_output, rendered_frame)
    
    assert not update_queue.empty()
    item = update_queue.get_nowait()
    assert "frame" in item
    assert "bev" in item
    assert "alerts" in item
    assert len(item["alerts"]) == 1
    assert item["alerts"][0]["target_id"] == 1
    assert item["alerts"][0]["class_name"] == "person"
    assert item["alerts"][0]["threat_score"] == 85.5
    assert item["alerts"][0]["threat_class"] == "HIGH"
    assert item["alerts"][0]["contributing_factors"] == ["sprint"]


def test_lifespan_handler():
    """Verify that entering/exiting the lifespan context manager works cleanly."""
    async def run():
        from ui.dashboard import app, lifespan
        
        # Enter and exit the lifespan context manager
        async with lifespan(app):
            # Give it a tiny bit of time to schedule/run startup
            await asyncio.sleep(0.01)
            
        # The background task is cancelled when context manager exits.
        # The test finishes successfully if no exceptions are raised.
        
    asyncio.run(run())
