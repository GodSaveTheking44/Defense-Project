"""
Defense COP v2.0 - Deterministic Time Provider
Ensures reproducible timestamps for telemetry and replay.
"""
from abc import ABC, abstractmethod
import time


class TimeProvider(ABC):
    """Abstract time provider for deterministic testing."""
    
    @abstractmethod
    def now(self) -> float:
        """Get current time in seconds since epoch."""
        pass
    
    @abstractmethod
    def sleep(self, duration: float) -> None:
        """Sleep for specified duration."""
        pass


class SystemTimeProvider(TimeProvider):
    """Production time provider using system clock."""
    
    def now(self) -> float:
        """Get current system time."""
        return time.time()
    
    def sleep(self, duration: float) -> None:
        """Sleep using system time."""
        time.sleep(duration)


class DeterministicTimeProvider(TimeProvider):
    """Deterministic time provider for testing and replay."""
    
    def __init__(self, start_time: float = 0.0, time_scale: float = 1.0):
        """
        Initialize deterministic time provider.
        
        Args:
            start_time: Initial time in seconds
            time_scale: Time scale factor (1.0 = real-time, 2.0 = 2x speed)
        """
        self._current_time = start_time
        self._time_scale = time_scale
    
    def now(self) -> float:
        """Get current deterministic time."""
        return self._current_time
    
    def sleep(self, duration: float) -> None:
        """Advance time without actual sleeping."""
        self._current_time += duration * self._time_scale
    
    def advance(self, delta: float) -> None:
        """Manually advance time by delta seconds."""
        self._current_time += delta
    
    def set_time(self, timestamp: float) -> None:
        """Set absolute time."""
        self._current_time = timestamp


class FrameTimeProvider(TimeProvider):
    """Time provider based on frame timestamps for video replay."""
    
    def __init__(self, fps: float = 30.0, start_time: float = 0.0):
        """
        Initialize frame-based time provider.
        
        Args:
            fps: Frames per second
            start_time: Initial timestamp
        """
        self._fps = fps
        self._frame_count = 0
        self._start_time = start_time
    
    def now(self) -> float:
        """Get time based on frame count."""
        return self._start_time + (self._frame_count / self._fps)
    
    def sleep(self, duration: float) -> None:
        """No-op for frame-based processing."""
        pass
    
    def next_frame(self) -> None:
        """Advance to next frame."""
        self._frame_count += 1
    
    def set_frame(self, frame_number: int) -> None:
        """Set absolute frame number."""
        self._frame_count = frame_number
