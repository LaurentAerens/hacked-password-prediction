"""
Thread-safe control signal for trainer pause/stop/resume operations.

This module provides a non-blocking, thread-safe state machine for controlling
long-running training sessions. State transitions are protected by a lock to
ensure atomicity under concurrent access.

State machine:
    RUNNING ──request_pause()──→ PAUSE_REQUESTED
                                        │
                            should_pause()=True
                                        ↓
                                    PAUSED ──resume()──→ RUNNING
                                        │
                                        └─→ request_stop() ──→ STOP_REQUESTED
                                                                      │
                                                        should_stop()=True
                                                                      ↓
                                                                  STOPPED

Polling semantics (< 1ms return time):
    - should_pause(): Returns True if PAUSE_REQUESTED, transitions to PAUSED, then returns False on retry
    - should_stop(): Returns True if STOP_REQUESTED, transitions to STOPPED, then returns False on retry
    - get_state(): Returns current state string
    - All methods are non-blocking
"""

import threading
import uuid


class ControlSignal:
    """Thread-safe control signal for pause/stop/resume operations.

    Provides non-blocking polling interface for trainer to check for user control requests.
    All state transitions are atomic and protected by a lock.

    Attributes:
        run_id (str): Unique identifier for this training run.
    """

    # Valid states in the state machine
    RUNNING = "RUNNING"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    PAUSED = "PAUSED"
    STOP_REQUESTED = "STOP_REQUESTED"
    STOPPED = "STOPPED"

    def __init__(self, run_id: str = None):
        """Initialize ControlSignal with optional run_id.

        Args:
            run_id (str, optional): Unique identifier for this training run.
                If None, generates a UUID-based run_id. Defaults to None.
        """
        self.run_id = run_id or str(uuid.uuid4())
        self._state = self.RUNNING
        self._lock = threading.Lock()

    def request_pause(self) -> None:
        """Request a pause (graceful checkpoint).

        Transitions from RUNNING → PAUSE_REQUESTED (other states remain unchanged).
        Non-blocking.
        """
        with self._lock:
            if self._state == self.RUNNING:
                self._state = self.PAUSE_REQUESTED

    def request_stop(self) -> None:
        """Request a stop (hard stop).

        Can be called from any state. Transitions to STOP_REQUESTED.
        Non-blocking.
        """
        with self._lock:
            if self._state != self.STOPPED:
                self._state = self.STOP_REQUESTED

    def resume(self) -> None:
        """Resume from pause.

        Transitions from PAUSED → RUNNING or PAUSE_REQUESTED → RUNNING.
        Non-blocking.
        """
        with self._lock:
            if self._state in (self.PAUSED, self.PAUSE_REQUESTED):
                self._state = self.RUNNING

    def should_pause(self) -> bool:
        """Poll for pause request.

        Returns True exactly once if in PAUSE_REQUESTED state, transitions to PAUSED.
        Returns False for all other states.
        Non-blocking (< 1ms return time).

        Returns:
            bool: True if pause was requested and this is the transition point.
                  False otherwise.
        """
        with self._lock:
            if self._state == self.PAUSE_REQUESTED:
                self._state = self.PAUSED
                return True
            return False

    def should_stop(self) -> bool:
        """Poll for stop request.

        Returns True exactly once if in STOP_REQUESTED state, transitions to STOPPED.
        Returns False for all other states.
        Non-blocking (< 1ms return time).

        Returns:
            bool: True if stop was requested and this is the transition point.
                  False otherwise.
        """
        with self._lock:
            if self._state == self.STOP_REQUESTED:
                self._state = self.STOPPED
                return True
            return False

    def get_state(self) -> str:
        """Get current state.

        Non-blocking (< 1ms return time).

        Returns:
            str: Current state (one of RUNNING, PAUSE_REQUESTED, PAUSED, STOP_REQUESTED, STOPPED).
        """
        with self._lock:
            return self._state
