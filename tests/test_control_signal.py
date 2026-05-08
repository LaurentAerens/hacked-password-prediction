"""
Unit tests for ControlSignal state machine.

Tests cover:
- State transitions (RUNNING → PAUSE_REQUESTED → PAUSED, STOP_REQUESTED → STOPPED)
- Thread safety (concurrent access)
- Non-blocking polling semantics
- Correct return values
"""

import pytest
import threading
import time
import sys
import os
from pathlib import Path

# Add ai-resources to path
ai_resources_path = str(Path(__file__).parent.parent / "ai-resources")
sys.path.insert(0, ai_resources_path)

from shared_lib.control_signal import ControlSignal


class TestControlSignalInitialization:
    """Test ControlSignal initialization and initial state."""

    def test_init_with_run_id(self):
        """ControlSignal should initialize with run_id and RUNNING state."""
        cs = ControlSignal(run_id="test-run-001")
        assert cs.run_id == "test-run-001"
        assert cs.get_state() == "RUNNING"

    def test_init_without_run_id(self):
        """ControlSignal should generate run_id if not provided."""
        cs = ControlSignal()
        assert cs.run_id is not None
        assert cs.get_state() == "RUNNING"


class TestControlSignalStateTransitions:
    """Test valid state transitions."""

    def test_request_pause_from_running(self):
        """request_pause() should transition RUNNING → PAUSE_REQUESTED."""
        cs = ControlSignal("test-run-001")
        cs.request_pause()
        assert cs.get_state() == "PAUSE_REQUESTED"

    def test_should_pause_transitions_to_paused(self):
        """should_pause() should return True and transition PAUSE_REQUESTED → PAUSED."""
        cs = ControlSignal("test-run-001")
        cs.request_pause()
        result = cs.should_pause()
        assert result is True
        assert cs.get_state() == "PAUSED"

    def test_resume_from_paused(self):
        """resume() should transition PAUSED → RUNNING."""
        cs = ControlSignal("test-run-001")
        cs.request_pause()
        cs.should_pause()  # Transition to PAUSED
        cs.resume()
        assert cs.get_state() == "RUNNING"

    def test_should_pause_returns_false_when_running(self):
        """should_pause() should return False when RUNNING."""
        cs = ControlSignal("test-run-001")
        result = cs.should_pause()
        assert result is False
        assert cs.get_state() == "RUNNING"

    def test_should_pause_returns_false_when_paused(self):
        """should_pause() should return False when PAUSED (already transitioned)."""
        cs = ControlSignal("test-run-001")
        cs.request_pause()
        cs.should_pause()  # Transition to PAUSED
        result = cs.should_pause()  # Check again
        assert result is False
        assert cs.get_state() == "PAUSED"

    def test_request_stop_from_running(self):
        """request_stop() should transition RUNNING → STOP_REQUESTED."""
        cs = ControlSignal("test-run-001")
        cs.request_stop()
        assert cs.get_state() == "STOP_REQUESTED"

    def test_request_stop_from_paused(self):
        """request_stop() should transition PAUSED → STOP_REQUESTED."""
        cs = ControlSignal("test-run-001")
        cs.request_pause()
        cs.should_pause()  # Transition to PAUSED
        cs.request_stop()
        assert cs.get_state() == "STOP_REQUESTED"

    def test_should_stop_transitions_to_stopped(self):
        """should_stop() should return True and transition STOP_REQUESTED → STOPPED."""
        cs = ControlSignal("test-run-001")
        cs.request_stop()
        result = cs.should_stop()
        assert result is True
        assert cs.get_state() == "STOPPED"

    def test_should_stop_returns_false_when_running(self):
        """should_stop() should return False when RUNNING."""
        cs = ControlSignal("test-run-001")
        result = cs.should_stop()
        assert result is False
        assert cs.get_state() == "RUNNING"

    def test_should_stop_returns_false_when_stopped(self):
        """should_stop() should return False when STOPPED (already transitioned)."""
        cs = ControlSignal("test-run-001")
        cs.request_stop()
        cs.should_stop()  # Transition to STOPPED
        result = cs.should_stop()  # Check again
        assert result is False
        assert cs.get_state() == "STOPPED"


class TestControlSignalThreadSafety:
    """Test thread-safe concurrent access."""

    def test_concurrent_state_reads(self):
        """Multiple threads reading state concurrently should not raise errors."""
        cs = ControlSignal("test-run-001")
        results = []

        def read_state():
            for _ in range(100):
                state = cs.get_state()
                results.append(state)

        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 500
        assert all(s == "RUNNING" for s in results)

    def test_concurrent_request_pause_and_read(self):
        """Concurrent pause requests and reads should be safe."""
        cs = ControlSignal("test-run-001")
        pause_count = [0]
        read_count = [0]

        def request_pause_multiple():
            for _ in range(50):
                cs.request_pause()
                pause_count[0] += 1

        def read_state_multiple():
            for _ in range(100):
                cs.get_state()
                read_count[0] += 1

        t1 = threading.Thread(target=request_pause_multiple)
        t2 = threading.Thread(target=read_state_multiple)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert pause_count[0] == 50
        assert read_count[0] == 100
        # State should be PAUSE_REQUESTED after pause requests
        assert cs.get_state() in ("PAUSE_REQUESTED", "PAUSED")

    def test_no_deadlock_concurrent_operations(self):
        """Concurrent pause, stop, and should_pause calls should not deadlock."""
        cs = ControlSignal("test-run-001")
        completed = []

        def operation_1():
            cs.request_pause()
            completed.append("pause_requested")

        def operation_2():
            cs.should_pause()
            completed.append("should_pause")

        def operation_3():
            cs.get_state()
            completed.append("get_state")

        threads = [
            threading.Thread(target=operation_1),
            threading.Thread(target=operation_2),
            threading.Thread(target=operation_3),
        ]
        for t in threads:
            t.start()

        # Wait with timeout to detect deadlock
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive(), "Thread deadlocked"

        assert len(completed) == 3

    def test_should_pause_idempotent_concurrent(self):
        """should_pause() should be idempotent under concurrent calls."""
        cs = ControlSignal("test-run-001")
        cs.request_pause()

        results = []

        def check_pause():
            result = cs.should_pause()
            results.append(result)

        threads = [threading.Thread(target=check_pause) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # First thread to transition should return True, rest False
        assert results.count(True) == 1
        assert results.count(False) == 4


class TestControlSignalNonBlockingBehavior:
    """Test non-blocking polling semantics."""

    def test_should_pause_returns_quickly(self):
        """should_pause() should return in < 1ms."""
        cs = ControlSignal("test-run-001")
        cs.request_pause()

        start = time.perf_counter()
        cs.should_pause()
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        assert elapsed < 1.0, f"should_pause() took {elapsed}ms, expected < 1ms"

    def test_should_stop_returns_quickly(self):
        """should_stop() should return in < 1ms."""
        cs = ControlSignal("test-run-001")
        cs.request_stop()

        start = time.perf_counter()
        cs.should_stop()
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        assert elapsed < 1.0, f"should_stop() took {elapsed}ms, expected < 1ms"

    def test_get_state_returns_quickly(self):
        """get_state() should return in < 1ms."""
        cs = ControlSignal("test-run-001")

        start = time.perf_counter()
        cs.get_state()
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        assert elapsed < 1.0, f"get_state() took {elapsed}ms, expected < 1ms"


class TestControlSignalEdgeCases:
    """Test edge cases and error conditions."""

    def test_pause_then_stop(self):
        """Pause followed by stop should work correctly."""
        cs = ControlSignal("test-run-001")
        cs.request_pause()
        cs.should_pause()
        assert cs.get_state() == "PAUSED"

        cs.request_stop()
        assert cs.get_state() == "STOP_REQUESTED"

        result = cs.should_stop()
        assert result is True
        assert cs.get_state() == "STOPPED"

    def test_stop_from_running(self):
        """Stop from RUNNING should transition directly to STOPPED."""
        cs = ControlSignal("test-run-001")
        cs.request_stop()
        assert cs.get_state() == "STOP_REQUESTED"

        result = cs.should_stop()
        assert result is True
        assert cs.get_state() == "STOPPED"

    def test_resume_idempotent(self):
        """Multiple resume calls on RUNNING should be safe."""
        cs = ControlSignal("test-run-001")
        cs.request_pause()
        cs.should_pause()
        cs.resume()
        assert cs.get_state() == "RUNNING"

        # Resume again from RUNNING
        cs.resume()
        assert cs.get_state() == "RUNNING"

    def test_request_pause_on_paused_is_noop(self):
        """Requesting pause when already PAUSED should be a no-op."""
        cs = ControlSignal("test-run-001")
        cs.request_pause()
        cs.should_pause()
        assert cs.get_state() == "PAUSED"

        # Request pause again
        cs.request_pause()
        assert cs.get_state() == "PAUSED"

    def test_full_lifecycle(self):
        """Test full lifecycle: start → pause → resume → stop."""
        cs = ControlSignal("test-run-001")

        # Initial state
        assert cs.get_state() == "RUNNING"
        assert cs.should_pause() is False
        assert cs.should_stop() is False

        # Request pause
        cs.request_pause()
        assert cs.get_state() == "PAUSE_REQUESTED"
        assert cs.should_pause() is True
        assert cs.get_state() == "PAUSED"

        # Resume
        cs.resume()
        assert cs.get_state() == "RUNNING"

        # Request stop
        cs.request_stop()
        assert cs.get_state() == "STOP_REQUESTED"
        assert cs.should_stop() is True
        assert cs.get_state() == "STOPPED"
