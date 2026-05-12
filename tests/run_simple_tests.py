"""
Simple test runner for telemetry emitter without pytest.
"""

import os
from pathlib import Path

from harp.shared_lib.telemetry_emitter import TelemetryEmitter, create_emitter
from datetime import datetime

def test_basic_event_emission():
    """Test basic event emission."""
    print("Test 1: Basic event emission...")
    emitter = TelemetryEmitter(run_id="test_run_123")
    events = []
    
    def capture_event(event):
        events.append(event)
    
    emitter.subscribe(capture_event)
    emitter.emit_event(
        event_type="training.phase.started",
        phase="phase_1a_screening",
        unit="combination",
        status="started",
        current=0,
        total=12
    )
    
    assert len(events) == 1, f"Expected 1 event, got {len(events)}"
    event = events[0]
    
    # Verify required fields
    assert event['run_id'] == "test_run_123"
    assert event['seq'] == 1
    assert event['event_id']
    assert event['emitted_at']
    assert event['event_type'] == "training.phase.started"
    assert event['phase'] == "phase_1a_screening"
    assert event['unit'] == "combination"
    assert event['status'] == "started"
    assert event['current'] == 0
    assert event['total'] == 12
    print("✓ PASSED")


def test_seq_monotonic():
    """Test sequence numbers are monotonically increasing."""
    print("Test 2: Monotonic sequence numbers...")
    emitter = TelemetryEmitter(run_id="test_run_456")
    events = []
    
    def capture(event):
        events.append(event)
    
    emitter.subscribe(capture)
    
    for i in range(5):
        emitter.emit_event(
            event_type=f"test.event.{i}",
            phase="test",
            unit="test",
            status="in_progress"
        )
    
    seqs = [e['seq'] for e in events]
    assert seqs == sorted(seqs), f"Seqs not sorted: {seqs}"
    assert len(set(seqs)) == len(seqs), f"Duplicate seqs: {seqs}"
    assert seqs == [1, 2, 3, 4, 5], f"Expected [1,2,3,4,5], got {seqs}"
    print("✓ PASSED")


def test_event_id_unique():
    """Test event IDs are unique."""
    print("Test 3: Unique event IDs...")
    emitter = TelemetryEmitter(run_id="test_run_789")
    events = []
    
    def capture(event):
        events.append(event)
    
    emitter.subscribe(capture)
    
    for _ in range(10):
        emitter.emit_event(
            event_type="test.event",
            phase="test",
            unit="test",
            status="in_progress"
        )
    
    event_ids = [e['event_id'] for e in events]
    assert len(set(event_ids)) == len(event_ids), f"Duplicate event IDs"
    print("✓ PASSED")


def test_callback_optional():
    """Test that callback is optional."""
    print("Test 4: Callback is optional...")
    emitter = TelemetryEmitter(run_id="no_callback")
    
    # Should not raise
    emitter.emit_event("training.run.started", "init", "run", "started")
    emitter.emit_event("training.phase.started", "phase_1a_screening", "phase", "started")
    emitter.emit_event("training.run.completed", "completion", "run", "completed")
    print("✓ PASSED")


def test_callback_error_isolation():
    """Test that callback errors are isolated."""
    print("Test 5: Callback error isolation...")
    emitter = TelemetryEmitter(run_id="callback_error")
    events = []
    
    def broken_callback(event):
        raise ValueError("Callback error")
    
    def working_callback(event):
        events.append(event)
    
    emitter.subscribe(broken_callback)
    emitter.subscribe(working_callback)
    
    # Should not raise despite broken callback
    emitter.emit_event("training.run.started", "init", "run", "started")
    emitter.emit_event("training.phase.started", "phase_1a_screening", "phase", "started")
    
    # Working callback should have received events
    assert len(events) == 2, f"Expected 2 events from working callback, got {len(events)}"
    print("✓ PASSED")


def test_unit_labels():
    """Test that unit labels are correct."""
    print("Test 6: Unit labels...")
    emitter = TelemetryEmitter(run_id="unit_labels")
    events = []
    
    def capture(event):
        events.append(event)
    
    emitter.subscribe(capture)
    
    # Phase 1a: combinations
    emitter.emit_event("training.phase.started", "phase_1a_screening", "combination", "started")
    emitter.emit_event("training.candidate.completed", "phase_1a_screening", "combination",
                      "completed", current=1, total=12)
    
    # Phase 1b: candidates (top subset)
    emitter.emit_event("training.phase.started", "phase_1b_full_cv", "candidate", "started")
    emitter.emit_event("training.candidate.completed", "phase_1b_full_cv", "candidate",
                      "completed", current=1, total=3)
    
    units = [e['unit'] for e in events]
    assert all(u in ['combination', 'candidate'] for u in units), f"Invalid units: {units}"
    assert 'epoch' not in units, f"Epoch found in units: {units}"
    print("✓ PASSED")


def test_lifecycle_events():
    """Test lifecycle events."""
    print("Test 7: Lifecycle events...")
    emitter = TelemetryEmitter(run_id="lifecycle")
    events = []
    
    def capture(event):
        events.append(event)
    
    emitter.subscribe(capture)
    
    # Simulate lifecycle
    emitter.emit_event("training.run.started", "init", "run", "started")
    emitter.emit_event("training.phase.started", "phase_1a_screening", "phase", "started")
    emitter.emit_event("training.phase.completed", "phase_1a_screening", "phase", "completed")
    emitter.emit_event("training.run.completed", "completion", "run", "completed",
                      metrics={"best_auc": 0.95})
    
    event_types = [e['event_type'] for e in events]
    assert "training.run.started" in event_types
    assert "training.run.completed" in event_types
    assert "training.phase.started" in event_types
    assert "training.phase.completed" in event_types
    print("✓ PASSED")


def test_create_emitter():
    """Test create_emitter factory function."""
    print("Test 8: create_emitter factory...")
    emitter = create_emitter(run_id="custom_run_123")
    assert emitter.run_id == "custom_run_123"
    print("✓ PASSED")


def test_metrics_and_model_identity():
    """Test metrics and model/preprocessor identity fields."""
    print("Test 9: Metrics and model identity...")
    emitter = TelemetryEmitter(run_id="metrics_test")
    events = []
    
    def capture(event):
        events.append(event)
    
    emitter.subscribe(capture)
    
    emitter.emit_event("training.candidate.completed", "phase_1a_screening", "combination",
                      "completed", model="RandomForest", preprocessor="StandardScaler",
                      metrics={"auc": 0.92, "fit_time": 2.5}, current=1, total=12)
    
    assert len(events) == 1
    event = events[0]
    assert event['model'] == "RandomForest"
    assert event['preprocessor'] == "StandardScaler"
    assert event['metrics']['auc'] == 0.92
    assert event['metrics']['fit_time'] == 2.5
    print("✓ PASSED")


def test_error_events():
    """Test error event emission."""
    print("Test 10: Error events...")
    emitter = TelemetryEmitter(run_id="error_test")
    events = []
    
    def capture(event):
        events.append(event)
    
    emitter.subscribe(capture)
    
    try:
        raise ValueError("Model training failed")
    except Exception as e:
        emitter.emit_event(
            "training.candidate.failed",
            "phase_1a_screening",
            "combination",
            "failed",
            model="RandomForest",
            preprocessor="StandardScaler",
            error_type=type(e).__name__,
            error_message=str(e)[:100]
        )
    
    assert len(events) == 1
    event = events[0]
    assert event['event_type'] == "training.candidate.failed"
    assert event['status'] == "failed"
    assert event['error_type'] == "ValueError"
    assert "Model training failed" in event['error_message']
    print("✓ PASSED")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Telemetry Emitter")
    print("=" * 70)
    
    try:
        test_basic_event_emission()
        test_seq_monotonic()
        test_event_id_unique()
        test_callback_optional()
        test_callback_error_isolation()
        test_unit_labels()
        test_lifecycle_events()
        test_create_emitter()
        test_metrics_and_model_identity()
        test_error_events()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED (10/10)")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
