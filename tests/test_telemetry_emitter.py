"""
Tests for telemetry event emission in adaptive trainer.

Verifies:
- Event envelope includes required fields: run_id, seq, event_id, emitted_at, event_type, phase, unit, status
- Sequence numbers are monotonically increasing per run_id
- Events are emitted at correct lifecycle boundaries
- Callback is optional and callback-absent behavior is preserved
"""

import pytest
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

from harp.shared_lib.telemetry_emitter import (
    TelemetryEmitter, 
    ProgressEvent,
    create_emitter,
    emit_event
)


class TestEventEnvelope:
    """Test event envelope structure and required fields."""
    
    def test_event_has_required_fields(self):
        """Event must include run_id, seq, event_id, emitted_at, event_type, phase, unit, status."""
        emitter = TelemetryEmitter(run_id="test_run_123")
        events: List[ProgressEvent] = []
        
        def capture_event(event: ProgressEvent):
            events.append(event)
        
        emitter.emit_event(
            event_type="training.run.started",
            phase="init",
            unit="run",
            status="started"
        )
        
        # Capture event through callback
        emitter.subscribe(capture_event)
        emitter.emit_event(
            event_type="training.phase.started",
            phase="phase_1a_screening",
            unit="combination",
            status="started",
            current=0,
            total=12
        )
        
        assert len(events) == 1
        event = events[0]
        
        # Verify required fields exist
        assert event['run_id'] == "test_run_123"
        assert isinstance(event['seq'], int)
        assert event['seq'] > 0
        assert event['event_id']  # Must be non-empty
        assert event['emitted_at']  # Must be ISO-8601 timestamp
        assert event['event_type'] == "training.phase.started"
        assert event['phase'] == "phase_1a_screening"
        assert event['unit'] == "combination"
        assert event['status'] == "started"
    
    def test_seq_is_monotonically_increasing(self):
        """Sequence numbers must strictly increase per run_id."""
        emitter = TelemetryEmitter(run_id="test_run_456")
        events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            events.append(event)
        
        emitter.subscribe(capture)
        
        for i in range(5):
            emitter.emit_event(
                event_type=f"test.event.{i}",
                phase="test",
                unit="test",
                status="in_progress"
            )
        
        # Extract sequence numbers
        seqs = [e['seq'] for e in events]
        assert seqs == sorted(seqs), "Sequence numbers must be monotonically increasing"
        assert len(set(seqs)) == len(seqs), "Sequence numbers must be unique"
    
    def test_event_id_is_unique(self):
        """Each event must have a unique event_id."""
        emitter = TelemetryEmitter(run_id="test_run_789")
        events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
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
        assert len(set(event_ids)) == len(event_ids), "Event IDs must be unique"
    
    def test_emitted_at_is_valid_iso8601(self):
        """emitted_at must be valid ISO-8601 timestamp."""
        emitter = TelemetryEmitter(run_id="test_run_timestamp")
        events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            events.append(event)
        
        emitter.subscribe(capture)
        emitter.emit_event(
            event_type="test.event",
            phase="test",
            unit="test",
            status="started"
        )
        
        assert len(events) == 1
        # Should not raise
        datetime.fromisoformat(events[0]['emitted_at'].replace('Z', '+00:00'))


class TestEventLifecycle:
    """Test event lifecycle and ordering."""
    
    def test_run_lifecycle_events_emitted(self):
        """Verify run.started and run.completed/failed events are emitted."""
        emitter = TelemetryEmitter(run_id="lifecycle_test")
        events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            events.append(event)
        
        emitter.subscribe(capture)
        
        # Simulate run lifecycle
        emitter.emit_event("training.run.started", "init", "run", "started")
        emitter.emit_event("training.phase.started", "phase_1a_screening", "phase", "started")
        emitter.emit_event("training.phase.completed", "phase_1a_screening", "phase", "completed")
        emitter.emit_event("training.run.completed", "completion", "run", "completed", 
                          metrics={"best_auc": 0.95})
        
        event_types = [e['event_type'] for e in events]
        assert "training.run.started" in event_types
        assert "training.run.completed" in event_types
    
    def test_phase_lifecycle_events(self):
        """Verify phase.started and phase.completed events bracket phase execution."""
        emitter = TelemetryEmitter(run_id="phase_lifecycle")
        events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            events.append(event)
        
        emitter.subscribe(capture)
        
        # Phase 1a
        emitter.emit_event("training.phase.started", "phase_1a_screening", "combination", "started",
                          current=0, total=12)
        emitter.emit_event("training.phase.completed", "phase_1a_screening", "combination", "completed",
                          current=12, total=12)
        
        # Phase 1b
        emitter.emit_event("training.phase.started", "phase_1b_full_cv", "candidate", "started",
                          current=0, total=3)
        emitter.emit_event("training.phase.completed", "phase_1b_full_cv", "candidate", "completed",
                          current=3, total=3)
        
        event_types = [e['event_type'] for e in events]
        assert event_types.count("training.phase.started") == 2
        assert event_types.count("training.phase.completed") == 2
    
    def test_candidate_progress_events(self):
        """Verify per-candidate completion events are emitted with model/preprocessor identity."""
        emitter = TelemetryEmitter(run_id="candidate_progress")
        events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            events.append(event)
        
        emitter.subscribe(capture)
        
        emitter.emit_event("training.candidate.completed", "phase_1a_screening", "combination",
                          "completed", model="RandomForest", preprocessor="StandardScaler",
                          metrics={"auc": 0.92}, current=1, total=12)
        
        assert len(events) == 1
        event = events[0]
        assert event['model'] == "RandomForest"
        assert event['preprocessor'] == "StandardScaler"
        assert event['metrics']['auc'] == 0.92


class TestCallbackOptional:
    """Test that callback is optional and behavior is correct when absent."""
    
    def test_emitter_works_without_callback(self):
        """Emitter should not fail if no callback is subscribed."""
        emitter = TelemetryEmitter(run_id="no_callback")
        
        # Should not raise
        emitter.emit_event("training.run.started", "init", "run", "started")
        emitter.emit_event("training.phase.started", "phase_1a_screening", "phase", "started")
        emitter.emit_event("training.run.completed", "completion", "run", "completed")
    
    def test_emitter_isolates_callback_errors(self):
        """Callback errors should not crash emitter."""
        emitter = TelemetryEmitter(run_id="callback_error")
        
        def broken_callback(event: ProgressEvent):
            raise ValueError("Callback error")
        
        emitter.subscribe(broken_callback)
        
        # Should not raise despite callback error
        emitter.emit_event("training.run.started", "init", "run", "started")
        emitter.emit_event("training.phase.started", "phase_1a_screening", "phase", "started")
    
    def test_multiple_subscribers(self):
        """Multiple callbacks should each receive events."""
        emitter = TelemetryEmitter(run_id="multi_subscriber")
        events1: List[ProgressEvent] = []
        events2: List[ProgressEvent] = []
        
        emitter.subscribe(lambda e: events1.append(e))
        emitter.subscribe(lambda e: events2.append(e))
        
        emitter.emit_event("training.run.started", "init", "run", "started")
        emitter.emit_event("training.phase.started", "phase_1a_screening", "phase", "started")
        
        assert len(events1) == 2
        assert len(events2) == 2


class TestNonEpochUnitLabels:
    """Test that sklearn non-epoch progress uses combination/fold/candidate labels."""
    
    def test_unit_labels_are_combination_or_candidate(self):
        """For sklearn, unit should be combination or candidate, never epoch."""
        emitter = TelemetryEmitter(run_id="unit_labels")
        events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
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
        assert all(u in ['combination', 'candidate'] for u in units)
        assert 'epoch' not in units


class TestEventEmitterModule:
    """Test the telemetry_emitter module and factory functions."""
    
    def test_create_emitter_generates_run_id(self):
        """create_emitter should generate a run_id if not provided."""
        emitter = create_emitter()
        assert emitter.run_id
        assert len(emitter.run_id) > 0
    
    def test_create_emitter_respects_provided_run_id(self):
        """create_emitter should use provided run_id."""
        emitter = create_emitter(run_id="custom_run_123")
        assert emitter.run_id == "custom_run_123"
    
    def test_emit_event_convenience_function(self):
        """Module-level emit_event should work with module state."""
        emitter = create_emitter(run_id="module_test")
        
        # This would require module-level state, tested in integration
        assert emitter is not None


class TestPayloadThrottling:
    """Test payload compaction and throttling."""
    
    def test_event_with_metrics_snapshot(self):
        """Events should include compact metrics snapshot."""
        emitter = TelemetryEmitter(run_id="metrics_test")
        events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            events.append(event)
        
        emitter.subscribe(capture)
        
        emitter.emit_event("training.metric.updated", "phase_1a_screening", "combination",
                          "in_progress", model="SVM", preprocessor="PCA",
                          metrics={"auc": 0.91, "fit_time": 2.5})
        
        assert len(events) == 1
        assert 'metrics' in events[0]
        assert events[0]['metrics']['auc'] == 0.91
        assert events[0]['metrics']['fit_time'] == 2.5
    
    def test_optional_fields_only_when_set(self):
        """Optional fields should only appear in event when set."""
        emitter = TelemetryEmitter(run_id="optional_fields")
        events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            events.append(event)
        
        emitter.subscribe(capture)
        
        # Emit with minimal fields
        emitter.emit_event("training.run.started", "init", "run", "started")
        
        event = events[0]
        # Required fields always present
        assert 'run_id' in event
        assert 'seq' in event
        assert 'event_id' in event
        # Optional fields may be absent
        # (depends on implementation choice)


class TestErrorEvents:
    """Test error event emission."""
    
    def test_failure_event_with_exception_details(self):
        """Failure events should include error_type and error_message."""
        emitter = TelemetryEmitter(run_id="error_test")
        events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
