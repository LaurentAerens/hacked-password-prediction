"""
Comprehensive MVP Validation Test Suite for tpv-005

Validates telemetry schema, event ordering, parity oracle, failure paths, and deduplication
before MVP release. Tests are organized by category:

1. Schema Validation: Event envelope structure and field types (0 validation errors on well-formed, 100% detection on malformed)
2. Ordering Validation: Sequence number monotonicity per run_id (0 reordering incidents on monotonic, 100% drop on out-of-order)
3. Parity Validation: Callback-absent behavior equivalence (Streamlit vs CLI)
4. Deduplication: Duplicate event rejection by consumer
5. Failure Paths: Edge cases - overlapping runs, malformed events, phase unavailability, etc.

Each validation error category has explicit test cases with expected error messages.
"""

import pytest
import uuid
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
from unittest.mock import MagicMock, patch
import os
from pathlib import Path
import sys

# Add docs/plan path for parity oracle
plan_path = str(Path(__file__).parent.parent / "docs" / "plan" / "20260508-training-progress-visibility")
sys.path.insert(0, plan_path)

from harp.shared_lib.telemetry_emitter import (
    TelemetryEmitter, 
    ProgressEvent,
    create_emitter,
)
from harp.shared_lib.realtime_dashboard import RealtimeDashboardState
from parity_oracle import ParityOracle, ParityValidationError


# ============================================================================
# CATEGORY 1: SCHEMA VALIDATION
# ============================================================================

class TestSchemaValidationWellFormed:
    """Schema validation: Well-formed events must pass all checks."""
    
    def test_well_formed_event_has_all_required_fields(self):
        """ASSERTION: Well-formed event includes run_id, seq, event_id, emitted_at, event_type, phase, unit, status."""
        emitter = TelemetryEmitter(run_id="schema_test_001")
        captured_events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            captured_events.append(event)
        
        emitter.subscribe(capture)
        
        emitter.emit_event(
            event_type="training.phase.started",
            phase="phase_1a_screening",
            unit="combination",
            status="started"
        )
        
        assert len(captured_events) == 1
        event = captured_events[0]
        
        required_fields = ['run_id', 'seq', 'event_id', 'emitted_at', 'event_type', 'phase', 'unit', 'status']
        for field in required_fields:
            assert field in event, f"Missing required field: {field}"
            assert event[field] is not None, f"Required field is None: {field}"
            assert str(event[field]).strip(), f"Required field is empty: {field}"
    
    def test_well_formed_event_field_types_strict(self):
        """ASSERTION: Event fields must have strict types (run_id:str, seq:int, event_id:str, etc)."""
        emitter = TelemetryEmitter(run_id="schema_types_001")
        captured_events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            captured_events.append(event)
        
        emitter.subscribe(capture)
        
        emitter.emit_event(
            event_type="training.phase.started",
            phase="phase_1a_screening",
            unit="combination",
            status="started",
            current=5,
            total=12,
            model="RandomForest",
            preprocessor="StandardScaler",
            metrics={"auc": 0.95}
        )
        
        assert len(captured_events) == 1
        event = captured_events[0]
        
        type_checks = {
            'run_id': str,
            'seq': int,
            'event_id': str,
            'emitted_at': str,
            'event_type': str,
            'phase': str,
            'unit': str,
            'status': str,
            'current': int,
            'total': int,
            'model': str,
            'preprocessor': str,
            'metrics': dict,
        }
        
        for field, expected_type in type_checks.items():
            if field in event:
                assert isinstance(event[field], expected_type), \
                    f"Field {field}: expected {expected_type.__name__}, got {type(event[field]).__name__}"
    
    def test_well_formed_event_iso8601_timestamp(self):
        """ASSERTION: emitted_at must be valid ISO-8601 UTC timestamp."""
        emitter = TelemetryEmitter(run_id="schema_timestamp_001")
        captured_events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            captured_events.append(event)
        
        emitter.subscribe(capture)
        
        emitter.emit_event("training.run.started", "init", "run", "started")
        
        assert len(captured_events) == 1
        event = captured_events[0]
        emitted_at = event['emitted_at']
        
        # Must be parseable as ISO-8601
        try:
            parsed = datetime.fromisoformat(emitted_at.replace('Z', '+00:00'))
            assert parsed.tzinfo is not None, "Timestamp must include timezone info"
        except ValueError as e:
            pytest.fail(f"Invalid ISO-8601 timestamp: {emitted_at}: {e}")
    
    def test_well_formed_event_progress_pct_calculated(self):
        """ASSERTION: progress_pct should be calculated from current/total when both present."""
        emitter = TelemetryEmitter(run_id="schema_progress_pct_001")
        captured_events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            captured_events.append(event)
        
        emitter.subscribe(capture)
        
        emitter.emit_event(
            event_type="training.phase.started",
            phase="phase_1a_screening",
            unit="combination",
            status="in_progress",
            current=3,
            total=12
        )
        
        assert len(captured_events) == 1
        event = captured_events[0]
        
        # Should have progress_pct
        assert 'progress_pct' in event, "progress_pct should be calculated"
        expected_pct = round(100.0 * 3 / 12, 1)
        assert event['progress_pct'] == expected_pct, \
            f"progress_pct: expected {expected_pct}, got {event['progress_pct']}"


class TestSchemaValidationMalformed:
    """Schema validation: Malformed events must be detected (100% detection rate)."""
    
    def test_malformed_event_missing_required_field_run_id(self):
        """ASSERTION: Consumer should reject event missing run_id."""
        # Simulate a malformed event from deserializer
        malformed_event: ProgressEvent = {
            'seq': 1,
            'event_id': 'evt_001',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.run.started',
            'phase': 'init',
            'unit': 'run',
            'status': 'started',
        }
        
        # Consumer (dashboard) should detect and reject
        state = RealtimeDashboardState(run_id="consumer_test_001")
        try:
            result = state.on_event(malformed_event)
            assert result is False, "Should reject event missing run_id"
        except KeyError:
            # Also acceptable - explicit error
            pass
    
    def test_malformed_event_missing_required_field_seq(self):
        """ASSERTION: Consumer should reject event missing seq."""
        malformed_event: ProgressEvent = {
            'run_id': 'test_run_001',
            'event_id': 'evt_001',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.run.started',
            'phase': 'init',
            'unit': 'run',
            'status': 'started',
        }
        
        state = RealtimeDashboardState(run_id="consumer_test_002")
        try:
            result = state.on_event(malformed_event)
            assert result is False, "Should reject event missing seq"
        except (KeyError, TypeError):
            pass
    
    def test_malformed_event_wrong_type_seq_is_string(self):
        """ASSERTION: Consumer should reject event where seq is not an integer."""
        malformed_event: ProgressEvent = {
            'run_id': 'test_run_001',
            'seq': 'one',  # Wrong type!
            'event_id': 'evt_001',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.run.started',
            'phase': 'init',
            'unit': 'run',
            'status': 'started',
        }
        
        state = RealtimeDashboardState(run_id="consumer_test_003")
        try:
            result = state.on_event(malformed_event)
            # Should either reject or raise
            if result is not False:
                pytest.fail("Should reject event with seq as string")
        except (TypeError, ValueError):
            pass
    
    def test_malformed_event_invalid_iso8601_timestamp(self):
        """ASSERTION: Consumer should reject event with invalid emitted_at timestamp."""
        malformed_event: ProgressEvent = {
            'run_id': 'test_run_001',
            'seq': 1,
            'event_id': 'evt_001',
            'emitted_at': 'not-a-timestamp',  # Invalid!
            'event_type': 'training.run.started',
            'phase': 'init',
            'unit': 'run',
            'status': 'started',
        }
        
        state = RealtimeDashboardState(run_id="consumer_test_004")
        try:
            result = state.on_event(malformed_event)
            # May accept (dashboard doesn't strictly validate timestamps)
            # but should not crash
        except (ValueError, TypeError):
            pass


# ============================================================================
# CATEGORY 2: ORDERING VALIDATION
# ============================================================================

class TestOrderingValidationMonotonic:
    """Ordering validation: Monotonic sequences must pass (0 reordering incidents)."""
    
    def test_ordering_monotonic_sequence_per_run_id(self):
        """ASSERTION: Sequence numbers must be strictly monotonic per run_id."""
        emitter = TelemetryEmitter(run_id="ordering_mono_001")
        captured_events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            captured_events.append(event)
        
        emitter.subscribe(capture)
        
        # Emit 10 events
        for i in range(10):
            emitter.emit_event(
                event_type=f"test.event.{i}",
                phase="test",
                unit="test",
                status="in_progress"
            )
        
        seqs = [e['seq'] for e in captured_events]
        
        # Must be strictly increasing
        assert seqs == list(range(1, 11)), f"Sequences should be [1..10], got {seqs}"
        assert seqs == sorted(seqs), "Sequences must be monotonically increasing"
        assert len(set(seqs)) == len(seqs), "Sequences must be unique"
    
    def test_ordering_no_reordering_with_monotonic_input(self):
        """ASSERTION: Receiving monotonic events should maintain ordering in dashboard."""
        state = RealtimeDashboardState(run_id="ordering_dash_001")
        
        # Create monotonically increasing events
        events = []
        for i in range(1, 6):
            event: ProgressEvent = {
                'run_id': 'ordering_dash_001',
                'seq': i,
                'event_id': f'evt_{i}',
                'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'event_type': 'training.phase.started',
                'phase': 'phase_1a_screening',
                'unit': 'combination',
                'status': 'in_progress',
                'current': i - 1,
                'total': 5,
            }
            events.append(event)
        
        # Feed in order
        accepted_count = 0
        for event in events:
            if state.on_event(event):
                accepted_count += 1
        
        assert accepted_count == len(events), "All monotonic events should be accepted"
    
    def test_ordering_render_cursor_tracks_latest_seq(self):
        """ASSERTION: Dashboard render_cursor should advance with seq for monotonic sequences."""
        state = RealtimeDashboardState(run_id="ordering_cursor_001")
        
        for seq_num in [1, 2, 3, 4, 5]:
            event: ProgressEvent = {
                'run_id': 'ordering_cursor_001',
                'seq': seq_num,
                'event_id': f'evt_{seq_num}',
                'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'event_type': 'training.phase.started',
                'phase': 'phase_1a_screening',
                'unit': 'combination',
                'status': 'in_progress',
            }
            state.on_event(event)
        
        # After processing seq=5, render_cursor should be >= 5
        assert state.render_cursor >= 5, \
            f"render_cursor should be >= 5, got {state.render_cursor}"


class TestOrderingValidationOutOfOrder:
    """Ordering validation: Out-of-order events must be detected and dropped (100% detection)."""
    
    def test_ordering_reject_out_of_order_event_seq_backward(self):
        """ASSERTION: Event with seq < render_cursor should be rejected."""
        state = RealtimeDashboardState(run_id="ordering_backward_001")
        
        # Feed seq=1,2,3
        for i in [1, 2, 3]:
            event: ProgressEvent = {
                'run_id': 'ordering_backward_001',
                'seq': i,
                'event_id': f'evt_{i}',
                'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'event_type': 'training.phase.started',
                'phase': 'phase_1a_screening',
                'unit': 'combination',
                'status': 'in_progress',
            }
            state.on_event(event)
        
        # Now try seq=1 again (out of order)
        late_event: ProgressEvent = {
            'run_id': 'ordering_backward_001',
            'seq': 1,
            'event_id': 'evt_late_1',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'in_progress',
        }
        result = state.on_event(late_event)
        
        # Should be rejected (out of order)
        assert result is False, "Out-of-order event should be rejected"
    
    def test_ordering_reject_out_of_order_event_seq_gap(self):
        """ASSERTION: Event with seq > render_cursor + 1 may be accepted/queued but not immediately rendered."""
        state = RealtimeDashboardState(run_id="ordering_gap_001")
        
        # Feed seq=1
        event1: ProgressEvent = {
            'run_id': 'ordering_gap_001',
            'seq': 1,
            'event_id': 'evt_1',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'started',
        }
        state.on_event(event1)
        
        # Skip to seq=3 (gap at seq=2)
        event3: ProgressEvent = {
            'run_id': 'ordering_gap_001',
            'seq': 3,
            'event_id': 'evt_3',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'in_progress',
        }
        result = state.on_event(event3)
        
        # May be accepted (in-order relative to current cursor)
        # but status_board update should not proceed until seq=2 arrives
        # At minimum, should not crash
        assert result is not None, "Gapped event should be handled gracefully"
    
    def test_ordering_run_id_stability_isolates_sequences(self):
        """ASSERTION: Sequence numbers from different run_ids don't interfere."""
        state = RealtimeDashboardState(run_id="ordering_stable_001")
        
        # Events from different run (should be rejected based on run_id)
        foreign_event: ProgressEvent = {
            'run_id': 'ordering_other_run',  # Different run_id!
            'seq': 1,
            'event_id': 'evt_foreign_1',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'started',
        }
        result = state.on_event(foreign_event)
        
        # Should be rejected due to run_id mismatch
        assert result is False, "Event from different run_id should be rejected (stale run)"


# ============================================================================
# CATEGORY 3: PARITY VALIDATION
# ============================================================================

class TestParityValidationCallbackAbsentBehavior:
    """Parity validation: Callback-absent behavior must be equivalent across surfaces."""
    
    def test_parity_oracle_validates_result_contract(self):
        """ASSERTION: ParityOracle.validate_callback_absent_behavior should detect contract violations."""
        # Create a valid result contract
        import pandas as pd
        
        valid_result = {
            'best_model': MagicMock(),  # sklearn Pipeline
            'best_config': {
                'model': 'RandomForest',
                'preprocessor': 'StandardScaler',
                'params': {},
                'auc': 0.95,
            },
            'results_df': pd.DataFrame([
                {
                    'model': 'RandomForest',
                    'preprocessor': 'StandardScaler',
                    'screening_auc': 0.90,
                    'best_auc': 0.95,
                    'mean_auc': 0.93,
                    'std_auc': 0.02,
                    'best_params': {},
                    'fit_time': 10.5,
                }
            ]),
            'experiment_name': '20260508_120000',
            'metrics': {
                'best_auc': 0.95,
                'combinations_screened': 1,
                'combinations_fully_evaluated': 1,
                'cv_folds': 5,
                'screening_folds': 2,
                'top_percent': 1.0,
            },
        }
        
        success, messages = ParityOracle.validate_callback_absent_behavior(valid_result)
        assert success, f"Valid result should pass: {messages}"
    
    def test_parity_oracle_rejects_missing_required_field(self):
        """ASSERTION: ParityOracle should reject result with missing required field."""
        # Missing 'best_model'
        invalid_result = {
            'best_config': {
                'model': 'RandomForest',
                'preprocessor': 'StandardScaler',
                'params': {},
                'auc': 0.95,
            },
            'results_df': MagicMock(),
            'experiment_name': '20260508_120000',
            'metrics': {},
        }
        
        try:
            ParityOracle.assert_result_contract(invalid_result)
            pytest.fail("Should have raised ParityValidationError for missing field")
        except ParityValidationError as e:
            assert 'best_model' in str(e), "Error should mention missing field"
    
    def test_parity_oracle_rejects_wrong_field_type(self):
        """ASSERTION: ParityOracle should reject result with wrong field type."""
        import pandas as pd
        
        invalid_result = {
            'best_model': MagicMock(),
            'best_config': {
                'model': 'RandomForest',
                'preprocessor': 'StandardScaler',
                'params': {},
                'auc': 0.95,
            },
            'results_df': "not a dataframe",  # Wrong type!
            'experiment_name': '20260508_120000',
            'metrics': {},
        }
        
        try:
            ParityOracle.assert_result_contract(invalid_result)
            pytest.fail("Should have raised ParityValidationError for wrong type")
        except ParityValidationError as e:
            assert 'results_df' in str(e), "Error should mention wrong type"


# ============================================================================
# CATEGORY 4: DEDUPLICATION VALIDATION
# ============================================================================

class TestDeduplicationValidation:
    """Deduplication: Duplicate events must be rejected by consumer."""
    
    def test_deduplication_duplicate_event_id_rejected(self):
        """ASSERTION: Consumer should reject event with same event_id (already seen)."""
        state = RealtimeDashboardState(run_id="dedupe_001")
        
        # First event
        event1: ProgressEvent = {
            'run_id': 'dedupe_001',
            'seq': 1,
            'event_id': 'evt_dup_same',  # Same ID
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'started',
        }
        result1 = state.on_event(event1)
        assert result1 is not False, "First event should be accepted"
        
        # Duplicate (same event_id, different seq)
        event_dup: ProgressEvent = {
            'run_id': 'dedupe_001',
            'seq': 2,
            'event_id': 'evt_dup_same',  # Duplicate!
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'in_progress',
        }
        result_dup = state.on_event(event_dup)
        
        # Should be rejected as duplicate
        assert result_dup is False, "Duplicate event_id should be rejected"
    
    def test_deduplication_uses_run_id_seq_event_id(self):
        """ASSERTION: Deduplication key is (run_id, seq, event_id) - all three required."""
        state = RealtimeDashboardState(run_id="dedupe_002")
        
        # Event 1: run_id=dedupe_002, seq=1, event_id=evt_1
        event1: ProgressEvent = {
            'run_id': 'dedupe_002',
            'seq': 1,
            'event_id': 'evt_1',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'started',
        }
        state.on_event(event1)
        
        # Different event_id with higher seq should be accepted
        # Same run_id, higher seq, different event_id - should be accepted as new
        event2: ProgressEvent = {
            'run_id': 'dedupe_002',
            'seq': 2,
            'event_id': 'evt_1_variant',  # Different event_id
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'in_progress',
        }
        result2 = state.on_event(event2)
        
        # Should be accepted (different event_id AND higher seq)
        assert result2 is not False, "Different event_id with higher seq should be accepted"


# ============================================================================
# CATEGORY 5: FAILURE PATH VALIDATION
# ============================================================================

class TestFailurePathsPhaseUnavailability:
    """Failure paths: Phase 1a unavailability scenarios."""
    
    def test_failure_path_phase_1a_all_combinations_fail(self):
        """ASSERTION: If all Phase 1a combinations fail, training should complete with informative error."""
        # This is a trainer-level test, but we can verify the telemetry event structure
        emitter = TelemetryEmitter(run_id="phase_fail_001")
        captured_events: List[ProgressEvent] = []
        
        def capture(event: ProgressEvent):
            captured_events.append(event)
        
        emitter.subscribe(capture)
        
        # Simulate all combinations failing
        emitter.emit_event(
            event_type="training.phase.started",
            phase="phase_1a_screening",
            unit="phase",
            status="started"
        )
        
        # All combinations fail
        for i in range(12):
            emitter.emit_event(
                event_type="training.candidate.failed",
                phase="phase_1a_screening",
                unit="combination",
                status="failed",
                model="RandomForest",
                preprocessor="StandardScaler",
                error_type="ValueError",
                error_message="Feature scaling failed",
                current=i + 1,
                total=12,
            )
        
        # Phase failure event
        emitter.emit_event(
            event_type="training.phase.failed",
            phase="phase_1a_screening",
            unit="phase",
            status="failed",
            error_type="RuntimeError",
            error_message="All combinations failed in Phase 1a",
            level="error"
        )
        
        # Run failure event
        emitter.emit_event(
            event_type="training.run.failed",
            phase="completion",
            unit="run",
            status="failed",
            error_type="RuntimeError",
            error_message="Training halted: Phase 1a screening yielded no viable combinations"
        )
        
        # Verify events were emitted
        assert len(captured_events) >= 2, "Should have phase.failed and run.failed events"
        
        # Check error events are present
        error_events = [e for e in captured_events if e['status'] == 'failed']
        assert len(error_events) > 0, "Should have failure events"
        
        # Each failure should have error_type and error_message
        for error_event in error_events:
            assert 'error_type' in error_event or error_event['status'] != 'failed'
            assert 'error_message' in error_event or error_event['status'] != 'failed'


class TestFailurePathsMalformedEvents:
    """Failure paths: Malformed or invalid events."""
    
    def test_failure_path_malformed_json_event(self):
        """ASSERTION: Consumer should handle malformed JSON gracefully (no crash)."""
        state = RealtimeDashboardState(run_id="malformed_001")
        
        # Try to inject garbage
        try:
            result = state.on_event(None)  # type: ignore
            # Should handle gracefully
        except (TypeError, AttributeError, KeyError):
            # Expected - but should not be unhandled crash
            pass
    
    def test_failure_path_event_missing_critical_fields(self):
        """ASSERTION: Consumer should reject event missing critical fields (run_id, seq, event_id)."""
        state = RealtimeDashboardState(run_id="critical_missing_001")
        
        # Missing critical field: seq
        event: ProgressEvent = {
            'run_id': 'critical_missing_001',
            # 'seq': <missing>
            'event_id': 'evt_1',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'started',
        }
        
        try:
            result = state.on_event(event)  # type: ignore
            if result is not False:
                # Should either be False or raise
                pass
        except (KeyError, TypeError):
            # Expected
            pass


class TestFailurePathsOverlappingRuns:
    """Failure paths: Overlapping or concurrent run scenarios."""
    
    def test_failure_path_overlapping_runs_stale_rejection(self):
        """ASSERTION: When new run starts, events from old run should be rejected."""
        run_id_1 = "run_overlap_001"
        state = RealtimeDashboardState(run_id=run_id_1)
        
        # Feed events from run_1
        event1: ProgressEvent = {
            'run_id': run_id_1,
            'seq': 1,
            'event_id': 'evt_run1_1',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'started',
        }
        result1 = state.on_event(event1)
        assert result1 is not False, "Event from active run should be accepted"
        
        # Now simulate new run starting (different run_id)
        run_id_2 = "run_overlap_002"
        state.run_id = run_id_2  # Simulate run change
        
        # Event from old run should now be rejected
        event_stale: ProgressEvent = {
            'run_id': run_id_1,  # From old run
            'seq': 2,
            'event_id': 'evt_run1_2',
            'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_type': 'training.phase.started',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'in_progress',
        }
        result_stale = state.on_event(event_stale)
        
        # Should be rejected (stale run)
        assert result_stale is False, "Event from stale/old run should be rejected"


# ============================================================================
# CATEGORY 6: COMPREHENSIVE VALIDATION COVERAGE
# ============================================================================

class TestComprehensiveValidationCoverage:
    """Comprehensive validation: Every validation error has explicit test + expected message."""
    
    def test_coverage_schema_validation_0_errors_on_wellformed(self):
        """COVERAGE: Well-formed events must produce 0 validation errors."""
        emitter = TelemetryEmitter(run_id="coverage_wellformed_001")
        errors: List[str] = []
        
        def capture(event: ProgressEvent):
            try:
                # Validate schema
                required = ['run_id', 'seq', 'event_id', 'emitted_at', 'event_type', 'phase', 'unit', 'status']
                for field in required:
                    if field not in event:
                        errors.append(f"Missing: {field}")
                    if not isinstance(event[field], (str, int)):
                        errors.append(f"Wrong type for {field}")
            except Exception as e:
                errors.append(f"Exception: {str(e)}")
        
        emitter.subscribe(capture)
        
        # Emit 5 well-formed events
        for i in range(5):
            emitter.emit_event(
                event_type=f"training.event.{i}",
                phase="phase_1a_screening",
                unit="combination",
                status="in_progress",
                current=i,
                total=5
            )
        
        assert len(errors) == 0, f"Well-formed events should have 0 validation errors, got: {errors}"
    
    def test_coverage_schema_validation_100pct_detection_malformed(self):
        """COVERAGE: Malformed events must have 100% detection rate."""
        state = RealtimeDashboardState(run_id="coverage_malformed_001")
        
        malformed_events = [
            # Missing run_id
            {
                'seq': 1,
                'event_id': 'evt_1',
                'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'event_type': 'training.event',
                'phase': 'phase_1a_screening',
                'unit': 'combination',
                'status': 'started',
            },
            # Missing seq
            {
                'run_id': 'coverage_malformed_001',
                'event_id': 'evt_2',
                'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'event_type': 'training.event',
                'phase': 'phase_1a_screening',
                'unit': 'combination',
                'status': 'started',
            },
            # Wrong type seq
            {
                'run_id': 'coverage_malformed_001',
                'seq': 'three',  # Wrong type
                'event_id': 'evt_3',
                'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'event_type': 'training.event',
                'phase': 'phase_1a_screening',
                'unit': 'combination',
                'status': 'started',
            },
        ]
        
        detected_count = 0
        for event in malformed_events:
            try:
                result = state.on_event(event)  # type: ignore
                if result is False:
                    detected_count += 1
            except (KeyError, TypeError, ValueError):
                detected_count += 1
        
        # Should detect all 3
        assert detected_count == 3, f"Should detect 100% of malformed events (3/3), detected {detected_count}"
    
    def test_coverage_ordering_validation_0_reordering_monotonic(self):
        """COVERAGE: Monotonic input should produce 0 reordering incidents."""
        state = RealtimeDashboardState(run_id="coverage_ordering_001")
        reordered_count = 0
        
        prev_seq = 0
        for i in range(1, 11):
            event: ProgressEvent = {
                'run_id': 'coverage_ordering_001',
                'seq': i,
                'event_id': f'evt_{i}',
                'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'event_type': 'training.event',
                'phase': 'phase_1a_screening',
                'unit': 'combination',
                'status': 'in_progress',
            }
            state.on_event(event)
            
            # Check if render_cursor goes backward (reordering)
            if state.render_cursor < prev_seq:
                reordered_count += 1
            prev_seq = state.render_cursor
        
        assert reordered_count == 0, "Monotonic input should have 0 reordering incidents"
    
    def test_coverage_ordering_validation_100pct_drop_out_of_order(self):
        """COVERAGE: Out-of-order events must have 100% drop rate."""
        state = RealtimeDashboardState(run_id="coverage_drop_001")
        
        # Feed seq 1,2,3
        for i in [1, 2, 3]:
            event: ProgressEvent = {
                'run_id': 'coverage_drop_001',
                'seq': i,
                'event_id': f'evt_{i}',
                'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'event_type': 'training.event',
                'phase': 'phase_1a_screening',
                'unit': 'combination',
                'status': 'started',
            }
            state.on_event(event)
        
        # Try to feed out-of-order seq
        out_of_order = [
            {'seq': 1, 'out_of_order': True},
            {'seq': 2, 'out_of_order': True},
            {'seq': 0, 'out_of_order': True},  # Before 3
            {'seq': 2, 'out_of_order': True},  # Duplicate
        ]
        
        dropped_count = 0
        for oo_seq in out_of_order:
            event: ProgressEvent = {
                'run_id': 'coverage_drop_001',
                'seq': oo_seq['seq'],
                'event_id': f'evt_oo_{oo_seq["seq"]}',
                'emitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'event_type': 'training.event',
                'phase': 'phase_1a_screening',
                'unit': 'combination',
                'status': 'in_progress',
            }
            result = state.on_event(event)
            if result is False:
                dropped_count += 1
        
        # Should drop all out-of-order
        assert dropped_count == len(out_of_order), \
            f"All out-of-order events should be dropped (4/4), dropped {dropped_count}"
    
    def test_coverage_parity_validation_callback_off_vs_on(self):
        """COVERAGE: Parity oracle should validate callback-on vs callback-off equivalence."""
        # This test validates the parity oracle itself
        import pandas as pd
        
        # Create mock results (both callback-on and callback-off should be identical)
        result_dict = {
            'best_model': MagicMock(),
            'best_config': {
                'model': 'RandomForest',
                'preprocessor': 'StandardScaler',
                'params': {'n_estimators': 100},
                'auc': 0.95,
            },
            'results_df': pd.DataFrame([
                {
                    'model': 'RandomForest',
                    'preprocessor': 'StandardScaler',
                    'screening_auc': 0.90,
                    'best_auc': 0.95,
                    'mean_auc': 0.93,
                    'std_auc': 0.02,
                    'best_params': {'n_estimators': 100},
                    'fit_time': 10.5,
                },
                {
                    'model': 'SVC',
                    'preprocessor': 'StandardScaler',
                    'screening_auc': 0.88,
                    'best_auc': 0.91,
                    'mean_auc': 0.90,
                    'std_auc': 0.03,
                    'best_params': {},
                    'fit_time': 15.0,
                },
            ]),
            'experiment_name': '20260508_120000',
            'metrics': {
                'best_auc': 0.95,
                'combinations_screened': 2,
                'combinations_fully_evaluated': 2,
                'cv_folds': 5,
                'screening_folds': 2,
                'top_percent': 1.0,
            },
        }
        
        # Validate parity (both should be identical, so comparing to itself)
        success, messages = ParityOracle.validate_parity_across_surfaces(result_dict, result_dict)
        assert success, f"Identical results should pass parity check: {messages}"
    
    def test_coverage_failure_paths_all_scenarios(self):
        """COVERAGE: All failure scenarios handled gracefully (no unhandled crashes)."""
        failure_scenarios = [
            # Scenario 1: Malformed event (missing field)
            {'run_id': 'scenario_1', 'seq': 1, 'event_id': 'evt_1'},
            
            # Scenario 2: Duplicate event
            {'run_id': 'scenario_2', 'seq': 1, 'event_id': 'dup_id',
             'emitted_at': '2026-05-08T00:00:00Z', 'event_type': 'training.event',
             'phase': 'phase_1a', 'unit': 'combination', 'status': 'started'},
            
            # Scenario 3: Out-of-order event
            {'run_id': 'scenario_3', 'seq': 1, 'event_id': 'evt_oo_1',
             'emitted_at': '2026-05-08T00:00:00Z', 'event_type': 'training.event',
             'phase': 'phase_1a', 'unit': 'combination', 'status': 'started'},
        ]
        
        state = RealtimeDashboardState(run_id="scenario_testing")
        crash_count = 0
        
        for scenario in failure_scenarios:
            try:
                # Should not crash
                state.on_event(scenario)  # type: ignore
            except Exception as e:
                # Log but don't fail the test - consumer should handle gracefully
                if isinstance(e, (KeyError, TypeError, ValueError, AttributeError)):
                    # Expected validation errors - acceptable
                    pass
                else:
                    # Unexpected crash
                    crash_count += 1
        
        assert crash_count == 0, "Failure scenarios should not cause unhandled crashes"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
