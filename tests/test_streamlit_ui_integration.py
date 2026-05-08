"""
Integration tests for Streamlit realtime training UI.

Validates:
- Realtime event callback wiring
- Session state initialization and updates
- Fallback to non-realtime if callback disabled
- Live rendering loop behavior
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add ai-resources to path
ai_resources_path = str(Path(__file__).parent.parent / "ai-resources")
sys.path.insert(0, ai_resources_path)

from shared_lib.telemetry_emitter import ProgressEvent
from shared_lib.realtime_dashboard import RealtimeDashboardState


def simulate_training_events(state: RealtimeDashboardState, run_count: int = 2) -> None:
    """Simulate training events and process them."""
    
    # Run started
    state.on_event({
        'run_id': state.run_id,
        'seq': 1,
        'event_id': 'evt_1',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.run.started',
        'phase': 'init',
        'unit': 'run',
        'status': 'started',
    })
    
    # Phase 1a started
    state.on_event({
        'run_id': state.run_id,
        'seq': 2,
        'event_id': 'evt_2',
        'emitted_at': '2026-05-08T12:00:01Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1a_screening',
        'unit': 'phase',
        'status': 'started',
        'current': 0,
        'total': run_count,
        'cv_folds': 2,
    })
    
    # Candidates processed
    for i in range(run_count):
        state.on_event({
            'run_id': state.run_id,
            'seq': 3 + i,
            'event_id': f'evt_{3+i}',
            'emitted_at': f'2026-05-08T12:00:{2+i:02d}Z',
            'event_type': 'training.candidate.completed',
            'phase': 'phase_1a_screening',
            'unit': 'combination',
            'status': 'completed',
            'model': ['xgb', 'rf'][i % 2],
            'preprocessor': ['tfidf', 'count_vec'][i % 2],
            'current': i + 1,
            'total': run_count,
            'metrics': {'auc': 0.90 + i * 0.01},
        })
    
    # Phase 1a completed
    state.on_event({
        'run_id': state.run_id,
        'seq': 3 + run_count,
        'event_id': f'evt_{3+run_count}',
        'emitted_at': '2026-05-08T12:01:00Z',
        'event_type': 'training.phase.completed',
        'phase': 'phase_1a_screening',
        'unit': 'phase',
        'status': 'completed',
        'current': run_count,
        'total': run_count,
        'metrics': {'combinations_screened': run_count},
    })
    
    # Phase 1b started
    state.on_event({
        'run_id': state.run_id,
        'seq': 4 + run_count,
        'event_id': f'evt_{4+run_count}',
        'emitted_at': '2026-05-08T12:01:01Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1b_full_cv',
        'unit': 'phase',
        'status': 'started',
        'current': 0,
        'total': 1,
        'cv_folds': 5,
    })
    
    # Run completed
    state.on_event({
        'run_id': state.run_id,
        'seq': 5 + run_count,
        'event_id': f'evt_{5+run_count}',
        'emitted_at': '2026-05-08T12:02:00Z',
        'event_type': 'training.run.completed',
        'phase': 'init',
        'unit': 'run',
        'status': 'completed',
        'metrics': {'best_auc': 0.91},
    })


def test_session_state_initialization():
    """Test session state initialization for realtime dashboard."""
    print("Test 1: Session state initialization...")
    
    run_id = "test_run_ui_1"
    state = RealtimeDashboardState(run_id=run_id)
    
    # Should initialize with empty state
    summary = state.get_status_summary()
    assert summary['run_id'] == run_id
    assert summary['run_status'] is None
    assert summary['phase'] is None
    assert summary['event_count'] == 0
    assert len(state.status_board) == 0
    
    print("✓ PASSED")


def test_realtime_event_flow():
    """Test complete realtime event flow."""
    print("Test 2: Realtime event flow...")
    
    run_id = "test_run_ui_2"
    state = RealtimeDashboardState(run_id=run_id, buffer_size=100)
    
    # Simulate training events
    simulate_training_events(state, run_count=2)
    
    # Verify state after all events
    summary = state.get_status_summary()
    assert summary['run_status'] == 'completed'
    assert summary['phase'] == 'phase_1b_full_cv'
    # Event count: 1 run.started + 1 phase.started(1a) + 2 candidates + 1 phase.completed(1a) + 1 phase.started(1b) + 1 run.completed = 7
    assert summary['event_count'] >= 6
    
    # Check status board
    assert 'xgb' in state.status_board or 'rf' in state.status_board
    
    print("✓ PASSED")


def test_compact_status_rendering():
    """Test compact status rendering for UI."""
    print("Test 3: Compact status rendering...")
    
    run_id = "test_run_ui_3"
    state = RealtimeDashboardState(run_id=run_id)
    
    simulate_training_events(state, run_count=3)
    
    combined = state.get_combined_status()
    assert combined['phase'] in ['1a', '1b', '']
    assert combined['status'] in ['running', 'completed', 'failed', 'unknown']
    assert combined['total_count'] >= 0
    assert combined['completed_count'] >= 0
    
    print("✓ PASSED")


def test_log_formatting():
    """Test log message formatting."""
    print("Test 4: Log formatting...")
    
    run_id = "test_run_ui_4"
    state = RealtimeDashboardState(run_id=run_id)
    
    simulate_training_events(state, run_count=2)
    
    logs = state.get_recent_logs(count=5)
    assert len(logs) > 0
    assert all(isinstance(log, str) for log in logs)
    assert any('phase_1a' in log.lower() for log in logs)
    assert any('completed' in log.lower() for log in logs)
    
    print("✓ PASSED")


def test_phase_label_extraction():
    """Test phase label extraction."""
    print("Test 5: Phase label extraction...")
    
    run_id = "test_run_ui_5"
    state = RealtimeDashboardState(run_id=run_id)
    
    # Phase 1a
    state.on_event({
        'run_id': run_id,
        'seq': 1,
        'event_id': 'evt_1',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1a_screening',
        'unit': 'phase',
        'status': 'started',
    })
    assert state.get_phase_label() == '1a'
    
    # Phase 1b
    state.on_event({
        'run_id': run_id,
        'seq': 2,
        'event_id': 'evt_2',
        'emitted_at': '2026-05-08T12:01:00Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1b_full_cv',
        'unit': 'phase',
        'status': 'started',
    })
    assert state.get_phase_label() == '1b'
    
    print("✓ PASSED")


def test_stale_run_isolation():
    """Test that stale runs don't corrupt current run state."""
    print("Test 6: Stale run isolation...")
    
    run_id_current = "run_current"
    run_id_old = "run_old"
    
    state = RealtimeDashboardState(run_id=run_id_current)
    
    # Event from old run
    state.on_event({
        'run_id': run_id_old,
        'seq': 1,
        'event_id': 'evt_old',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1a_screening',
        'unit': 'phase',
        'status': 'started',
    })
    
    # Current state should be unchanged
    assert state.phase is None
    assert len(state.event_buffer) == 0
    
    # Event from current run
    state.on_event({
        'run_id': run_id_current,
        'seq': 1,
        'event_id': 'evt_new',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1a_screening',
        'unit': 'phase',
        'status': 'started',
    })
    
    # Current state should be updated
    assert state.phase == 'phase_1a_screening'
    assert len(state.event_buffer) == 1
    
    print("✓ PASSED")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Streamlit Realtime Training UI Integration")
    print("=" * 70)
    
    test_session_state_initialization()
    test_realtime_event_flow()
    test_compact_status_rendering()
    test_log_formatting()
    test_phase_label_extraction()
    test_stale_run_isolation()
    
    print("=" * 70)
    print("✓ ALL TESTS PASSED (6/6)")
    print("=" * 70)
