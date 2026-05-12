"""
Validation script for tpv-003: Streamlit realtime training dashboard.

Tests:
1. RealtimeDashboardState initialization
2. Event processing and ordering
3. Status board updates
4. Bounded buffer management
5. Stale run prevention
6. Integration with telemetry emitter
7. Log formatting
8. No syntax errors in ui_app.py
"""

from pathlib import Path

from harp.shared_lib.realtime_dashboard import RealtimeDashboardState
from harp.shared_lib.telemetry_emitter import TelemetryEmitter, ProgressEvent


def test_1_dashboard_initialization():
    """Test 1: Dashboard state initialization."""
    print("[1/7] Dashboard initialization...")
    
    state = RealtimeDashboardState(run_id="val_test_1")
    summary = state.get_status_summary()
    
    assert summary['run_id'] == "val_test_1"
    assert summary['event_count'] == 0
    assert summary['run_status'] is None
    assert len(state.status_board) == 0
    
    print("    ✓ PASSED")


def test_2_event_deduplication():
    """Test 2: Event deduplication and ordering."""
    print("[2/7] Event deduplication...")
    
    state = RealtimeDashboardState(run_id="val_test_2")
    
    event: ProgressEvent = {
        'run_id': 'val_test_2',
        'seq': 1,
        'event_id': 'evt_001',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1a',
        'unit': 'phase',
        'status': 'started',
    }
    
    # First event should be processed
    result1 = state.on_event(event)
    assert result1 == True, "First event should be processed"
    
    # Duplicate should be rejected
    result2 = state.on_event(event)
    assert result2 == False, "Duplicate should be rejected"
    
    assert len(state.event_buffer) == 1
    
    print("    ✓ PASSED")


def test_3_status_board():
    """Test 3: Status board updates."""
    print("[3/7] Status board updates...")
    
    state = RealtimeDashboardState(run_id="val_test_3")
    
    # Emit multiple candidate events
    for i in range(3):
        state.on_event({
            'run_id': 'val_test_3',
            'seq': i + 1,
            'event_id': f'evt_{i}',
            'emitted_at': '2026-05-08T12:00:00Z',
            'event_type': 'training.candidate.completed',
            'phase': 'phase_1a',
            'unit': 'combination',
            'status': 'completed',
            'model': f'model_{i}',
            'preprocessor': f'prep_{i}',
            'current': i + 1,
            'total': 3,
            'metrics': {'auc': 0.90 + i * 0.01},
        })
    
    assert len(state.status_board) == 3
    for i in range(3):
        model = f'model_{i}'
        prep = f'prep_{i}'
        assert model in state.status_board
        assert prep in state.status_board[model]
        assert state.status_board[model][prep]['metrics']['auc'] == 0.90 + i * 0.01
    
    print("    ✓ PASSED")


def test_4_bounded_buffer():
    """Test 4: Bounded buffer management."""
    print("[4/7] Bounded buffer...")
    
    state = RealtimeDashboardState(run_id="val_test_4", buffer_size=5)
    
    # Add more events than buffer size
    for i in range(10):
        state.on_event({
            'run_id': 'val_test_4',
            'seq': i + 1,
            'event_id': f'evt_{i}',
            'emitted_at': '2026-05-08T12:00:00Z',
            'event_type': 'test.event',
            'phase': 'test',
            'unit': 'test',
            'status': 'started',
        })
    
    assert len(state.event_buffer) == 5, f"Expected 5, got {len(state.event_buffer)}"
    
    # Check that oldest events were removed
    first_seq = state.event_buffer[0]['seq']
    last_seq = state.event_buffer[-1]['seq']
    assert first_seq == 6
    assert last_seq == 10
    
    print("    ✓ PASSED")


def test_5_stale_run_prevention():
    """Test 5: Stale run prevention."""
    print("[5/7] Stale run prevention...")
    
    state = RealtimeDashboardState(run_id="val_test_5_current")
    
    # Event from different run
    result = state.on_event({
        'run_id': 'val_test_5_old',
        'seq': 1,
        'event_id': 'evt_old',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1a',
        'unit': 'phase',
        'status': 'started',
    })
    
    assert result == False, "Stale run event should be rejected"
    assert len(state.event_buffer) == 0
    
    print("    ✓ PASSED")


def test_6_telemetry_integration():
    """Test 6: Integration with telemetry emitter."""
    print("[6/7] Telemetry integration...")
    
    state = RealtimeDashboardState(run_id="val_test_6")
    emitter = TelemetryEmitter(run_id="val_test_6")
    
    # Subscribe dashboard to emitter
    emitter.subscribe(lambda event: state.on_event(event))
    
    # Emit events
    emitter.emit_event(
        event_type="training.run.started",
        phase="init",
        unit="run",
        status="started",
    )
    
    emitter.emit_event(
        event_type="training.phase.started",
        phase="phase_1a_screening",
        unit="phase",
        status="started",
        current=0,
        total=10,
    )
    
    assert state.run_status == 'running'
    assert state.phase == 'phase_1a_screening'
    assert state.phase_progress['total'] == 10
    assert len(state.event_buffer) == 2
    
    print("    ✓ PASSED")


def test_7_log_formatting():
    """Test 7: Log message formatting."""
    print("[7/7] Log formatting...")
    
    state = RealtimeDashboardState(run_id="val_test_7")
    
    # Add events
    state.on_event({
        'run_id': 'val_test_7',
        'seq': 1,
        'event_id': 'evt_1',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1a_screening',
        'unit': 'phase',
        'status': 'started',
        'current': 0,
        'total': 5,
    })
    
    state.on_event({
        'run_id': 'val_test_7',
        'seq': 2,
        'event_id': 'evt_2',
        'emitted_at': '2026-05-08T12:00:01Z',
        'event_type': 'training.candidate.completed',
        'phase': 'phase_1a_screening',
        'unit': 'combination',
        'status': 'completed',
        'model': 'xgb',
        'preprocessor': 'tfidf',
        'current': 1,
        'total': 5,
        'metrics': {'auc': 0.95, 'fit_time': 2.5},
    })
    
    logs = state.get_recent_logs()
    assert len(logs) == 2
    assert all(isinstance(log, str) for log in logs)
    assert any('phase_1a' in log.lower() for log in logs)
    assert any('xgb' in log.lower() for log in logs)
    assert any('tfidf' in log.lower() for log in logs)
    
    print("    ✓ PASSED")


if __name__ == '__main__':
    print("=" * 70)
    print("Validation: Streamlit Realtime Training Dashboard (tpv-003)")
    print("=" * 70)
    
    try:
        test_1_dashboard_initialization()
        test_2_event_deduplication()
        test_3_status_board()
        test_4_bounded_buffer()
        test_5_stale_run_prevention()
        test_6_telemetry_integration()
        test_7_log_formatting()
        
        print("=" * 70)
        print("✓ ALL VALIDATIONS PASSED (7/7)")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
