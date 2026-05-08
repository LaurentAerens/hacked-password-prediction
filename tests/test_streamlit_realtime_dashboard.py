"""
Tests for Streamlit realtime training dashboard.

Validates:
- Event ordering and idempotency (run_id + seq + event_id)
- Status board per model+preprocessor
- Progress counters and phase labels
- Bounded log buffer
- Stale run prevention
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add ai-resources to path
ai_resources_path = str(Path(__file__).parent.parent / "ai-resources")
sys.path.insert(0, ai_resources_path)

from shared_lib.telemetry_emitter import ProgressEvent


class RealtimeDashboardState:
    """
    Simulates Streamlit session_state for realtime dashboard.
    
    Tracks:
    - Current run_id
    - Status board: {model_name: {preprocessor: {state, progress, metrics}}}
    - Event buffer: bounded log of events
    - Render cursor: seq number of last rendered event
    - Dedupe set: {event_id} to prevent duplicate rendering
    """
    
    def __init__(self, run_id: str, buffer_size: int = 100):
        self.run_id = run_id
        self.buffer_size = buffer_size
        self.status_board: Dict[str, Dict[str, Any]] = {}  # {model: {preprocessor: state}}
        self.event_buffer: List[ProgressEvent] = []  # Bounded ring buffer
        self.render_cursor = 0  # Last rendered seq number
        self.dedupe_set: set = set()  # {event_id}
        self.phase = None  # Current phase label
        self.phase_progress = None  # {"current": int, "total": int}
    
    def on_event(self, event: ProgressEvent) -> bool:
        """
        Process an event. Returns True if event was new and processed.
        Prevents duplicate rendering via event_id deduplication.
        """
        event_id = event.get('event_id')
        
        # Stale run check
        if event.get('run_id') != self.run_id:
            return False
        
        # Duplicate check
        if event_id in self.dedupe_set:
            return False
        
        self.dedupe_set.add(event_id)
        
        # Bounded buffer: remove oldest if at capacity
        if len(self.event_buffer) >= self.buffer_size:
            self.event_buffer.pop(0)
        
        self.event_buffer.append(event)
        
        # Update status board and render cursor only if seq is monotonic (no out-of-order)
        event_seq = event.get('seq', 0)
        if event_seq > self.render_cursor:
            self.render_cursor = event_seq
            self._update_status_board(event)
            return True
        
        return False
    
    def _update_status_board(self, event: ProgressEvent) -> None:
        """Update internal status board from event."""
        event_type = event.get('event_type', '')
        
        # Phase tracking
        if 'phase.started' in event_type:
            self.phase = event.get('phase')
            self.phase_progress = {
                'current': event.get('current', 0),
                'total': event.get('total', 1),
            }
        elif 'phase.completed' in event_type:
            self.phase_progress = {
                'current': event.get('total', 1),
                'total': event.get('total', 1),
            }
        
        # Candidate/combination tracking
        if 'candidate' in event_type or 'combination' in event_type:
            model = event.get('model', 'unknown')
            preprocessor = event.get('preprocessor', 'unknown')
            key = f"{model}+{preprocessor}"
            
            if model not in self.status_board:
                self.status_board[model] = {}
            
            status = event.get('status', 'unknown')
            metrics = event.get('metrics', {})
            
            self.status_board[model][preprocessor] = {
                'state': status,
                'progress': {
                    'current': event.get('current'),
                    'total': event.get('total'),
                },
                'metrics': metrics,
            }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current status for rendering."""
        return {
            'run_id': self.run_id,
            'phase': self.phase,
            'phase_progress': self.phase_progress,
            'status_board': self.status_board,
            'event_count': len(self.event_buffer),
            'buffer_size': self.buffer_size,
        }


def test_basic_event_processing():
    """Test basic event processing and deduplication."""
    print("Test 1: Basic event processing...")
    
    state = RealtimeDashboardState(run_id="run_123")
    
    event1: ProgressEvent = {
        'run_id': 'run_123',
        'seq': 1,
        'event_id': 'evt_1',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.run.started',
        'phase': 'init',
        'unit': 'run',
        'status': 'started',
    }
    
    # First event should be processed
    assert state.on_event(event1) == True
    assert len(state.event_buffer) == 1
    assert state.render_cursor == 1
    
    # Duplicate should be rejected
    assert state.on_event(event1) == False
    assert len(state.event_buffer) == 1
    
    print("✓ PASSED")


def test_stale_run_prevention():
    """Test that events from old runs are rejected."""
    print("Test 2: Stale run prevention...")
    
    state = RealtimeDashboardState(run_id="run_123")
    
    # Event from different run should be rejected
    stale_event: ProgressEvent = {
        'run_id': 'run_456',
        'seq': 1,
        'event_id': 'evt_old',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.run.started',
        'phase': 'init',
        'unit': 'run',
        'status': 'started',
    }
    
    assert state.on_event(stale_event) == False
    assert len(state.event_buffer) == 0
    
    print("✓ PASSED")


def test_monotonic_seq_ordering():
    """Test that out-of-order events don't update render cursor."""
    print("Test 3: Monotonic sequence ordering...")
    
    state = RealtimeDashboardState(run_id="run_123")
    
    # Event 2
    event2: ProgressEvent = {
        'run_id': 'run_123',
        'seq': 2,
        'event_id': 'evt_2',
        'emitted_at': '2026-05-08T12:00:01Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1a',
        'unit': 'phase',
        'status': 'started',
        'current': 0,
        'total': 12,
    }
    
    # Event 1 (out of order, arrives after event 2)
    event1: ProgressEvent = {
        'run_id': 'run_123',
        'seq': 1,
        'event_id': 'evt_1',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.run.started',
        'phase': 'init',
        'unit': 'run',
        'status': 'started',
    }
    
    # Process event 2
    result2 = state.on_event(event2)
    assert result2 == True, "Event 2 should be processed"
    assert state.render_cursor == 2
    assert len(state.event_buffer) == 1
    
    # Process event 1 (out of order)
    result1 = state.on_event(event1)
    assert result1 == False, "Out-of-order event 1 should not update cursor"
    assert state.render_cursor == 2, "Render cursor should stay at 2"
    assert len(state.event_buffer) == 2, "Both events in buffer, but only 2 rendered"
    
    print("✓ PASSED")


def test_status_board_updates():
    """Test that status board is updated correctly."""
    print("Test 4: Status board updates...")
    
    state = RealtimeDashboardState(run_id="run_123")
    
    # Phase started
    phase_event: ProgressEvent = {
        'run_id': 'run_123',
        'seq': 1,
        'event_id': 'evt_1',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1a_screening',
        'unit': 'phase',
        'status': 'started',
        'current': 0,
        'total': 12,
    }
    
    state.on_event(phase_event)
    assert state.phase == 'phase_1a_screening'
    assert state.phase_progress['current'] == 0
    assert state.phase_progress['total'] == 12
    
    # Candidate completed
    candidate_event: ProgressEvent = {
        'run_id': 'run_123',
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
        'total': 12,
        'metrics': {'auc': 0.95},
    }
    
    state.on_event(candidate_event)
    assert 'xgb' in state.status_board
    assert 'tfidf' in state.status_board['xgb']
    assert state.status_board['xgb']['tfidf']['state'] == 'completed'
    assert state.status_board['xgb']['tfidf']['metrics'] == {'auc': 0.95}
    
    print("✓ PASSED")


def test_bounded_buffer():
    """Test that event buffer is bounded."""
    print("Test 5: Bounded event buffer...")
    
    state = RealtimeDashboardState(run_id="run_123", buffer_size=5)
    
    for i in range(10):
        event: ProgressEvent = {
            'run_id': 'run_123',
            'seq': i + 1,
            'event_id': f'evt_{i}',
            'emitted_at': '2026-05-08T12:00:00Z',
            'event_type': 'test.event',
            'phase': 'test',
            'unit': 'test',
            'status': 'started',
        }
        state.on_event(event)
    
    assert len(state.event_buffer) == 5, f"Buffer should stay at 5, got {len(state.event_buffer)}"
    # Oldest events (1-5) should be removed, newest events (6-10) should be in buffer
    first_seq = state.event_buffer[0]['seq']
    assert first_seq == 6, f"First event in buffer should have seq 6, got {first_seq}"
    last_seq = state.event_buffer[-1]['seq']
    assert last_seq == 10, f"Last event in buffer should have seq 10, got {last_seq}"
    
    print("✓ PASSED")


def test_progress_labels():
    """Test that progress labels (phase 1a/1b, fold, candidate) are tracked."""
    print("Test 6: Progress labels and counters...")
    
    state = RealtimeDashboardState(run_id="run_123")
    
    # Phase 1a started
    phase_1a: ProgressEvent = {
        'run_id': 'run_123',
        'seq': 1,
        'event_id': 'evt_1',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1a_screening',
        'unit': 'phase',
        'status': 'started',
        'current': 0,
        'total': 20,
        'cv_folds': 2,
    }
    
    state.on_event(phase_1a)
    summary = state.get_status_summary()
    assert summary['phase'] == 'phase_1a_screening'
    assert '1a' in summary['phase']
    
    # Phase 1b started
    phase_1b: ProgressEvent = {
        'run_id': 'run_123',
        'seq': 30,
        'event_id': 'evt_30',
        'emitted_at': '2026-05-08T12:00:10Z',
        'event_type': 'training.phase.started',
        'phase': 'phase_1b_full_cv',
        'unit': 'phase',
        'status': 'started',
        'current': 0,
        'total': 4,
        'cv_folds': 5,
    }
    
    state.on_event(phase_1b)
    summary = state.get_status_summary()
    assert summary['phase'] == 'phase_1b_full_cv'
    assert '1b' in summary['phase']
    
    print("✓ PASSED")


def test_fallback_without_callback():
    """Test that status is accessible even without live updates."""
    print("Test 7: Fallback behavior (no-callback mode)...")
    
    # Initialize state without any subscriptions/callbacks
    state = RealtimeDashboardState(run_id="run_123")
    
    # Should have no errors accessing status
    summary = state.get_status_summary()
    assert summary['run_id'] == 'run_123'
    assert summary['event_count'] == 0
    
    # After events, should return valid status
    event: ProgressEvent = {
        'run_id': 'run_123',
        'seq': 1,
        'event_id': 'evt_1',
        'emitted_at': '2026-05-08T12:00:00Z',
        'event_type': 'training.run.started',
        'phase': 'init',
        'unit': 'run',
        'status': 'started',
    }
    
    state.on_event(event)
    summary = state.get_status_summary()
    assert summary['event_count'] == 1
    
    print("✓ PASSED")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Streamlit Realtime Dashboard")
    print("=" * 70)
    
    test_basic_event_processing()
    test_stale_run_prevention()
    test_monotonic_seq_ordering()
    test_status_board_updates()
    test_bounded_buffer()
    test_progress_labels()
    test_fallback_without_callback()
    
    print("=" * 70)
    print("✓ ALL TESTS PASSED (7/7)")
    print("=" * 70)
