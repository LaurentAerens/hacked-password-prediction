"""
Streamlit realtime training dashboard consumer.

Consumes structured telemetry events with:
- Event ordering/idempotency (run_id + seq + event_id)
- Status board per model+preprocessor
- Bounded log buffer (ring buffer for events)
- Render cursor to prevent duplicate rendering
- Stale run prevention
"""

from typing import Dict, List, Any, Optional
from shared_lib.telemetry_emitter import ProgressEvent


class RealtimeDashboardState:
    """
    Manages realtime dashboard state from telemetry events.
    
    Thread-safe consumer that:
    - Tracks current run_id
    - Maintains status board: {model: {preprocessor: {state, progress, metrics}}}
    - Keeps bounded event buffer for log display
    - Prevents duplicate rendering via event_id deduplication
    - Enforces monotonic seq ordering
    - Rejects stale events from old runs
    
    Use pattern:
    >>> state = RealtimeDashboardState(run_id="train_001")
    >>> def event_callback(event):
    ...     state.on_event(event)
    >>> # Pass event_callback to train_with_adaptive_search
    >>> # Then periodically call:
    >>> summary = state.get_status_summary()
    >>> logs = state.get_recent_logs(count=50)
    """
    
    def __init__(self, run_id: str, buffer_size: int = 100):
        """
        Initialize dashboard state.
        
        Args:
            run_id: Unique run identifier to track
            buffer_size: Maximum number of events to retain in buffer
        """
        self.run_id = run_id
        self.buffer_size = buffer_size
        
        # Status board: {model: {preprocessor: {state, progress, metrics}}}
        self.status_board: Dict[str, Dict[str, Any]] = {}
        
        # Bounded ring buffer for events
        self.event_buffer: List[ProgressEvent] = []
        
        # Render cursor: last seq number that updated the board
        self.render_cursor = 0
        
        # Dedupe set to prevent duplicate rendering
        self.dedupe_set: set = set()
        
        # Phase tracking
        self.phase: Optional[str] = None
        self.phase_progress: Optional[Dict[str, int]] = None
        
        # Run-level status
        self.run_status: Optional[str] = None
    
    def on_event(self, event: ProgressEvent) -> bool:
        """
        Process an event.
        
        Returns True if event was new and updated the status board.
        Returns False if event was stale, duplicate, or out-of-order.
        
        Processing rules:
        - Reject if run_id doesn't match current run
        - Reject if event_id in dedupe_set
        - Accept and buffer (even if seq is out-of-order)
        - Only update status board if seq > render_cursor (monotonic)
        
        Args:
            event: ProgressEvent from telemetry emitter
        
        Returns:
            True if status board was updated, False otherwise
        """
        event_id = event.get('event_id')
        event_seq = event.get('seq', 0)
        event_run_id = event.get('run_id')
        
        # Rule 1: Stale run check
        if event_run_id != self.run_id:
            return False
        
        # Rule 2: Duplicate check
        if event_id in self.dedupe_set:
            return False
        
        self.dedupe_set.add(event_id)
        
        # Rule 3: Add to bounded buffer (always, even if out-of-order)
        if len(self.event_buffer) >= self.buffer_size:
            # Remove oldest event
            self.event_buffer.pop(0)
        
        self.event_buffer.append(event)
        
        # Rule 4: Update status board only if seq is monotonic
        if event_seq > self.render_cursor:
            self.render_cursor = event_seq
            self._update_status_board(event)
            return True
        
        return False
    
    def _update_status_board(self, event: ProgressEvent) -> None:
        """Update internal state from event."""
        event_type = event.get('event_type', '')
        status = event.get('status', 'unknown')
        
        # Track run status
        if 'run.started' in event_type:
            self.run_status = 'running'
        elif 'run.completed' in event_type:
            self.run_status = 'completed'
        elif 'run.failed' in event_type or 'failed' in event_type:
            self.run_status = 'failed'
        
        # Track phase
        if 'phase.started' in event_type:
            self.phase = event.get('phase')
            self.phase_progress = {
                'current': event.get('current', 0),
                'total': event.get('total', 1),
            }
        elif 'phase.completed' in event_type:
            self.phase = event.get('phase')
            self.phase_progress = {
                'current': event.get('total', 1),
                'total': event.get('total', 1),
            }
        
        # Track candidate/combination progress
        if 'candidate' in event_type or 'combination' in event_type:
            model = event.get('model', 'unknown')
            preprocessor = event.get('preprocessor', 'unknown')
            metrics = event.get('metrics', {})

            # Keep top-level phase progress in sync with candidate completion events.
            event_current = event.get('current')
            event_total = event.get('total')
            if event_current is not None and event_total is not None:
                self.phase_progress = {
                    'current': event_current,
                    'total': event_total,
                }
                if event.get('phase'):
                    self.phase = event.get('phase')
            
            if model not in self.status_board:
                self.status_board[model] = {}
            
            self.status_board[model][preprocessor] = {
                'status': status,
                'state': status,
                'progress': {
                    'current': event.get('current'),
                    'total': event.get('total'),
                },
                'metrics': metrics,
                'phase': event.get('phase'),
                'unit': event.get('unit'),
            }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get current dashboard status for rendering.
        
        Returns dict with:
        - run_id: current run identifier
        - run_status: 'running', 'completed', 'failed', or None
        - phase: current phase label (e.g., 'phase_1a_screening')
        - phase_progress: {current, total} for current phase
        - status_board: {model: {preprocessor: {state, progress, metrics}}}
        - event_count: number of events in buffer
        - buffer_size: max buffer size
        """
        return {
            'run_id': self.run_id,
            'run_status': self.run_status,
            'phase': self.phase,
            'phase_progress': self.phase_progress,
            'status_board': self.status_board,
            'event_count': len(self.event_buffer),
            'buffer_size': self.buffer_size,
        }
    
    def get_recent_logs(self, count: Optional[int] = None) -> List[str]:
        """
        Get recent events formatted as log lines.
        
        Args:
            count: Number of recent events to return. If None, return all in buffer.
        
        Returns:
            List of formatted event strings
        """
        events = self.event_buffer[-count:] if count else self.event_buffer
        
        logs = []
        for event in events:
            phase = event.get('phase', '')
            event_type = event.get('event_type', '')
            unit = event.get('unit', '')
            status = event.get('status', '')
            model = event.get('model', '')
            preprocessor = event.get('preprocessor', '')
            current = event.get('current')
            total = event.get('total')
            message = event.get('message', '')
            metrics = event.get('metrics', {})
            
            # Format based on event type
            if 'phase' in event_type:
                if current is not None and total is not None:
                    log = f"[{phase}] {status.upper()} ({current}/{total})"
                else:
                    log = f"[{phase}] {status.upper()}"
            elif 'candidate' in event_type or 'combination' in event_type:
                if model and preprocessor:
                    metrics_str = " | ".join(f"{k}={v}" for k, v in metrics.items())
                    detail = message or ""
                    if current is not None and total is not None:
                        if detail:
                            log = f"  {model} + {preprocessor}: {status.upper()} ({current}/{total}) | {detail}"
                            if metrics_str:
                                log += f" | {metrics_str}"
                        else:
                            log = f"  {model} + {preprocessor}: {status.upper()} ({current}/{total})"
                            if metrics_str:
                                log += f" | {metrics_str}"
                    else:
                        log = f"  {model} + {preprocessor}: {status.upper()}"
                        if detail:
                            log += f" | {detail}"
                        if metrics_str:
                            log += f" | {metrics_str}"
                else:
                    log = f"  {status.upper()}"
            else:
                log = f"[{event_type}] {status.upper()}"
            
            logs.append(log)
        
        return logs
    
    def get_phase_label(self) -> str:
        """Get human-readable phase label (e.g., '1a' or '1b')."""
        if not self.phase:
            return ""
        if 'phase_1a' in self.phase:
            return "1a"
        elif 'phase_1b' in self.phase:
            return "1b"
        return self.phase
    
    def get_combined_status(self) -> Dict[str, Any]:
        """
        Get combined status for compact rendering.
        
        Returns:
        {
            'phase': '1a' or '1b',
            'status': 'queued', 'running', 'completed', or 'failed',
            'queued_count': number of queued combinations
            'running_count': number of running combinations
            'completed_count': number of completed combinations
            'failed_count': number of failed combinations
            'total_count': total combinations across all models+preprocessors
        }
        """
        queued = 0
        running = 0
        completed = 0
        failed = 0
        
        for model_dict in self.status_board.values():
            for combo_state in model_dict.values():
                state = combo_state.get('state', 'unknown')
                if state == 'started' or state == 'in_progress':
                    running += 1
                elif state == 'completed':
                    completed += 1
                elif state == 'failed':
                    failed += 1
                else:
                    queued += 1
        
        total = queued + running + completed + failed
        
        return {
            'phase': self.get_phase_label(),
            'status': self.run_status or 'unknown',
            'queued_count': queued,
            'running_count': running,
            'completed_count': completed,
            'failed_count': failed,
            'total_count': total,
        }
