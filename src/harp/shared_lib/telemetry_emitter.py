"""
Transport-agnostic telemetry emitter for training progress events.

Provides structured event emission with optional callback subscription.
Events include required fields: run_id, seq, event_id, emitted_at, event_type, phase, unit, status.

Design:
- Callback is optional. If no callback, events are generated but not sent anywhere.
- Errors in callbacks are isolated and logged, not propagated.
- Multiple subscribers can be registered.
- Sequence numbers are monotonically increasing per run.
"""

import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, List, Any, TypedDict
import logging

logger = logging.getLogger(__name__)


class ProgressEvent(TypedDict, total=False):
    """Event envelope for training progress telemetry."""
    
    # Required fields
    run_id: str
    seq: int
    event_id: str
    emitted_at: str  # ISO-8601 UTC
    event_type: str
    phase: str
    unit: str
    status: str
    
    # Optional fields
    current: Optional[int]
    total: Optional[int]
    progress_pct: Optional[float]
    model: Optional[str]
    preprocessor: Optional[str]
    message: Optional[str]
    level: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    metrics: Optional[Dict[str, Any]]
    cv_folds: Optional[int]
    top_percent: Optional[float]


class TelemetryEmitter:
    """
    Emits structured progress events with optional callback subscription.
    
    Example:
        emitter = TelemetryEmitter(run_id="train_001")
        emitter.subscribe(callback_fn)
        emitter.emit_event(
            event_type="training.phase.started",
            phase="phase_1a_screening",
            unit="combination",
            status="started",
            current=0,
            total=12
        )
    """
    
    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize emitter.
        
        Args:
            run_id: Unique run identifier. Generated if not provided.
        """
        self.run_id = run_id or str(uuid.uuid4())
        self._seq = 0
        self._subscribers: List[Callable[[ProgressEvent], None]] = []
        self._events: List[ProgressEvent] = []  # Bounded buffer for event retrieval
        self._max_events = 500  # Keep last 500 events for polling
        self._lock = None  # Can be upgraded to threading.Lock if needed for concurrency
    
    def subscribe(self, callback: Callable[[ProgressEvent], None]) -> None:
        """
        Subscribe callback to receive all emitted events.
        
        Args:
            callback: Function that accepts a ProgressEvent dict.
                     Errors in callback will be logged but not propagated.
        """
        self._subscribers.append(callback)
    
    def emit_event(
        self,
        event_type: str,
        phase: str,
        unit: str,
        status: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
        model: Optional[str] = None,
        preprocessor: Optional[str] = None,
        message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        level: Optional[str] = None,
        cv_folds: Optional[int] = None,
        top_percent: Optional[float] = None,
    ) -> ProgressEvent:
        """
        Emit a structured event.
        
        Args:
            event_type: Event type (e.g., 'training.run.started', 'training.phase.started')
            phase: Phase identifier (e.g., 'init', 'phase_1a_screening', 'phase_1b_full_cv')
            unit: Unit type ('run', 'phase', 'combination', 'candidate', 'fold', 'trial')
            status: Status ('started', 'in_progress', 'completed', 'failed')
            current: Current position in sequence (optional)
            total: Total count (optional)
            model: Model name (optional, for combination-scoped events)
            preprocessor: Preprocessor name (optional)
            message: Human-readable message (optional)
            metrics: Dictionary of metrics (optional, e.g., {'auc': 0.95, 'fit_time': 2.5})
            error_type: Exception type for failures (optional)
            error_message: Sanitized exception message (optional)
            level: Log level ('info', 'warning', 'error') (optional)
            cv_folds: Number of CV folds (optional)
            top_percent: Top percent fraction (optional)
        
        Returns:
            The emitted ProgressEvent dict.
        """
        # Increment sequence
        self._seq += 1
        
        # Generate event ID
        event_id = f"{self.run_id}_{self._seq}_{uuid.uuid4().hex[:8]}"
        
        # Generate timestamp
        emitted_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Build event envelope
        event: ProgressEvent = {
            'run_id': self.run_id,
            'seq': self._seq,
            'event_id': event_id,
            'emitted_at': emitted_at,
            'event_type': event_type,
            'phase': phase,
            'unit': unit,
            'status': status,
        }
        
        # Add optional fields if provided
        if current is not None:
            event['current'] = current
        if total is not None:
            event['total'] = total
        if current is not None and total is not None and total > 0:
            event['progress_pct'] = round(100.0 * current / total, 1)
        if model is not None:
            event['model'] = model
        if preprocessor is not None:
            event['preprocessor'] = preprocessor
        if message is not None:
            event['message'] = message
        if level is not None:
            event['level'] = level
        if error_type is not None:
            event['error_type'] = error_type
        if error_message is not None:
            event['error_message'] = error_message
        if metrics is not None:
            event['metrics'] = metrics
        if cv_folds is not None:
            event['cv_folds'] = cv_folds
        if top_percent is not None:
            event['top_percent'] = top_percent
        
        # Dispatch to subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception as e:
                logger.error(
                    f"Error in telemetry subscriber: {type(e).__name__}: {str(e)[:100]}",
                    exc_info=False
                )
        
        # Store in bounded buffer
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events.pop(0)
        
        return event
    
    def get_events(self, limit: Optional[int] = None) -> List[ProgressEvent]:
        """
        Get recent events from the buffer.
        
        Args:
            limit: Maximum number of events to return (None = all).
        
        Returns:
            List of recent events, oldest first.
        """
        if limit is None:
            return list(self._events)
        return list(self._events[-limit:]) if self._events else []


# Module-level state for convenience functions
_default_emitter: Optional[TelemetryEmitter] = None


def create_emitter(run_id: Optional[str] = None) -> TelemetryEmitter:
    """
    Create a new telemetry emitter.
    
    Args:
        run_id: Optional run identifier. Generated if not provided.
    
    Returns:
        New TelemetryEmitter instance.
    """
    global _default_emitter
    _default_emitter = TelemetryEmitter(run_id=run_id)
    return _default_emitter


def get_emitter() -> TelemetryEmitter:
    """
    Get the current module-level emitter.
    
    Returns:
        Current TelemetryEmitter instance.
    
    Raises:
        RuntimeError: If no emitter has been created yet.
    """
    global _default_emitter
    if _default_emitter is None:
        raise RuntimeError("No emitter created. Call create_emitter() first.")
    return _default_emitter


def emit_event(
    event_type: str,
    phase: str,
    unit: str,
    status: str,
    **kwargs
) -> ProgressEvent:
    """
    Emit event using module-level emitter.
    
    Args:
        event_type: Event type
        phase: Phase identifier
        unit: Unit type
        status: Status
        **kwargs: Additional event fields
    
    Returns:
        The emitted ProgressEvent dict.
    """
    emitter = get_emitter()
    return emitter.emit_event(event_type=event_type, phase=phase, unit=unit, status=status, **kwargs)
