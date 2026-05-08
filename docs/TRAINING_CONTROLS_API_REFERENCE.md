# Training Controls: API Reference

## Overview

This document provides complete API reference for the training control system.

## Core Components

### 1. ControlSignal

Thread-safe state machine for pause/stop/resume control.

#### Location
```
ai-resources/shared_lib/control_signal.py
```

#### Class: ControlSignal

```python
class ControlSignal:
    """Thread-safe control signal for pause/stop/resume operations."""
    
    # Valid states
    RUNNING = "RUNNING"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    PAUSED = "PAUSED"
    STOP_REQUESTED = "STOP_REQUESTED"
    STOPPED = "STOPPED"
    
    def __init__(self, run_id: str = None):
        """
        Initialize ControlSignal.
        
        Args:
            run_id: Optional unique ID for this training run
        """
    
    def request_pause(self) -> None:
        """Request graceful pause."""
    
    def request_stop(self) -> None:
        """Request immediate stop."""
    
    def resume(self) -> None:
        """Resume from paused state."""
    
    def should_pause(self) -> bool:
        """
        Poll for pause request.
        
        Returns True exactly once if pause was requested.
        Transitions PAUSE_REQUESTED → PAUSED.
        Called by trainer after each combination.
        """
    
    def should_stop(self) -> bool:
        """
        Poll for stop request.
        
        Returns True exactly once if stop was requested.
        Transitions STOP_REQUESTED → STOPPED.
        Called by trainer after each combination.
        """
    
    def get_state(self) -> str:
        """Get current state string."""
```

#### Usage Example

```python
from shared_lib.control_signal import ControlSignal

# Create signal
signal = ControlSignal(run_id="training-001")

# Request pause
signal.request_pause()

# Check state
if signal.should_pause():
    print("Pausing training...")
    while signal.should_pause() == False:  # Wait in loop
        time.sleep(0.1)

# Resume
signal.resume()

# Stop
signal.request_stop()
if signal.should_stop():
    print("Training stopped")
```

#### Thread Safety

All methods are thread-safe:
- Protected by `threading.Lock()`
- Non-blocking (< 1ms)
- Can be called from multiple threads

---

### 2. CheckpointManager

Manages checkpoint save/load with atomic writes.

#### Location
```
ai-resources/shared_lib/checkpoint_manager.py
```

#### Class: CheckpointManager

```python
class CheckpointManager:
    """Manages checkpoint save/load with atomic writes and validation."""
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory path for storing checkpoints
        """
    
    def save_checkpoint(
        self,
        run_id: str,
        phase: str,
        combination_index: int,
        total_combinations: int,
        results_df: pd.DataFrame,
        best_model: Any,
        best_auc: float,
        best_config: Dict[str, Any],
        error: Optional[str] = None
    ) -> str:
        """
        Save checkpoint atomically to disk.
        
        Uses temp file + rename pattern for atomic writes.
        
        Args:
            run_id: Unique run identifier
            phase: Phase name (e.g., 'phase_1a_screening')
            combination_index: Current combination index
            total_combinations: Total combinations in this phase
            results_df: DataFrame with results so far
            best_model: Best model object
            best_auc: Best AUC score
            best_config: Best configuration dict
            error: Optional error message if checkpoint saved on error
            
        Returns:
            Path to saved checkpoint pkl file
        """
    
    def load_checkpoint(self, run_id: str) -> Dict:
        """
        Load latest checkpoint for run_id.
        
        Returns:
            Dict with keys:
                - phase: Phase name
                - combination_index: Combination index
                - total_combinations: Total combinations
                - results_df: DataFrame
                - best_model: Best model object
                - best_auc: Best AUC
                - best_config: Best config dict
        """
    
    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Validate checkpoint integrity.
        
        Returns:
            True if valid, raises RuntimeError if corrupted
        """
    
    def delete_checkpoint_files(self, run_id: str) -> None:
        """Delete all checkpoint files for a run."""
```

#### Checkpoint Format

Checkpoints are stored as:

```
{checkpoint_dir}/
  {run_id}_phase_{phase}.pkl    # joblib-serialized checkpoint data
  {run_id}_phase_{phase}.json   # metadata (read-only, for debugging)
```

#### Checkpoint Data Structure

```python
checkpoint = {
    "phase": str,                      # e.g., "phase_1a_screening"
    "combination_index": int,          # Current combination (0-based)
    "total_combinations": int,         # Total combos in phase
    "results_df": pd.DataFrame,        # Results so far
    "best_model": sklearn.Pipeline,    # Best model object
    "best_auc": float,                 # Best AUC score
    "best_config": Dict[str, Any],     # Best config
}
```

#### Usage Example

```python
from shared_lib.checkpoint_manager import CheckpointManager
import pandas as pd

mgr = CheckpointManager("results/checkpoints/")

# Save after a combination completes
path = mgr.save_checkpoint(
    run_id="training-001",
    phase="phase_1a_screening",
    combination_index=5,
    total_combinations=10,
    results_df=df,
    best_model=model,
    best_auc=0.85,
    best_config={"model": "LogReg", "preprocessor": "Scaler"},
)

# Later, load the checkpoint
checkpoint = mgr.load_checkpoint(run_id="training-001")
model = checkpoint["best_model"]
df = checkpoint["results_df"]
```

---

### 3. Trainer Integration

The `train_with_adaptive_search` function accepts control signal and checkpoint manager.

#### Signature

```python
def train_with_adaptive_search(
    X: List[str],
    y: List[int],
    models: Optional[List[str]] = None,
    preprocessors: Optional[List[str]] = None,
    cv_folds: int = 5,
    top_percent: float = 0.20,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    run_id: Optional[str] = None,
    control_signal: Optional[ControlSignal] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> Dict[str, Any]:
    """
    Train adaptive search with optional controls.
    
    Args:
        X, y: Training data
        models: List of model names
        preprocessors: List of preprocessor names
        cv_folds: Number of CV folds
        top_percent: Fraction to advance
        n_jobs: Parallel jobs
        random_state: Random seed
        progress_callback: Callback for events
        run_id: Unique run ID
        control_signal: Optional ControlSignal for pause/stop
        checkpoint_manager: Optional CheckpointManager
        
    Returns:
        Dict with keys:
            - best_auc: Best AUC found
            - best_config: Best configuration
            - results_df: All results
            - metrics: Summary metrics
    """
```

#### Control Signal Polling

The trainer polls the control signal after each combination:

```python
# After each combination completes:
if control_signal.should_pause():
    # Wait for resume (in while loop with sleep)
    while control_signal.should_pause() == False:
        time.sleep(0.1)
    # Resume continues from here

if control_signal.should_stop():
    break  # Exit training loop
```

---

### 4. Telemetry Events

The trainer emits telemetry events during execution.

#### Event Schema

```python
event = {
    "run_id": str,                    # Unique run ID
    "seq": int,                       # Monotonic sequence number
    "event_id": str,                  # Unique event ID (UUID)
    "emitted_at": str,                # ISO 8601 timestamp
    "event_type": str,                # See event types below
    "phase": str,                     # Phase name
    "unit": str,                      # "combination" or "candidate"
    "status": str,                    # "queued", "running", "paused", etc.
    "current": int,                   # Current progress
    "total": int,                     # Total progress
    "metrics": Dict,                  # Optional metrics
}
```

#### Event Types

| Event Type | When | Payload |
|------------|------|---------|
| `training.run.started` | Run starts | - |
| `training.phase.started` | Phase starts | combinations count |
| `training.candidate.completed` | Combination done | metrics (AUC, etc.) |
| `training.paused` | Training paused | current state |
| `training.resumed` | Training resumed | - |
| `training.phase.completed` | Phase done | best metrics |
| `training.run.completed` | Run done | final metrics |

#### Usage Example

```python
from adaptive_trainer import train_with_adaptive_search
from shared_lib.control_signal import ControlSignal

events = []

def capture_event(event):
    events.append(event)
    print(f"Event: {event['event_type']} - {event['status']}")

control_signal = ControlSignal(run_id="my-run")

result = train_with_adaptive_search(
    X, y,
    progress_callback=capture_event,
    control_signal=control_signal,
)

# All events are captured
for event in events:
    print(f"{event['emitted_at']}: {event['event_type']}")
```

---

## Streaming UI Integration (Streamlit)

### RealtimeDashboardState

Consumes telemetry events and maintains UI state.

#### Location
```
ai-resources/shared_lib/realtime_dashboard.py
```

#### Usage in Streamlit

```python
from shared_lib.realtime_dashboard import RealtimeDashboardState
from shared_lib.control_signal import ControlSignal
import streamlit as st

# Initialize on button click
run_id = str(uuid.uuid4())
control_signal = ControlSignal(run_id)
dashboard_state = RealtimeDashboardState(run_id=run_id, buffer_size=500)

# Define callback
def progress_callback(event):
    dashboard_state.on_event(event)

# Run training
result = train_with_adaptive_search(
    X, y,
    progress_callback=progress_callback,
    control_signal=control_signal,
)

# Display progress
status = dashboard_state.get_status_summary()
st.progress(status['phase_progress']['current'] / status['phase_progress']['total'])

# Display buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Pause"):
        control_signal.request_pause()
with col2:
    if st.button("Stop"):
        control_signal.request_stop()
```

---

## Error Handling

### ControlSignal

```python
# No exceptions thrown - state machine is robust
signal = ControlSignal()

# Safe to call multiple times
signal.request_pause()  # OK
signal.request_pause()  # OK (no-op if already paused)
signal.resume()         # OK (transitions to RUNNING)
```

### CheckpointManager

```python
from shared_lib.checkpoint_manager import CheckpointManager

mgr = CheckpointManager("results/checkpoints/")

try:
    # Validate checkpoint
    if not mgr.validate_checkpoint(path):
        print("Checkpoint corrupted")
except RuntimeError as e:
    print(f"Checkpoint error: {e}")

try:
    # Load checkpoint
    data = mgr.load_checkpoint(run_id)
except FileNotFoundError:
    print(f"No checkpoint for run {run_id}")
```

---

## Testing

### Unit Tests

Located in `tests/`:

- `test_control_signal.py`: State transitions, thread safety
- `test_checkpoint_manager.py`: Save/load, validation, atomicity
- `test_training_controls_integration.py`: End-to-end pause/stop/resume

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific suite
pytest tests/test_control_signal.py -v

# With coverage
pytest tests/ --cov=ai-resources/shared_lib
```

---

## Performance Characteristics

### ControlSignal

- `request_pause()`: < 1μs (lock acquisition + state change)
- `request_stop()`: < 1μs
- `should_pause()`: < 10μs (atomic state transition)
- `should_stop()`: < 10μs
- `get_state()`: < 1μs

### CheckpointManager

- `save_checkpoint()`: ~300-500ms (depends on model + data size)
- `load_checkpoint()`: ~300-500ms
- `validate_checkpoint()`: ~50ms

### Trainer Overhead

- Control polling after each combination: < 1ms
- No overhead if control signal not used
- Checkpoint saves only on pause/stop (not every iteration)

---

## Backward Compatibility

The control system is fully backward compatible:

```python
# Old code (without controls) still works
result = train_with_adaptive_search(X, y)

# New code (with controls)
signal = ControlSignal()
result = train_with_adaptive_search(X, y, control_signal=signal)

# Both produce identical results
```

---

## Best Practices

### 1. Always Provide run_id

```python
# Good - explicit run ID for tracking
signal = ControlSignal(run_id="my-training-run-001")

# Acceptable - auto-generates UUID
signal = ControlSignal()
```

### 2. Handle Checkpoint Cleanup

```python
# Clean up old checkpoints periodically
import shutil
from pathlib import Path

checkpoint_dir = Path("results/checkpoints/")
for pkl_file in checkpoint_dir.glob("*.pkl"):
    age = (datetime.now() - pkl_file.stat().st_mtime) / 86400  # days
    if age > 7:  # Older than 7 days
        pkl_file.unlink()
```

### 3. Test Pause/Resume in Development

```python
def test_pause_resume():
    signal = ControlSignal()
    
    # Simulate pause in background thread
    def pause_later():
        time.sleep(0.5)
        signal.request_pause()
        time.sleep(1)
        signal.resume()
    
    thread = threading.Thread(target=pause_later)
    thread.start()
    
    # Training should pause and resume
    result = train_with_adaptive_search(X, y, control_signal=signal)
    
    assert result is not None
```

---

## Rollback Strategy

If you need to disable controls:

```python
# Option 1: Don't pass control_signal
result = train_with_adaptive_search(X, y)  # No controls

# Option 2: Remove UI buttons from ui_app.py
# Comment out or delete the pause/stop/resume button sections

# Option 3: Disable at import time
# Remove imports of ControlSignal and CheckpointManager
```

---

## Further Reading

- [TRAINING_CONTROLS_USER_GUIDE.md](TRAINING_CONTROLS_USER_GUIDE.md) - User guide
- [CONTROL_STATE_MACHINE_DESIGN.md](../docs/plan/20260508-training-controls/CONTROL_STATE_MACHINE_DESIGN.md) - Design details
- [CHECKPOINT_DESIGN.md](../docs/plan/20260508-training-controls/CHECKPOINT_DESIGN.md) - Checkpoint format
