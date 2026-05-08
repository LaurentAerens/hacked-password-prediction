# Training Controls: User Guide

## Overview

The H.A.R.P. training system now supports **pause**, **resume**, and **stop** controls. You can pause training between combinations to save progress, resume later, or stop early to get partial results.

## Features

### ✅ Live Progress Visualization
- **Progress bars** showing phase, combination count, and percentage
- **Status board** tracking each model+preprocessor combination
- **Real-time event log** of training events
- **Fold/trial counters** (sklearn alternative to epochs)

### ⏸️ Pause Training
- Pause gracefully between combinations
- Training freezes after completing the current combination
- Progress and checkpoints are automatically saved
- **Button**: Yellow "⏸️ PAUSE TRAINING" button

### ▶️ Resume Training
- Resume from the pause point
- Training continues with the saved checkpoint
- No loss of progress
- **Button**: Green "▶️ RESUME TRAINING" button (appears when paused)

### 🛑 Stop Training
- Stop training immediately
- Get partial results (combinations completed so far)
- Checkpoints are automatically saved
- **Button**: Red "🛑 STOP TRAINING" button

## How to Use

### 1. Start Training

```python
# In Streamlit UI:
# 1. Configure hyperparameters (top_percent, cv_folds)
# 2. Click "Start Adaptive Training"
# 3. Training begins with live progress bars
```

### 2. Pause Training

During training, you'll see three control buttons:

```
⏸️ PAUSE TRAINING  |  🛑 STOP TRAINING  |  (Resume button when paused)
```

- Click **⏸️ PAUSE TRAINING** to pause after the current combination completes
- Status indicator shows "⏳ Pausing..." while transition occurs
- Once paused, status shows "⏸️ Paused"

### 3. Resume Training

When training is paused:

- Click **▶️ RESUME TRAINING** to continue
- Training resumes from the checkpoint
- Status indicator shows "✅ Running"

### 4. Stop Training

- Click **🛑 STOP TRAINING** at any time
- Training stops after completing the current combination
- Status shows "🛑 Stopped"
- Partial results are available immediately

## Checkpoint System

Checkpoints are automatically created and managed. You don't need to manually manage them.

### Automatic Checkpoints

- Created after **each combination completes**
- Stored in: `results/checkpoints/`
- Named: `{run_id}_phase_{phase}.pkl`
- Metadata saved as: `{run_id}_phase_{phase}.json`

### Recovery

If you pause and resume:
- Training automatically loads from the latest checkpoint
- No data loss
- Results are combined with new training runs

### Manual Checkpoint Inspection

```python
from shared_lib.checkpoint_manager import CheckpointManager

mgr = CheckpointManager("results/checkpoints/")
checkpoint = mgr.load_checkpoint(run_id="abc123")

# Access saved state
print(checkpoint['phase'])  # e.g., "phase_1a_screening"
print(checkpoint['combination_index'])  # e.g., 5
print(checkpoint['best_auc'])  # Best AUC so far
```

## Status Indicators

The status line shows current training state:

| Emoji | Status | Meaning |
|-------|--------|---------|
| ✅ | Running | Training is active |
| ⏳ | Pausing... | Pause requested, waiting for combination to complete |
| ⏸️ | Paused | Training is paused, waiting for resume |
| ⏳ | Stopping... | Stop requested, waiting for combination to complete |
| 🛑 | Stopped | Training has stopped |

## Results

### During Training

- **Progress bar**: Visual progress through all combinations
- **Status board**: Per-model status (queued, in-progress, completed)
- **Live log**: Event stream of training progress

### After Training

- **Final results table**: DataFrame of all combinations and their AUC scores
- **Best model**: Top configuration with highest AUC
- **Time saved**: Percentage of time saved vs. exhaustive search

## Troubleshooting

### Pause doesn't seem to work

- Pause only takes effect between combinations
- If training is fast, pause may occur during a fold evaluation
- Check the status indicator for current state
- Wait for the current combination to complete

### Resume seems to be missing

- Resume button only appears when status is "Paused"
- If training isn't paused, the button is disabled
- Click "Pause" first, then wait for status to change

### Stopped training but want to continue

- You cannot resume after stopping (this is intentional for safety)
- Stop is a hard-stop for data safety
- If you want to continue, restart training with a fresh run

### Checkpoints taking up disk space

Checkpoints are cleaned up automatically, but you can manually clean:

```python
from shared_lib.checkpoint_manager import CheckpointManager

mgr = CheckpointManager("results/checkpoints/")

# Delete specific run's checkpoints
mgr.delete_checkpoint_files(run_id="abc123")

# Delete all checkpoints in directory
import shutil
shutil.rmtree("results/checkpoints/")
```

## Advanced: API Usage

### Using Control Signals Programmatically

```python
from shared_lib.control_signal import ControlSignal
from shared_lib.checkpoint_manager import CheckpointManager
from adaptive_trainer import train_with_adaptive_search

# Create control signal
run_id = "my-training-001"
control_signal = ControlSignal(run_id=run_id)
checkpoint_manager = CheckpointManager("results/checkpoints/")

# Run training with controls
result = train_with_adaptive_search(
    X, y,
    models=["LogisticRegression", "RandomForest"],
    preprocessors=["StandardScaler", "TfidfVectorizer"],
    control_signal=control_signal,
    checkpoint_manager=checkpoint_manager,
)

# From another thread, you can control training:
control_signal.request_pause()  # Request pause
control_signal.resume()          # Resume from pause
control_signal.request_stop()    # Request stop

# Check current state
state = control_signal.get_state()
# Returns: "RUNNING", "PAUSE_REQUESTED", "PAUSED", "STOP_REQUESTED", or "STOPPED"
```

### Understanding State Transitions

The control signal uses a thread-safe state machine:

```
RUNNING ─request_pause()─> PAUSE_REQUESTED ─should_pause()─> PAUSED ─resume()─> RUNNING
                                                                  │
                                                           request_stop()
                                                                  │
                                                          ─────────┴─> STOP_REQUESTED ─should_stop()─> STOPPED
```

- `request_pause()`: User requests pause
- `should_pause()`: Trainer checks (returns True once, then transitions to PAUSED)
- `resume()`: User requests resume
- `request_stop()`: User requests stop
- `should_stop()`: Trainer checks (returns True once, then transitions to STOPPED)

## Performance Impact

### Pause/Resume Overhead

- **Control signal polling**: < 1ms per check (after each combination)
- **Checkpoint saving**: ~500ms per checkpoint (depends on data size)
- **Checkpoint loading**: ~500ms per checkpoint

### Typical Training Times (Example)

With 4 combinations × 2 folds × 2 CV:
- **Without controls**: ~10 seconds
- **With controls** (no pause): ~10 seconds (< 1ms overhead per check)
- **With one pause**: ~12 seconds (2s additional for checkpoint save/load)

## Safety Guarantees

✅ **Data integrity**: Pause waits for combination to complete (no interruption mid-fold)
✅ **Atomic writes**: Checkpoints use atomic file operations (rename-based)
✅ **No model corruption**: Trainer never interrupted during model fit
✅ **State consistency**: Thread-safe state machine (lock-protected)
✅ **Loss-free**: All progress is saved, no loss on pause/resume

## Next Steps

For more information:
- See [CONTROL_STATE_MACHINE_DESIGN.md](../docs/plan/20260508-training-controls/CONTROL_STATE_MACHINE_DESIGN.md) for low-level details
- See [CHECKPOINT_DESIGN.md](../docs/plan/20260508-training-controls/CHECKPOINT_DESIGN.md) for checkpoint format specification
- See [API_REFERENCE.md](API_REFERENCE.md) for developer documentation
