#!/usr/bin/env python
"""Simple debug script for pause/resume mechanism."""

import sys
import os
import time
import threading
from pathlib import Path
import numpy as np

from harp.adaptive_trainer import train_with_adaptive_search
from harp.shared_lib.control_signal import ControlSignal
from harp.shared_lib.checkpoint_manager import CheckpointManager

# Monkey-patch the trainer to add debug output
print("Setting up debugging instrumentation...")

# Generate small test data
np.random.seed(42)
X = np.random.randn(50, 5)
y = np.random.randint(0, 2, 50)

run_id = "debug-pause-test"
control_signal = ControlSignal(run_id=run_id)
checkpoint_manager = CheckpointManager(str(Path(__file__).parent.parent / 'results' / 'checkpoints'))

events = []

def capture_event(event):
    events.append(event)
    print(f"  [Event] {event['event_type']} @ {event['phase']}")

def pause_then_resume():
    time.sleep(0.5)
    print("\n[Background] Requesting pause...")
    control_signal.request_pause()
    print(f"[Background] Pause requested, current state: {control_signal.get_state()}")
    time.sleep(2)
    print("[Background] Calling resume...")
    control_signal.resume()
    print(f"[Background] Resume called, current state: {control_signal.get_state()}\n")

bg_thread = threading.Thread(target=pause_then_resume, daemon=True)
bg_thread.start()

print("Starting training with pause/resume...")
start_time = time.time()

try:
    result = train_with_adaptive_search(
        X, y,
        models=['LogisticRegression', 'RandomForest'],
        preprocessors=['StandardScaler'],
        cv_folds=2,
        top_percent=1.0,
        n_jobs=1,
        random_state=42,
        control_signal=control_signal,
        checkpoint_manager=checkpoint_manager,
        progress_callback=capture_event,
        run_id=run_id
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s")
    print(f"Results: {len(result['results_df'])} combinations, best_auc={result['metrics']['best_auc']:.4f}")
    
    # Check events
    pause_events = [e for e in events if e['event_type'] == "training.paused"]
    resume_events = [e for e in events if e['event_type'] == "training.resumed"]
    print(f"Pause events: {len(pause_events)}, Resume events: {len(resume_events)}")
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\nTraining failed after {elapsed:.2f}s: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    bg_thread.join(timeout=5)
    print("Done.")

