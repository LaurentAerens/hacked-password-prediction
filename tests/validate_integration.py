"""
Quick validation that telemetry emitter integration is correctly implemented.
Checks imports, function signatures, and event emission without full training.
"""

import sys
import os
from pathlib import Path

# Add ai-resources directory to path
ai_resources_path = str(Path(__file__).parent.parent / "ai-resources")
sys.path.insert(0, ai_resources_path)

print("=" * 70)
print("Quick Validation: Telemetry Emitter Integration")
print("=" * 70)

# Test 1: Import telemetry emitter
print("\n[1/5] Testing telemetry emitter import...")
try:
    from shared_lib.telemetry_emitter import TelemetryEmitter, create_emitter
    print("✓ Telemetry emitter imports successfully")
except Exception as e:
    print(f"✗ Failed to import telemetry emitter: {e}")
    sys.exit(1)

# Test 2: Create emitter instance
print("\n[2/5] Testing emitter instantiation...")
try:
    emitter = create_emitter(run_id="validation_run_001")
    assert emitter.run_id == "validation_run_001"
    print("✓ Emitter instantiates correctly")
except Exception as e:
    print(f"✗ Failed to create emitter: {e}")
    sys.exit(1)

# Test 3: Emit events
print("\n[3/5] Testing event emission...")
try:
    events = []
    emitter.subscribe(lambda e: events.append(e))
    
    emitter.emit_event("training.run.started", "init", "run", "started")
    emitter.emit_event("training.phase.started", "phase_1a_screening", "combination", "started", 
                      current=0, total=12)
    emitter.emit_event("training.phase.completed", "phase_1a_screening", "combination", "completed",
                      current=12, total=12)
    emitter.emit_event("training.run.completed", "completion", "run", "completed",
                      metrics={"best_auc": 0.95})
    
    assert len(events) == 4, f"Expected 4 events, got {len(events)}"
    assert all('run_id' in e for e in events), "Missing run_id in events"
    assert all('seq' in e for e in events), "Missing seq in events"
    print(f"✓ Emitted 4 events with required fields")
except Exception as e:
    print(f"✗ Failed to emit events: {e}")
    sys.exit(1)

# Test 4: Import adaptive trainer
print("\n[4/5] Testing adaptive trainer import...")
try:
    from adaptive_trainer import train_with_adaptive_search
    print("✓ Adaptive trainer imports successfully")
except Exception as e:
    print(f"✗ Failed to import adaptive trainer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify function signature
print("\n[5/5] Testing function signature...")
try:
    import inspect
    sig = inspect.signature(train_with_adaptive_search)
    params = list(sig.parameters.keys())
    
    # Check for required parameters
    assert 'progress_callback' in params, "Missing progress_callback parameter"
    assert 'run_id' in params, "Missing run_id parameter"
    
    # Check defaults
    assert sig.parameters['progress_callback'].default is None
    assert sig.parameters['run_id'].default is None
    
    print(f"✓ Function signature includes progress_callback and run_id parameters")
except Exception as e:
    print(f"✗ Failed to verify function signature: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL VALIDATIONS PASSED")
print("=" * 70)
print("\nSummary:")
print("  - Telemetry emitter module loads correctly")
print("  - Event emission works with required fields")
print("  - Adaptive trainer imports and has updated signature")
print("  - Integration ready for full testing")
print("=" * 70)
