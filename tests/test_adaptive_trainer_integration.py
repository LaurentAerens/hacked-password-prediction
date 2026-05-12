"""
Integration test for adaptive trainer telemetry emission.

Tests that:
1. Events are emitted at correct lifecycle boundaries
2. Callback is optional and trainer works without it
3. Event envelope contains required fields and values
4. Sequence of events follows expected lifecycle
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

from harp.adaptive_trainer import train_with_adaptive_search


def test_adaptive_trainer_without_callback():
    """Test that trainer works without callback (callback-absent behavior)."""
    print("\nTest 1: Adaptive trainer without callback...")
    
    # Generate small test data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    try:
        result = train_with_adaptive_search(
            X, y,
            models=['LogisticRegression'],
            preprocessors=['StandardScaler'],
            cv_folds=2,
            top_percent=1.0,
            n_jobs=1,
            random_state=42
        )
        
        # Verify result structure
        assert 'best_model' in result
        assert 'best_config' in result
        assert 'results_df' in result
        assert 'experiment_name' in result
        assert 'metrics' in result
        
        print("✓ PASSED: Trainer works without callback")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_trainer_with_callback():
    """Test that trainer emits events when callback is provided."""
    print("\nTest 2: Adaptive trainer with callback...")
    
    # Generate small test data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    events = []
    
    def capture_event(event):
        events.append(event)
    
    try:
        result = train_with_adaptive_search(
            X, y,
            models=['LogisticRegression', 'RandomForest'],
            preprocessors=['StandardScaler'],
            cv_folds=2,
            top_percent=1.0,
            n_jobs=1,
            random_state=42,
            progress_callback=capture_event
        )
        
        # Verify events were emitted
        assert len(events) > 0, f"No events emitted, got {len(events)}"
        print(f"✓ Emitted {len(events)} events")
        
        # Verify lifecycle events
        event_types = [e['event_type'] for e in events]
        assert "training.run.started" in event_types, f"Missing run.started in {event_types}"
        assert "training.phase.started" in event_types, f"Missing phase.started in {event_types}"
        assert "training.candidate.completed" in event_types, f"Missing candidate.completed in {event_types}"
        assert "training.phase.completed" in event_types, f"Missing phase.completed in {event_types}"
        assert "training.run.completed" in event_types, f"Missing run.completed in {event_types}"
        
        print("✓ PASSED: All lifecycle events present")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_event_envelope_fields():
    """Test that emitted events have required envelope fields."""
    print("\nTest 3: Event envelope fields...")
    
    # Generate small test data
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    events = []
    
    def capture_event(event):
        events.append(event)
    
    try:
        result = train_with_adaptive_search(
            X, y,
            models=['LogisticRegression'],
            preprocessors=['StandardScaler'],
            cv_folds=2,
            top_percent=1.0,
            n_jobs=1,
            random_state=42,
            progress_callback=capture_event,
            run_id="test_run_001"
        )
        
        # Verify required fields in all events
        required_fields = ['run_id', 'seq', 'event_id', 'emitted_at', 'event_type', 'phase', 'unit', 'status']
        
        for event in events:
            for field in required_fields:
                assert field in event, f"Missing required field '{field}' in event {event['event_type']}"
        
        # Verify run_id correlation
        for event in events:
            assert event['run_id'] == "test_run_001", f"Run ID mismatch: {event['run_id']}"
        
        # Verify seq is monotonic
        seqs = [e['seq'] for e in events]
        assert seqs == sorted(seqs), f"Sequence not monotonic: {seqs}"
        
        print(f"✓ PASSED: All {len(events)} events have required fields and valid seq")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_lifecycle_sequence():
    """Test that phase start/end events bracket phase execution."""
    print("\nTest 4: Phase lifecycle sequence...")
    
    # Generate small test data
    np.random.seed(42)
    X = np.random.randn(60, 5)
    y = np.random.randint(0, 2, 60)
    
    events = []
    
    def capture_event(event):
        events.append(event)
    
    try:
        result = train_with_adaptive_search(
            X, y,
            models=['LogisticRegression', 'RandomForest'],
            preprocessors=['StandardScaler'],
            cv_folds=2,
            top_percent=1.0,
            n_jobs=1,
            random_state=42,
            progress_callback=capture_event
        )
        
        # Extract phase events
        phase_1a_started = None
        phase_1a_completed = None
        phase_1b_started = None
        phase_1b_completed = None
        
        for event in events:
            if event['event_type'] == "training.phase.started" and event['phase'] == "phase_1a_screening":
                phase_1a_started = event
            elif event['event_type'] == "training.phase.completed" and event['phase'] == "phase_1a_screening":
                phase_1a_completed = event
            elif event['event_type'] == "training.phase.started" and event['phase'] == "phase_1b_full_cv":
                phase_1b_started = event
            elif event['event_type'] == "training.phase.completed" and event['phase'] == "phase_1b_full_cv":
                phase_1b_completed = event
        
        # Verify phase lifecycle
        assert phase_1a_started is not None, "Phase 1a not started"
        assert phase_1a_completed is not None, "Phase 1a not completed"
        assert phase_1b_started is not None, "Phase 1b not started"
        assert phase_1b_completed is not None, "Phase 1b not completed"
        
        # Verify ordering
        assert phase_1a_started['seq'] < phase_1a_completed['seq'], "Phase 1a ordering wrong"
        assert phase_1a_completed['seq'] < phase_1b_started['seq'], "Phase ordering wrong between 1a and 1b"
        assert phase_1b_started['seq'] < phase_1b_completed['seq'], "Phase 1b ordering wrong"
        
        print("✓ PASSED: Phase lifecycle sequence correct")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_candidate_events_with_metrics():
    """Test that candidate completion events include metrics."""
    print("\nTest 5: Candidate events with metrics...")
    
    # Generate small test data
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    events = []
    
    def capture_event(event):
        events.append(event)
    
    try:
        result = train_with_adaptive_search(
            X, y,
            models=['LogisticRegression'],
            preprocessors=['StandardScaler'],
            cv_folds=2,
            top_percent=1.0,
            n_jobs=1,
            random_state=42,
            progress_callback=capture_event
        )
        
        # Find candidate completion events
        candidate_events = [e for e in events if e['event_type'] == 'training.candidate.completed']
        assert len(candidate_events) > 0, "No candidate completion events found"
        
        # Verify each candidate event has expected fields
        for event in candidate_events:
            assert 'model' in event, f"Missing model in {event}"
            assert 'preprocessor' in event, f"Missing preprocessor in {event}"
            assert 'metrics' in event, f"Missing metrics in {event}"
            assert 'current' in event, f"Missing current in {event}"
            assert 'total' in event, f"Missing total in {event}"
            
            # Verify metrics
            assert 'auc' in event['metrics'], f"Missing auc metric"
            assert event['metrics']['auc'] >= 0.0 and event['metrics']['auc'] <= 1.0, f"AUC out of range: {event['metrics']['auc']}"
        
        print(f"✓ PASSED: {len(candidate_events)} candidate events have metrics")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("Integration Tests: Adaptive Trainer Telemetry")
    print("=" * 70)
    
    results = []
    results.append(("Without callback", test_adaptive_trainer_without_callback()))
    results.append(("With callback", test_adaptive_trainer_with_callback()))
    results.append(("Event envelope", test_event_envelope_fields()))
    results.append(("Phase lifecycle", test_phase_lifecycle_sequence()))
    results.append(("Candidate metrics", test_candidate_events_with_metrics()))
    
    print("\n" + "=" * 70)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    print("=" * 70)
    
    sys.exit(0 if passed == total else 1)
