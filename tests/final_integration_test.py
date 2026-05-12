"""Final integration test: Telemetry Emitter -> Realtime Dashboard."""

from harp.shared_lib.telemetry_emitter import TelemetryEmitter
from harp.shared_lib.realtime_dashboard import RealtimeDashboardState

# Simulate complete training flow
print("=" * 70)
print("FINAL INTEGRATION TEST: Telemetry Emitter -> Realtime Dashboard")
print("=" * 70)

# Create emitter and dashboard
run_id = "integration_test_run"
emitter = TelemetryEmitter(run_id=run_id)
dashboard = RealtimeDashboardState(run_id=run_id, buffer_size=100)

# Subscribe dashboard to emitter
emitter.subscribe(lambda event: dashboard.on_event(event))

# Simulate training lifecycle
print("\n1. Training started...")
emitter.emit_event("training.run.started", "init", "run", "started")

print("2. Phase 1a started (2-fold CV)...")
emitter.emit_event(
    "training.phase.started", "phase_1a_screening", "phase", "started",
    current=0, total=6, cv_folds=2
)

print("3. Processing candidates in phase 1a...")
for i in range(6):
    emitter.emit_event(
        "training.candidate.completed", "phase_1a_screening", "combination", "completed",
        model=['xgb', 'rf', 'svc'][i % 3],
        preprocessor=['tfidf', 'count_vec'][i % 2],
        current=i+1, total=6,
        metrics={'auc': round(0.88 + i*0.01, 4)}
    )
    print(f"  - Candidate {i+1}/6 processed")

print("4. Phase 1a completed...")
emitter.emit_event(
    "training.phase.completed", "phase_1a_screening", "phase", "completed",
    current=6, total=6, metrics={'combinations_screened': 6}
)

print("5. Phase 1b started (5-fold CV on top 2)...")
emitter.emit_event(
    "training.phase.started", "phase_1b_full_cv", "phase", "started",
    current=0, total=2, cv_folds=5
)

print("6. Processing top candidates in phase 1b...")
for i in range(2):
    emitter.emit_event(
        "training.candidate.completed", "phase_1b_full_cv", "candidate", "completed",
        model=['xgb', 'rf'][i],
        preprocessor=['tfidf', 'count_vec'][i],
        current=i+1, total=2,
        metrics={'auc': round(0.91 + i*0.005, 4)}
    )
    print(f"  - Top candidate {i+1}/2 completed")

print("7. Phase 1b completed...")
emitter.emit_event(
    "training.phase.completed", "phase_1b_full_cv", "phase", "completed",
    current=2, total=2, metrics={'combinations_fully_evaluated': 2}
)

print("8. Training completed...")
emitter.emit_event(
    "training.run.completed", "init", "run", "completed",
    metrics={'best_auc': 0.916}
)

# Verify dashboard state
print("\n" + "=" * 70)
print("DASHBOARD STATE AFTER TRAINING")
print("=" * 70)

summary = dashboard.get_status_summary()
print(f"\nRun ID: {summary['run_id']}")
print(f"Status: {summary['run_status']}")
print(f"Phase: {summary['phase']}")
print(f"Phase Progress: {summary['phase_progress']}")
print(f"Event Buffer: {summary['event_count']}/{summary['buffer_size']}")

combined = dashboard.get_combined_status()
print(f"\nCombined Status:")
print(f"  Phase: {combined['phase']}")
print(f"  Status: {combined['status']}")
print(f"  Queued: {combined['queued_count']}")
print(f"  Running: {combined['running_count']}")
print(f"  Completed: {combined['completed_count']}")
print(f"  Failed: {combined['failed_count']}")

print(f"\nStatus Board ({len(dashboard.status_board)} models):")
for model, prep_dict in dashboard.status_board.items():
    for prep, state in prep_dict.items():
        print(f"  {model} + {prep}: {state['state']} | metrics={state['metrics']}")

print(f"\nRecent Logs ({len(dashboard.event_buffer)} events in buffer):")
logs = dashboard.get_recent_logs(count=10)
for log in logs:
    print(f"  {log}")

# Verify safeguards
print("\n" + "=" * 70)
print("SAFETY CHECKS")
print("=" * 70)

# Test 1: Stale run rejection
print("\n1. Testing stale run rejection...")
stale_event = {
    'run_id': 'old_run',
    'seq': 100,
    'event_id': 'evt_stale',
    'emitted_at': '2026-05-08T12:00:00Z',
    'event_type': 'training.test',
    'phase': 'test',
    'unit': 'test',
    'status': 'started',
}
result = dashboard.on_event(stale_event)
assert result == False, "Stale run should be rejected"
print("  ✓ Stale run rejected")

# Test 2: Duplicate rejection
print("2. Testing duplicate rejection...")
first_event = emitter.emit_event("training.test.duplicate", "test", "test", "test")
dashboard.on_event(first_event)
result2 = dashboard.on_event(first_event)
assert result2 == False, "Duplicate should be rejected"
print("  ✓ Duplicate rejected")

# Test 3: Out-of-order ordering
print("3. Testing out-of-order event handling...")
old_seq_event = {
    'run_id': run_id,
    'seq': 1,
    'event_id': 'evt_old_seq',
    'emitted_at': '2026-05-08T11:59:00Z',
    'event_type': 'training.old.event',
    'phase': 'old',
    'unit': 'test',
    'status': 'started',
}
dashboard.on_event(old_seq_event)
assert dashboard.render_cursor > 1, "Out-of-order should not update cursor"
assert len(dashboard.event_buffer) > 0, "Out-of-order should still be buffered"
print("  ✓ Out-of-order buffered but not rendering")

print("\n" + "=" * 70)
print("✓ INTEGRATION TEST PASSED")
print("=" * 70)
