# Live Training Progress Visualization

## What You'll See Now

When you click "Start Adaptive Training" in the Streamlit UI, instead of just a "this might take a few minutes" message, you'll now see:

### 📊 Training Progress
- **Phase**: Shows current phase (1a or 1b)
- **Combinations**: Counter showing how many combinations have been processed (e.g., "5/42")
- **Progress**: Percentage complete (e.g., "12%")
- **Progress Bar**: Visual bar showing overall progress through screening and evaluation

### 🎯 Model Status Board
For each model + preprocessor combination, you'll see:
- **Status Emoji**: 
  - ⏳ Queued (waiting to run)
  - 🔄 Running (currently training)
  - ✅ Completed (finished)
  - ❌ Failed (error occurred)
- **Combination Name**: Which model and preprocessor
- **Fold Progress**: For running combinations, shows "Fold 3/5" (or trial/candidate progress)
- **Best Metrics So Far**: Current best score for that combination
- **Per-Combination Progress Bar**: Visual bar showing folds completed

Example:
```
🔄 RandomForest + TF-IDF: Fold 3/5 (60%) | best_auc=0.8234
████████░░ 60%

✅ LogisticRegression + CountVectorizer: COMPLETED | best_auc=0.7891

⏳ SVM + TF-IDF: QUEUED
```

### 📋 Live Event Log
Real-time streaming of training events with:
- Phase transitions (1a screening → 1b full CV)
- Each combination's start/end and current progress
- Best score updates across all combinations
- Any errors or warnings

Example log output:
```
[phase_1a_screening] STARTED (0/42)
  RandomForest + TF-IDF: STARTED (0/5) | best_auc=0.0000
  RandomForest + TF-IDF: COMPLETED (5/5) | best_auc=0.8234
  LogisticRegression + CountVectorizer: STARTED (0/5) | best_auc=0.0000
[phase_1a_screening] COMPLETED (42/42)
[phase_1b_full_cv] STARTED (0/5)
  RandomForest + TF-IDF: STARTED (0/5) | best_auc=0.8234
```

### ✅ Training Complete
After training finishes, you'll see:
- **Summary Metrics**: Best AUC, combinations screened, fully evaluated, time saved
- **Best Configuration**: The winning model + preprocessor + hyperparameters (as JSON)
- **Results Table**: Top 25 combinations ranked by AUC
- **Results Chart**: Visual bar chart of top 15 combinations
- **Full Training Event Log**: All events from the complete training run

## Key Improvements Over Old UI

| Before | Now |
|--------|-----|
| "This might take a few minutes" | See exactly which model/preprocessor is running |
| Spinner for entire duration | Progress bars and fold/trial counters |
| No insight into what's happening | Real-time event log of every step |
| Results only after completion | Incremental feedback throughout |
| Epochs unavailable (sklearn) | Fold/trial/candidate progress (sklearn-compatible) |

## Understanding the Metrics

- **Phase 1a (Screening)**: Quick evaluation on all combinations to find top candidates
- **Phase 1b (Full CV)**: Detailed k-fold cross-validation on the top ~20% of combinations
- **Fold X/Y**: Current fold in cross-validation (e.g., Fold 3/5 means testing fold 3 out of 5)
- **Best AUC**: Best AUC score achieved so far for that combination

## Tips

- The live event log updates as training progresses
- Each model+preprocessor combination shows its own fold progress
- The main progress bar tracks overall completion (all combinations)
- You can see which combinations are queued, running, or completed
- Time Saved % shows efficiency gain from Phase 1a screening

## Performance Tuning (New)

The Training tab now includes **Performance tuning (CPU/RAM)** controls for smarter parallel scheduling in Phase 1a:

- **CPU utilization target**: Fraction of available CPUs to use (default `0.9`)
- **Estimated RAM per concurrent candidate (GB)**: Memory guardrail per model+preprocessor job (default `2.0`)
- **Max parallel model+preprocessor jobs**: Hard cap for concurrent combinations (`0` = auto)

How the scheduler works:

1. It computes a CPU budget from your machine core count and the CPU target.
2. It estimates safe parallel candidate count from total RAM and RAM-per-candidate.
3. It picks an outer worker count (`parallel models`) and inner GridSearch `n_jobs` (`parallel CV/params`) to avoid heavy oversubscription.

Recommended presets:

- High-core workstation / multi-CPU server: CPU target `0.9` to `1.0`.
- High-core workstation / multi-CPU server: RAM per candidate `2.0` to `4.0`.
- High-core workstation / multi-CPU server: Max parallel jobs `0` (auto) or manual cap.
- Laptop / low-memory machine: CPU target `0.6` to `0.8`.
- Laptop / low-memory machine: RAM per candidate `3.0` to `6.0`.
- Laptop / low-memory machine: Max parallel jobs explicit low value (for example `2` to `4`).

## Troubleshooting

If you don't see live updates:
1. Check that training actually started (should take 10+ seconds)
2. Refresh the browser if the page seems stuck
3. Check browser console (F12) for errors
4. Try a smaller dataset to test with fewer combinations

The data is updated continuously as training runs, captured in real-time telemetry events.
