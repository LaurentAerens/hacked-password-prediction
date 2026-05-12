# NN Migration Guide

Phase 1/2 to Phase 1/2 + Neural Network Backend

Version: 1.0  
Date: 2026-05-08

---

## 1. Migration Summary

This migration adds a Neural Network backend to H.A.R.P. without replacing existing adaptive training phases.

What changes:

- New Neural Network Training page in UI
- New Comparison page for side-by-side metrics
- New NN model storage under models/nn
- New ensemble option spanning phase1, phase2, and nn

What does not change:

- Existing Phase 1 behavior
- Existing Phase 2 behavior
- Existing phase1/phase2 model formats and checkpoints

---

## 2. Before NN Backend (Phase 1/2 Only)

### 2.1 Existing Workflow

Typical workflow before NN:

1. Generate combined dataset
2. Run adaptive search (Phase 1)
3. Optionally run Optuna fine-tuning (Phase 2)
4. Save best sklearn artifacts
5. Use Predict tab or external scripts

### 2.2 Existing Artifacts

Phase 1/2 outputs generally include:

- Best model files (joblib/pickle-based)
- Result tables in results directory
- Metadata and checkpoint artifacts from adaptive flow

### 2.3 Existing UI Shape

Primary training path focused on adaptive trainer.

No dedicated NN training panel and no unified three-backend comparison table.

---

## 3. After NN Backend (Phase 1/2 + NN)

### 3.1 New UI Surfaces

Added pages:

- Neural Network Training
- Comparison

New capabilities:

- Train compact PyTorch CNN model independently
- View live NN progress and telemetry
- Compare latest Phase 1, Phase 2, and NN metrics
- Run soft-voting ensemble predictions

### 3.2 New Model Registry Branch

NN artifacts are stored in dedicated structure:

```text
models/nn/
├── checkpoints/{run_id}/epoch_{epoch}.pt
├── final/{run_id}/best_model.pt
├── final/{run_id}/metadata.json
└── history/{run_id}/metrics.csv
```

This does not overwrite phase1 or phase2 artifact trees.

### 3.3 Backward Compatibility

Compatibility guarantees:

- Existing phase1 and phase2 checkpoints still work
- Existing model formats still load in their native paths
- Existing scripts that do not use NN continue to run

---

## 4. Training Both Workflows

### 4.1 Train Phase 1/2 As Before

No migration steps are required for legacy adaptive training.

Continue using your current:

- Data generation flow
- Adaptive training controls
- Existing scripts and tests

### 4.2 Train NN Independently

NN can be trained even if you skip phase1/phase2 training in the current session.

Independent prerequisites:

- combined_data.csv exists
- torch installed

### 4.3 Parallel Usage Strategy

Common strategy:

1. Train phase1 baseline
2. Train phase2 fine-tuned variant
3. Train NN baseline
4. Compare all in Comparison tab
5. Select best single model or ensemble

### 4.4 Practical Scheduling

Because all tracks can be compute-intensive, schedule runs by device capacity:

- CPU-only systems: run sequentially
- GPU-capable systems: NN on GPU, sklearn on CPU where possible

---

## 5. Ensemble Usage Migration

### 5.1 Load Individual Models

Comparison and ensemble flow relies on discovered best model paths.

Models can be selectively enabled:

- Phase 1 only
- Phase 2 only
- NN only
- Any combination of two
- All three

### 5.2 Configure Included Models

In Comparison page, checkboxes determine active models for ensemble calculation.

### 5.3 Predict with Ensemble

EnsemblePredictor computes soft-vote probability mean across active models and supports threshold-based binary labels.

### 5.4 Compare Ensemble vs Individual

Recommended evaluation:

- Inspect per-model metrics table first
- Evaluate ensemble sample predictions
- Validate threshold impact on your risk profile

---

## 6. Breaking Changes

None.

### 6.1 Explicit Non-Breaking Guarantees

- Phase 1/2 behavior unchanged
- Existing checkpoints still valid
- Existing model file formats still supported
- Existing adaptive controls remain available

### 6.2 Operational Caveats

No breaking API changes, but note:

- NN introduces torch as an additional dependency
- NN telemetry and controls are separate from adaptive event stream

---

## 7. Migration Checklist

Use this checklist for rollout:

- Install torch in environment
- Verify Streamlit app launches successfully
- Confirm combined_data.csv exists
- Run one NN training with default settings
- Confirm model artifacts appear under models/nn
- Open Comparison page and verify NN row appears
- Run ensemble prediction with at least one active model
- Execute regression tests relevant to your workflow

---

## 8. Validation Commands

Install dependencies:

```bash
pip install -r requirements/requirements.txt
pip install torch
```

Run app:

```bash
python -m streamlit run ai-resources/ui_app.py
```

Optional targeted tests:

```bash
pytest tests/test_nn_backend_regression.py
pytest tests/test_nn_trainer.py
pytest tests/test_streamlit_nn_integration.py
```

---

## 9. Rollback Plan

If NN backend is temporarily disabled, you can continue with Phase 1/2 only by:

- Not using Neural Network Training page
- Ignoring models/nn artifacts
- Keeping existing adaptive pipeline unchanged

Because migration is additive, rollback is operationally simple.

---

## 10. FAQ-Like Migration Decisions

Should I retrain all old models immediately?

- No. Existing phase1/phase2 outputs remain usable.

Should I switch to ensemble by default?

- Not automatically. Compare first, then decide by metric priorities.

Do I need to modify old scripts now?

- Only if you want NN-specific features.

---

## 11. Outcome

After migration you retain your existing workflows and gain:

- Additional NN training capability
- Better cross-model observability
- More flexible prediction strategy through soft voting

This is an additive, backward-compatible evolution path.
