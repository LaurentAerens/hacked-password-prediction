# NN Backend Release Notes

Release: NN Backend v1.0  
Release Date: 2026-05-08

---

## 1. Release Summary

NN Backend v1.0 introduces a PyTorch-based neural network training track that runs in parallel with existing Phase 1 and Phase 2 pipelines.

This release focuses on:

- Additive architecture
- Local-first usability
- Live training control and visibility
- Backward compatibility

---

## 2. Key Features

### 2.1 PyTorch CNN Classifier

- Compact PasswordCNN baseline (~4,980 parameters)
- Character-level input embedding
- Multi-branch conv kernels [2, 3, 4]
- Binary classification output with sigmoid thresholding

### 2.2 Character-Level Tokenization

- ASCII 0-255 mapping
- Max sequence length 32
- Embedding dimension 8

### 2.3 Device-Aware Training

- Auto device detection (CUDA or CPU)
- Batch size helper tuned by device class
- Mixed precision context helper available for CUDA path extension

### 2.4 Training Controls

- Start, Pause, Resume, Stop control flow
- ControlSignal integration for cooperative loop control
- Live status updates in UI

### 2.5 Live Progress Visualization

- Epoch progress bar
- Runtime metrics cards
- Event log display via telemetry emitter

### 2.6 Ensemble Predictions

- Soft-vote averaging across Phase 1 / Phase 2 / NN
- Equal-weight normalized voting
- Graceful handling when some models are missing

### 2.7 Model Comparison Dashboard

- Unified table across latest models by phase
- Plot-based metric comparison
- Quick best-model-by-metric helper

### 2.8 Backward-Compatible Integration

- Existing Phase 1/2 logic unchanged
- Existing model formats and checkpoints preserved
- NN artifacts isolated in models/nn

---

## 3. Performance Characteristics

Observed or targeted guidance:

- Training runtime:
  - CPU: approximately 5 to 10 minutes in common local runs
  - GPU: approximately 1 to 2 minutes for comparable settings
- Inference latency:
  - Often below 2 ms per password in local lightweight scenarios
- Model footprint:
  - Small artifact footprint (commonly well below 1 MB for core model state)

Notes:

- Runtime depends on dataset size, hardware, and hyperparameters.

---

## 4. Artifact Layout

NN backend stores outputs in:

```text
models/nn/
├── checkpoints/{run_id}/epoch_{epoch}.pt
├── final/{run_id}/best_model.pt
├── final/{run_id}/metadata.json
└── history/{run_id}/metrics.csv
```

This structure is isolated from phase1 and phase2 registries.

---

## 5. Compatibility Notes

Breaking changes:

- None

Compatibility guarantees:

- Phase 1 behavior unchanged
- Phase 2 behavior unchanged
- Existing checkpoints remain usable
- Old model formats remain supported

---

## 6. Known Limitations

Current limitations in v1.0:

- PoC runtime profile is primarily validated on CPU paths in-repo; GPU behavior may depend on external hardware configuration
- Early stopping with configurable patience is not fully implemented
- Attention-based NN variant is not included (planned for Wave 3b)
- Automated NN hyperparameter sweeps are not included (planned for Wave 3b)

Additional implementation notes:

- Some telemetry events exist in design schemas but only a subset is emitted in current trainer code
- Trainer-level load_checkpoint convenience method is not yet present; registry-level loading is available

---

## 7. Upgrade and Adoption Guidance

Recommended first-use sequence:

1. Install torch in your environment
2. Train NN with default settings
3. Validate metrics in Neural Network Training page
4. Compare against Phase 1/2
5. Optionally enable ensemble for robustness

---

## 8. Future Roadmap (Wave 3b+)

Planned evolutions:

- Early stopping with validation patience
- Learning rate scheduling
- Attention visualization and enhanced model variants
- Automated NN hyperparameter sweep (Optuna integration)
- Transfer learning with pretrained embedding strategies

Roadmap items are not part of v1.0 runtime guarantees.

---

## 9. Operational Honesty Statement

This release is intentionally practical and incremental.

What is production-useful today:

- Independent NN training
- Artifact persistence
- Live UI feedback
- Cross-backend comparison and soft voting

What remains iterative:

- Advanced stopping policies
- Automated tuning pipelines
- Expanded NN architecture families

---

## 10. Credits and Traceability

Primary implementation modules:

- ai-resources/nn_tokenizer.py
- ai-resources/nn_models.py
- ai-resources/nn_trainer.py
- ai-resources/nn_registry.py
- ai-resources/ensemble.py
- ai-resources/model_registry.py
- ai-resources/model_comparison.py
- ai-resources/ui_app.py

Planning references:

- docs/plan/20260508-nn-backend/architecture_design.md
- docs/plan/20260508-nn-backend/telemetry_schema.yaml
- docs/plan/20260508-nn-backend/ui_config_spec.md
