# NN Backend FAQ

Version: 1.0  
Date: 2026-05-08

---

## 1. Can I train NN without Phase 1/2?

Yes.

The NN backend is independent and can train directly from combined_data.csv without requiring a Phase 1 or Phase 2 run in the same session.

---

## 2. Is NN always better than Phase 1/2?

No.

Performance depends on:

- Dataset quality
- Class balance
- Hyperparameter choices
- Evaluation metric priorities

Use the Comparison tab to compare metrics directly before deciding.

---

## 3. How long does NN training take?

Typical guideline from project notes:

- CPU: around 5 to 10 minutes for common local runs
- GPU: around 1 to 2 minutes for similar epoch budgets

Actual runtime varies by:

- Dataset size
- Epoch count
- Batch size
- Device and thermal limits

---

## 4. Can I use GPU training on a CPU-only machine?

No GPU acceleration is possible without CUDA-capable hardware and a compatible torch install.

However, NN trainer automatically falls back to CPU and continues to work.

---

## 5. What is the minimum NN size?

The baseline architecture is approximately 4,980 parameters.

This compact size is intentional for:

- Fast local iteration
- Low memory footprint
- Lightweight inference

---

## 6. Can I customize NN architecture?

You can tune several NN-related controls through the UI:

- Layers: 2 to 4
- Hidden Dim: 32 to 256
- Dropout: 0.0 to 0.5
- Learning Rate: 1e-4 to 1e-2
- Epochs: 5 to 100
- Batch Size: 16/32/64/128

Current implementation note:

- The core PasswordCNN architecture is fixed in code; some UI knobs are forward-compatible and intended for extended configurable variants.

---

## 7. How does ensemble voting work?

EnsemblePredictor uses soft voting:

1. Get probability outputs from active models
2. Normalize equal weights
3. Average probabilities
4. Apply threshold for final label

If a model is missing or fails prediction, it is skipped.

---

## 8. Why use ensemble over just NN?

Ensemble can be more stable because it combines different model biases:

- Phase 1/2 (sklearn-based patterns)
- NN (character-level CNN patterns)

Use ensemble when:

- Model outputs differ and you want consensus
- You prefer robustness over single-model variance

---

## 9. Why does training appear stalled after Pause?

Pause is cooperative. The trainer checks control signals at safe points in the epoch loop.

This means a short delay can occur before state changes to PAUSED.

---

## 10. Why is there no explicit early stopping option yet?

The current trainer does not include full patience-based early stopping logic.

Workaround:

- Monitor Val Loss and Val Accuracy live
- Stop manually when overfitting signs appear

---

## 11. Why is my loss not decreasing?

Common reasons:

- Learning rate too high or too low
- Data quality issues (noise, label issues)
- Underpowered configuration for current dataset

Try:

- 1e-3 to 5e-4 learning rate adjustment
- Increasing epochs moderately
- Reviewing data balance

---

## 12. Why do I get out-of-memory errors?

Most often caused by batch size too large for available memory.

Fix sequence:

1. Reduce batch size: 128 -> 64 -> 32 -> 16
2. Lower hidden dim
3. Close memory-heavy apps
4. Use CPU if GPU VRAM is insufficient

---

## 13. Why does Comparison show missing metrics for some models?

Comparison reads metadata from model-specific metadata.json files.

If metadata keys are absent for a run, some cells may show zeros or blanks.

---

## 14. Why does ensemble fail with "No models enabled"?

No model contributed probabilities.

Check:

- Did model loading succeed?
- Are file paths valid?
- Is at least one model checkbox enabled?

---

## 15. Is NN training reproducible?

It can be reproducible if seeds and environment are controlled, but exact repeatability depends on:

- Device type
- Library versions
- Data order and randomness

For stronger reproducibility, pin dependencies and set explicit seeds in training entrypoints.

---

## 16. Where are NN models stored?

Under models/nn:

- checkpoints/{run_id}/epoch_n.pt
- final/{run_id}/best_model.pt
- final/{run_id}/metadata.json
- history/{run_id}/metrics.csv

---

## 17. Can I load NN model directly for inference?

Yes.

You can load state_dict into PasswordCNN or use EnsemblePredictor.load_nn_model.

---

## 18. How do I choose threshold for hacked prediction?

Default threshold is 0.5.

Adjust based on risk:

- Lower threshold for higher recall
- Higher threshold for higher precision

Use validation data to pick a threshold aligned with your operational goals.

---

## 19. Does NN backend change API server behavior?

Not by default.

NN backend is primarily integrated through the training UI and model utilities. Existing API behavior remains as-is unless extended explicitly.

---

## 20. Is this production-ready?

The NN backend is practical for local and experimental use, with explicit known limitations documented in release notes.

Use measured validation and monitoring before production decisions.

---

## 21. Quick Troubleshooting Commands

Verify torch:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Run app:

```bash
python -m streamlit run ai-resources/ui_app.py
```

Run NN-focused tests:

```bash
pytest tests/test_nn_trainer.py
pytest tests/test_nn_backend_regression.py
```

---

## 22. Where to Learn More

- NN_USER_GUIDE.md
- NN_API_REFERENCE.md
- NN_ARCHITECTURE.md
- NN_MIGRATION_GUIDE.md
- NN_RELEASE_NOTES.md
