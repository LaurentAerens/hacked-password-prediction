# Neural Network Backend User Guide

Version: 1.0  
Date: 2026-05-08  
Audience: Developers, data scientists, and power users running H.A.R.P. locally

---

## 1. Overview

### 1.1 What Is the Neural Network Backend?

The Neural Network (NN) backend is an additional model training path in H.A.R.P. v2 that runs alongside the existing adaptive sklearn workflow.

It introduces a compact PyTorch CNN classifier designed for password risk prediction:

- Character-level tokenization (ASCII 0-255)
- Embedding dimension: 8
- CNN branches with kernel sizes: 2, 3, 4
- Binary output (hacked vs safe)
- Approximately 4,980 trainable parameters

This backend is intentionally lightweight for local development:

- Fast startup
- Small on-disk footprint
- Good CPU performance
- Optional GPU acceleration with automatic device detection

### 1.2 Why Use NN Alongside Phase 1 and Phase 2?

The NN backend is not a replacement for existing phases. It is a parallel track that gives you another modeling perspective.

Use NN when you want:

- A deep-learning baseline for character-level patterns
- A model that does not depend on handcrafted feature engineering
- Better ensemble diversity with Phase 1 and Phase 2
- A rapid local experiment cycle with telemetry and controls

Use Phase 1/2 when you want:

- Familiar sklearn explainability and search space control
- Existing workflows that are already validated in your setup

Use all three when you want:

- Head-to-head comparison in the Comparison tab
- Soft-voting ensemble predictions
- More robust decisions via model diversity

### 1.3 Quick Start (5 Minutes)

1. Install Python dependencies:

```bash
pip install -r requirements.txt
pip install torch
```

2. Launch Streamlit UI:

```bash
python -m streamlit run ai-resources/ui_app.py
```

3. In the sidebar, open Neural Network Training.

4. Keep defaults:

- Layers: 2
- Hidden Dim: 64
- Dropout: 0.2
- Learning Rate: 1e-3
- Epochs: 20
- Batch Size: 32

5. Click Train NN and monitor Training Progress.

6. After completion:

- Review Val Loss, Val Accuracy, Best Epoch
- Open Comparison tab to compare against other phases

---

## 2. Getting Started

### 2.1 Install Dependencies

Current requirements file includes core ML packages and Streamlit. PyTorch must be installed explicitly.

Recommended installation flow:

```bash
pip install -r requirements.txt
pip install torch
```

Optional CUDA-enabled torch install depends on your CUDA runtime and wheel source. If unsure, start with CPU torch and verify flow first.

### 2.2 Verify PyTorch and Device

Run this quick check:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"
```

Expected behavior:

- If cuda_available is True, trainer can use GPU
- If False, trainer automatically runs on CPU

### 2.3 Check GPU Availability in App Runtime

The NN trainer uses automatic device detection through GPUManager.detect_device().

In the UI Training Progress section, Device metric shows:

- GPU when CUDA available
- CPU otherwise

### 2.4 Run Your First NN Training

1. Ensure combined dataset exists:

- Data generation tab must produce ai-resources/data/combined_data.csv

2. Open Neural Network Training tab.

3. Configure hyperparameters.

4. Click Train NN.

5. Monitor:

- Epoch progress
- Train loss
- Validation loss
- Validation accuracy
- Event log

6. After completion, use Download Model to export best checkpoint.

### 2.5 CLI Training Example

You can also train without UI:

```python
import pandas as pd
from nn_trainer import PasswordNNTrainer


df = pd.read_csv("ai-resources/data/combined_data.csv")
trainer = PasswordNNTrainer(model_dir="models/nn")

result = trainer.train(
    X=df["password"],
    y=df["target"],
    epochs=20,
    batch_size=32,
    learning_rate=1e-3,
)

print(result["best_metrics"])
print(result["checkpoint_path"])
```

---

## 3. Configuration Guide

This section explains NN controls exposed in the UI and how they affect quality, stability, and speed.

### 3.1 Layers (2 to 4)

UI exposes Layers to represent dense-layer depth intent for configurable NN experiments.

Practical guidance:

- 2 layers: fastest, easiest to stabilize, strong default
- 3 layers: can model more complex interactions, moderate cost
- 4 layers: highest capacity among UI options, more overfitting risk

Trade-off summary:

- More layers increase expressive power
- More layers also increase optimization difficulty and variance
- Start at 2, increase only when validation metrics plateau

### 3.2 Hidden Dimensions (32 to 256)

Hidden dimension controls dense representation capacity.

Guidance:

- 32: low memory and fastest inference
- 64: balanced default for many datasets
- 128: higher capacity, moderate memory cost
- 256: strongest capacity, greatest risk of overfitting and slower training

Rule of thumb:

- If train and validation both underfit, increase hidden dim
- If train improves but validation degrades, reduce hidden dim or increase regularization

### 3.3 Learning Rate (1e-4 to 1e-2)

Learning rate has the highest impact on training stability.

Recommended sequence:

- Start: 1e-3
- If loss oscillates/diverges: reduce to 5e-4 or 1e-4
- If loss is very slow but stable: try 5e-3 carefully

Failure patterns:

- Too high: validation loss spikes, unstable updates
- Too low: little progress per epoch, long runtime

### 3.4 Epochs (5 to 100)

Epoch count sets maximum training length.

Guidance:

- 5 to 20: good for smoke tests and quick iteration
- 20 to 50: normal tuning range
- 50 to 100: for difficult datasets after confirming generalization

Validation-based stopping guidance:

- Watch Val Loss curve
- If Val Loss flattens for several epochs, additional epochs may not help
- If Val Loss increases while Train Loss decreases, stop early manually

Note:

- Automated early stopping is not fully implemented in current trainer
- Use manual judgment from live curves and metrics

### 3.5 Dropout (0.0 to 0.5)

Dropout mitigates overfitting in dense layers.

Guidance:

- 0.0: no regularization, fastest fitting, highest overfit risk
- 0.2 to 0.3: balanced default zone
- 0.4 to 0.5: strong regularization, may underfit if data is small/noisy

When to increase dropout:

- Training accuracy rises but validation stalls or worsens

When to decrease dropout:

- Both train and validation remain weak

### 3.6 Batch Size (16, 32, 64, 128)

Batch size affects memory, throughput, and optimization noise.

Guidance by device:

- CPU:
  - 16 or 32 usually best for responsiveness
- GPU:
  - 64 or 128 often improves throughput if memory allows

Trade-offs:

- Larger batch:
  - Faster wall-clock throughput
  - Smoother gradients
  - Higher memory use
- Smaller batch:
  - Lower memory use
  - Noisier gradients that can sometimes improve generalization

---

## 4. Training UI

### 4.1 End-to-End Steps

1. Go to Data Generation and build combined dataset.
2. Open Neural Network Training.
3. Expand NN Configuration.
4. Set hyperparameters.
5. Click Train NN.
6. Observe status, progress, and metrics.
7. Use Pause/Resume/Stop if needed.
8. Review final metrics and curves.
9. Download model artifact if required.

### 4.2 Live Progress Explained

Training Progress section includes:

- Status metric:
  - RUNNING, PAUSED, STOPPED
- Device metric:
  - GPU or CPU
- Progress bar:
  - Based on completed epoch count
- Metrics cards:
  - Train Loss
  - Val Loss
  - Val Accuracy
- Event Log:
  - Recent NN telemetry events

### 4.3 Pause, Resume, Stop

Controls are backed by ControlSignal.

Behavior:

- Pause:
  - Requests pause at a safe boundary
  - Worker waits in paused state
- Resume:
  - Continues from paused state
- Stop:
  - Requests stop and exits training loop

Operational notes:

- These controls are cooperative and state-driven
- Expect short delay between click and state transition

### 4.4 Early Stopping

Current status:

- Explicit patience-based early stopping is not implemented in PasswordNNTrainer

Recommended workflow today:

- Monitor validation metrics live
- Pause/Stop manually when overfitting begins

---

## 5. Understanding Results

### 5.1 Metrics Explained

Loss:

- BCEWithLogitsLoss objective
- Lower is better
- Tracks probability calibration quality for binary classification

Accuracy:

- Fraction of correct predictions at threshold 0.5
- Easy to read but can hide class imbalance issues

Precision:

- Of predicted hacked passwords, how many were truly hacked
- Important when false alarms are costly

Recall:

- Of truly hacked passwords, how many were found
- Important when misses are costly

F1:

- Harmonic mean of precision and recall
- Useful when you need balanced detection performance

### 5.2 Reading Loss Curves

Healthy pattern:

- Train Loss decreases
- Val Loss decreases or stabilizes near train loss

Overfit pattern:

- Train Loss continues downward
- Val Loss turns upward

Underfit pattern:

- Both train and validation losses remain high and flat

### 5.3 When NN Can Perform Better Than Phase 1/2

Common cases:

- Character-level patterns not well captured by handcrafted features
- Nonlinear interactions across local token patterns
- Mixed syntactic structures where CNN n-gram filters help

### 5.4 When NN Can Perform Worse Than Phase 1/2

Common cases:

- Small datasets where simpler models generalize better
- Highly engineered features where sklearn models excel
- Poor hyperparameter settings (lr, dropout, epochs)

### 5.5 Inference Speed Implications

Current target characteristics:

- Small model (roughly 5k parameters)
- Typical inference is fast on CPU
- Batch inference improves throughput

Practical implications:

- Good fit for local batch scoring
- Usually acceptable latency for interactive single-password checks

---

## 6. Model Comparison

### 6.1 Using the Comparison Tab

1. Train one or more models across Phase 1, Phase 2, and NN.
2. Open Comparison page in sidebar.
3. Build and inspect table:

- Phase
- Run ID
- Path
- Accuracy
- Precision
- Recall
- F1
- Params
- Training Time

4. Use chart view to compare metrics by phase.

### 6.2 Soft Voting Ensemble

EnsemblePredictor performs soft voting by averaging class probabilities from enabled models.

Key points:

- Equal weight per active model
- Missing models are skipped gracefully
- Final probability is weighted mean of available model probabilities

### 6.3 Trusting Ensemble vs Individual Models

Use ensemble when:

- Models disagree and you want stability
- You have at least two reasonably calibrated models

Use individual models when:

- One model clearly dominates across your target metric
- You require deterministic behavior from a single architecture

### 6.4 Threshold Selection

Threshold controls binary decision from hacked probability.

Default:

- 0.5

Tuning examples:

- Higher threshold (0.7): fewer positives, higher precision, lower recall
- Lower threshold (0.3): more positives, higher recall, lower precision

Choose threshold from your risk profile:

- Minimize misses: lower threshold
- Minimize false alarms: higher threshold

---

## 7. Advanced Topics

### 7.1 GPU Training and Mixed Precision

GPUManager supports:

- detect_device()
- get_batch_size(device)
- get_mixed_precision_context(device)

Current trainer behavior:

- Device auto-detection is active
- Mixed precision context helper exists and can be integrated further

### 7.2 CPU Optimization

If CPU-only:

- Use batch size 16 or 32
- Start with 20 epochs and evaluate curves before extending
- Prefer lower hidden dim for faster iteration

### 7.3 Saving and Loading Models

During training:

- Epoch checkpoints saved under models/nn/checkpoints/{run_id}/epoch_{n}.pt

After training:

- Best model saved under models/nn/final/{run_id}/best_model.pt
- Metadata saved to models/nn/final/{run_id}/metadata.json
- History saved to models/nn/history/{run_id}/metrics.csv

Loading best model via registry:

```python
from nn_registry import NNModelRegistry

registry = NNModelRegistry("models/nn")
model, metadata = registry.load_best_model(run_id="your_run_id")
print(metadata["metrics"])
```

### 7.4 Inference on New Passwords

```python
import pandas as pd
from ensemble import EnsemblePredictor

ensemble = EnsemblePredictor()
ensemble.load_nn_model("models/nn/final/your_run_id/best_model.pt")

passwords = pd.Series(["P@ssw0rd!", "correcthorsebatterystaple"])
proba = ensemble.predict_proba(passwords)
pred = ensemble.predict(passwords, threshold=0.5)

print(proba)
print(pred)
```

---

## 8. Troubleshooting

### 8.1 "Training is slow"

Checks:

- Verify device shown in progress panel
- If CPU, reduce batch size to 16 or 32
- Reduce epochs for quick validation pass
- Reduce hidden dimension if experimenting with larger setups

### 8.2 "Loss is not decreasing"

Try:

- Lower learning rate (1e-3 to 5e-4 or 1e-4)
- Increase epochs moderately
- Check dataset label quality and balance
- Ensure correct input column mapping and preprocessing

### 8.3 "Out of memory"

Try:

- Reduce batch size (128 to 64, 32, or 16)
- Reduce hidden dimension
- Close other memory-heavy applications
- Fall back to CPU if GPU memory is constrained

### 8.4 "PyTorch not found"

Install torch:

```bash
pip install torch
```

Re-check:

```bash
python -c "import torch; print(torch.__version__)"
```

### 8.5 "No models enabled for ensemble"

Cause:

- No model loaded successfully in EnsemblePredictor

Fix:

- Train at least one phase
- Ensure model path exists
- Ensure model type matches loader (joblib for phase1/phase2, .pt for NN)

### 8.6 "Comparison table is empty"

Checks:

- Verify models are saved under expected folders
- Ensure metadata.json exists for each run
- Run registry scan again by reopening Comparison tab

---

## 9. Operational Best Practices

- Start with default NN config and establish baseline metrics.
- Change one hyperparameter at a time.
- Track each run_id and retain metadata for reproducibility.
- Compare NN against Phase 1 and Phase 2 before promoting changes.
- Use ensemble only after each component model is validated.

---

## 10. Quick Command Reference

Install:

```bash
pip install -r requirements.txt
pip install torch
```

Run UI:

```bash
python -m streamlit run ai-resources/ui_app.py
```

Run API:

```bash
python application.py
```

Run lightweight trainer:

```bash
python ai-resources/main.py
```

---

## 11. Implementation Notes and Scope Honesty

This guide reflects current code behavior.

Current limitations to be aware of:

- Explicit early stopping logic is not yet implemented in PasswordNNTrainer
- UI exposes some architecture-oriented controls as forward-compatible knobs
- Some telemetry events are defined in design docs but only a subset is emitted in the current trainer runtime

For details on internals, see:

- NN_API_REFERENCE.md
- NN_ARCHITECTURE.md
- NN_RELEASE_NOTES.md
