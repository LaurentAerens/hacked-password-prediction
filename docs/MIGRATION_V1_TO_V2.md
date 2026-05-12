# Migration Guide: v1 → v2

This document provides guidance for users and developers migrating code from v1 (Azure-integrated) to v2 (open-source).

## Overview

**v1** was tightly coupled to Azure services and required credentials. **v2** is pure Python with open-source libraries, prioritizing accessibility and ease of deployment.

## Key Changes

### Removed Components

| v1 Component | Status | v2 Replacement |
|---|---|---|
| `use_azure_model.py` | ❌ Removed | Local model serialization (joblib) |
| `use_model.py` | ❌ Removed | Merged into modern_trainers/ modules |
| Azure SDK imports | ❌ Removed | Scikit-learn, XGBoost, Optuna |
| Cloud credentials | ❌ Not needed | Local-first architecture |

### Added in v2

| Component | Purpose | Status |
|---|---|---|
| `modern_trainers/` | Modular trainer implementations | 🚧 In development (Phase 1-2) |
| Optuna integration | Hyperparameter optimization | ✅ Available |
| MLflow support | Experiment tracking | ✅ Available |
| Local model registry | Model persistence | ✅ Local joblib storage |

## Migration Steps

### For Model Training Scripts

**Before (v1 with Azure):**
```python
from ai_resources.use_azure_model import AzureModelTrainer
trainer = AzureModelTrainer(azure_creds=...)
model = trainer.train(data)
```

**After (v2 open-source):**
```python
# Coming: modern_trainers module
from ai_resources.modern_trainers import SklearnTrainer
trainer = SklearnTrainer()
model = trainer.train(data)
```

### For Model Inference

**Before (v1):**
```python
from use_model import AzurePredictor
predictor = AzurePredictor(model_id="...")
predictions = predictor.predict(data)
```

**After (v2):**
```python
import joblib
model = joblib.load("models/best_model.pkl")
predictions = model.predict(data)
```

### For Hyperparameter Tuning

**Before (v1):** Manual grid search or Azure ML studio

**After (v2):** Use Optuna for automated HPO
```python
import optuna
from ai_resources.modern_trainers import OptunaTuner

study = optuna.create_study()
study.optimize(objective_function, n_trials=100)
```

## Configuration

### Environment Variables

v2 uses only standard Python environment variables:

```bash
# Data paths
DATA_DIR=./ai-resources/data
MODEL_OUTPUT_DIR=./models

# Logging
LOG_LEVEL=INFO

# Optuna (optional)
OPTUNA_STORAGE=sqlite:///optuna.db
```

No Azure secrets or credentials required!

## Backwards Compatibility

❌ v1 and v2 code are **not compatible**. You must:

1. ✅ Choose **v2 (main branch)** for new projects
2. ✅ Keep **v1 (archive/v1-azure)** for maintaining legacy Azure integrations
3. ✅ Use git branches to manage both if needed

## Common Issues & Troubleshooting

### Issue: ImportError for `use_azure_model`

**Cause**: You're on v2 main, which doesn't have this module.

**Solution**: Either checkout v1 (`git checkout archive/v1-azure`) or refactor to use modern_trainers.

### Issue: Azure SDK dependencies not found

**Cause**: v2 doesn't require Azure SDK.

**Solution**: Remove Azure imports; install v2 requirements: `pip install -r requirements/requirements.txt`

### Issue: Model persistence incompatible

**Cause**: v1 used Azure blob storage; v2 uses local joblib.

**Solution**: Retrain models with v2 trainers and save locally.

## Getting Help

- Check [README.md](../README.md) for v2 setup
- See archived [README_DEPRECATED_V1.md](../README_DEPRECATED_V1.md) for v1 info
- Open an issue for questions

---

**Last Updated**: May 2026  
**Status**: 🚧 This guide will expand as Phase 1-2 development completes.
