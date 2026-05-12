# H.A.R.P. - Hacked Account Risk Predictor

```
/ ',        ,--,
`;, '------'  @
 //'\-\-\-'\.'/
|| \ \ \ \,' /
||  \ \ \'  /
 \\  \,\   /
  \\,'    /
   `-----'
Tostig
```

## ℹ️ About v2 (Current Branch)

**v2 is open-source only** — No cloud vendor lock-in, no Azure credentials required. Run this project locally with just Python and open-source libraries.

- ✅ **Phase 1-2 focus**: Modern ML frameworks, local training, open-source tools
- ✅ **No credentials needed**: No Azure subscriptions, API keys, or cloud setup
- ✅ **Pure Python**: Scikit-learn, XGBoost, optuna, and more
- ⚠️ **v1 archived**: If you need the old Azure-integrated version, check out `archive/v1-azure` branch

### Migration from v1?

See [docs/MIGRATION_V1_TO_V2.md](docs/MIGRATION_V1_TO_V2.md) for upgrade guidance.

---

## Background

In 2021 during COVID, I set up an SSH honeypot mainly as a college experiment. The captured real-world password attempts are now used as a dataset for password strength classification.

Original data: [SSH-findings](https://github.com/anakwaboe4/SSH-findings) project

## Goal

Simple classification task: predict if a given string would likely be in the list of passwords attempted on an SSH honeypot.

## Usage

### Quick Start (Recommended)

**Setup (one-time):**

```bash
# Clone and cd into the repo
git clone https://github.com/yourusername/hacked-password-prediction.git
cd hacked-password-prediction

# Create a virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install the package (with automatic hardware detection)
pip install -e .
python install/install.py --auto
```

**Run the training CLI:**

```bash
harp-train
```

Or manually invoke via Python:

```bash
python -m harp.main
```

**Run the Streamlit UI:**

```bash
python -m streamlit run src/harp/ui_app.py
```

### Customization

**For GPU support:**

```bash
pip install -r requirements/requirements-gpu.txt
```

**For Intel optimization:**

```bash
pip install -r requirements/requirements-intel.txt
```

**For development (with testing tools):**

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Project Structure

```
.
├── src/harp/              # Source code (package)
│   ├── main.py            # CLI trainer
│   ├── ui_app.py          # Streamlit UI
│   ├── adaptive_trainer.py # HPO engine
│   ├── shared_lib/        # Common utilities
│   └── modern_trainers/   # Trainer implementations
├── data/                  # CSV datasets
├── results/               # Generated artifacts (training checkpoints, logs)
├── tests/                 # Test suite
├── debug/                 # Debug/experimental scripts
├── docs/                  # Documentation
├── install/               # Installation helpers
├── pyproject.toml         # Package configuration
├── requirements/          # Dependency pins and constraints
└── README.md              # This file
```

---

## API Server (In Development)

The API is under development and not yet functional.

---

## Documentation

- [Neural Network Guide](docs/NN_USER_GUIDE.md) — NN training, API reference, FAQ
- [Training Controls](docs/TRAINING_CONTROLS_USER_GUIDE.md) — Phase 1/2 HPO control signals
- [GPU Acceleration](docs/GPU_ACCELERATION_GUIDE.md) — GPU setup and benchmarks
- [v1→v2 Migration](docs/MIGRATION_V1_TO_V2.md) — Upgrading from v1

---

## Contributing

Current status: Phase 1-2 active development.  
Last updated: May 2026

