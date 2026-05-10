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

In 2021 during COVID, I set up an SSH honeypot to collect real-world password attempts. This data is now used as a dataset for password strength classification.

Original data: [SSH-findings](https://github.com/anakwaboe4/SSH-findings) project

## Goal

Simple classification task: predict if a given string would likely be in the list of passwords attempted on an SSH honeypot.

## Usage

Install requirements:

```bash
pip install -r requirements.txt
```

### Manual

Run the trainer:

```bash
python ai-resources/main.py
```

Run the new UI (Python + Streamlit):

If you installed the UI with the helper (`--with-ui`), Streamlit is installed into `.venv-ui` to avoid protobuf conflicts with TensorFlow. Run the app using the venv Python:

```powershell
.venv-ui\Scripts\python.exe -m streamlit run ai-resources/ui_app.py
```

Or activate the venv and run Streamlit normally:

```powershell
.venv-ui\Scripts\Activate.ps1
streamlit run ai-resources/ui_app.py
```

Do not run `ui_app.py` directly with `python`. Streamlit apps should be started with `-m streamlit run`.

If you see an error like "ImportError: cannot import name 'builder' from 'google.protobuf.internal'", it means your current Python environment has an incompatible `protobuf` version; use the `.venv-ui` environment (above) so Streamlit uses a compatible protobuf.

### API

Run the API server:

```bash
python application.py
```

**Note**: The API is under development. 🚧

---

**Automatic, hardware-aware install**

There is a helper installer that detects your hardware and installs the best-matching requirements (pins `protobuf` for compatibility):

- Auto-detect and install (recommended):

```bash
python scripts/install.py --auto
```

- Force Intel-optimized build:

```bash
python scripts/install.py --intel
```

- Force generic CPU build:

```bash
python scripts/install.py --cpu
```

- Force GPU build (ensure CUDA/cuDNN installed):

```bash
python scripts/install.py --gpu
```

Internally the script uses `constraints.txt` to pin `protobuf` to a compatible version.

Note: Streamlit (UI) depends on a newer `protobuf` than TensorFlow 2.11 allows. To avoid conflicts we split packages:

- Core/training dependencies: `requirements-core.txt` (used by hardware-specific installs)
- UI dependencies: `requirements-ui.txt` (install separately if you want the Streamlit app)

Install the UI into a separate virtual environment:

```bash
python -m venv .venv-ui
.venv-ui\Scripts\activate
pip install -r requirements-ui.txt
```

---

**Current Status**: Under active development - Phase 1-2 in progress.  
**Last Updated**: May 2026

