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

```bash
"C:\\Program Files\\Python314\\python.exe" -m streamlit run ai-resources/ui_app.py
```

Do not run `ui_app.py` directly with `python`. Streamlit apps should be started with `-m streamlit run`.

### API

Run the API server:

```bash
python application.py
```

**Note**: The API is under development. 🚧

---

**Current Status**: Under active development - Phase 1-2 in progress.  
**Last Updated**: May 2026

