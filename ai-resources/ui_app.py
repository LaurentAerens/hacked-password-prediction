"""H.A.R.P. v2 Streamlit UI.

Run with:
    streamlit run ai-resources/ui_app.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow imports from ai-resources when launched from repository root.
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from adaptive_trainer import train_with_adaptive_search
from data_generation import generate_data
from shared_lib.data_utils import get_data


DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

EVENTS_CSV = DATA_DIR / "events.csv"
COMBINED_CSV = DATA_DIR / "combined_data.csv"
MODEL_FILE = MODELS_DIR / "phase1_best_model.pkl"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

            :root {
                --harp-text: #121826;
                --harp-text-muted: #334155;
                --harp-surface: #ffffff;
                --harp-surface-soft: #f8f6ef;
                --harp-border: #d9d3c4;
                --harp-focus: #0f766e;
                --harp-accent: #8b6f3d;
            }

            html, body, [class*="css"]  {
                font-family: 'Space Grotesk', sans-serif;
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 8% 16%, rgba(255, 231, 186, 0.40) 0%, rgba(0, 0, 0, 0) 34%),
                    radial-gradient(circle at 90% 8%, rgba(214, 232, 255, 0.28) 0%, rgba(0, 0, 0, 0) 36%),
                    linear-gradient(135deg, #fcfbf7 0%, #f8fbff 58%, #f9f8f3 100%);
            }

            [data-testid="stAppViewContainer"],
            [data-testid="stSidebar"] {
                color: var(--harp-text);
            }

            h1, h2, h3, h4, h5, h6,
            p, label, li,
            [data-testid="stMarkdownContainer"],
            [data-testid="stWidgetLabel"] p,
            [data-testid="stCaptionContainer"],
            [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"],
            [data-testid="stSidebar"] * {
                color: var(--harp-text);
            }

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-baseweb="select"] *,
            [data-testid="stNumberInput"] input {
                color: var(--harp-text) !important;
            }

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-baseweb="select"] > div,
            [data-testid="stNumberInput"] input {
                background: var(--harp-surface) !important;
                border: 1px solid var(--harp-border) !important;
            }

            [data-testid="stSidebar"] {
                background: rgba(255, 255, 255, 0.96);
                border-right: 1px solid var(--harp-border) !important;
            }

            [data-testid="stButton"] > button {
                color: #ffffff !important;
                border: 1px solid #0d5f59 !important;
            }

            [data-testid="stButton"] > button[kind="secondary"] {
                background: var(--harp-surface) !important;
                color: var(--harp-text) !important;
                border: 1px solid var(--harp-border) !important;
            }

            [data-testid="stButton"] > button:focus-visible,
            [data-testid="stTextInput"] input:focus,
            [data-testid="stTextArea"] textarea:focus,
            [data-baseweb="select"] > div:focus-within {
                outline: 2px solid var(--harp-focus) !important;
                outline-offset: 1px;
                box-shadow: none !important;
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }

            .harp-hero {
                border-radius: 16px;
                padding: 1rem 1.2rem;
                background: linear-gradient(115deg, #fffdf6 0%, #fff8e7 42%, #f7f6f1 100%);
                border: 1px solid var(--harp-border);
                color: var(--harp-text);
                box-shadow: 0 10px 28px rgba(50, 45, 30, 0.10);
                margin-bottom: 1rem;
            }

            .harp-hero-title,
            .harp-hero-subtitle {
                color: var(--harp-text) !important;
            }

            .harp-hero-title {
                margin: 0;
                letter-spacing: 0.2px;
            }

            .harp-hero-subtitle {
                margin: 0.2rem 0 0 0;
                color: var(--harp-text-muted) !important;
            }

            .harp-card {
                border: 1px solid var(--harp-border);
                border-radius: 14px;
                background: var(--harp-surface);
                padding: 0.8rem 1rem;
                color: var(--harp-text);
            }

            [data-testid="stDataFrame"],
            [data-testid="stTable"] {
                background: var(--harp-surface) !important;
                border: 1px solid var(--harp-border);
                border-radius: 12px;
            }

            [data-testid="stDataFrame"] * {
                color: var(--harp-text) !important;
            }

            [data-testid="stPlotlyChart"] {
                background: var(--harp-surface);
                border: 1px solid var(--harp-border);
                border-radius: 12px;
                padding: 0.35rem;
            }

            [data-testid="stAlert"] {
                color: var(--harp-text) !important;
            }

            [data-testid="stPlotlyChart"] .js-plotly-plot .plotly text,
            [data-testid="stPlotlyChart"] .js-plotly-plot .plotly .xtick text,
            [data-testid="stPlotlyChart"] .js-plotly-plot .plotly .ytick text,
            [data-testid="stPlotlyChart"] .js-plotly-plot .plotly .gtitle {
                fill: var(--harp-text) !important;
            }

            .mono {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.9rem;
                color: var(--harp-text-muted);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def _dataset_overview(csv_path: str) -> dict:
    path = Path(csv_path)
    if not path.exists():
        return {"exists": False}

    df = pd.read_csv(path)
    if "target" not in df.columns:
        return {
            "exists": True,
            "rows": len(df),
            "columns": list(df.columns),
            "target_ones": 0,
            "target_zeros": 0,
        }

    target_counts = df["target"].value_counts(dropna=False).to_dict()
    return {
        "exists": True,
        "rows": len(df),
        "columns": list(df.columns),
        "target_ones": int(target_counts.get(1, 0)),
        "target_zeros": int(target_counts.get(0, 0)),
    }


def _run_data_generation(generated_data_factor: int) -> str:
    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        generate_data(str(EVENTS_CSV), str(COMBINED_CSV), generated_data_factor)
    return log_stream.getvalue()


def _run_training(top_percent: float, cv_folds: int) -> tuple[dict, str]:
    df = get_data(str(COMBINED_CSV))
    x_data = df["password"].astype(str).tolist()
    y_data = df["target"].tolist()

    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        result = train_with_adaptive_search(
            x_data,
            y_data,
            models=None,
            preprocessors=None,
            cv_folds=cv_folds,
            top_percent=top_percent,
            n_jobs=-1,
            random_state=42,
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    results_name = f"phase1_gridsearch_{result['experiment_name']}.csv"
    result["results_df"].to_csv(RESULTS_DIR / results_name, index=False)
    joblib.dump(result["best_model"], MODEL_FILE)

    return result, log_stream.getvalue()


def _predict_with_model(model, password: str) -> tuple[int, float | None]:
    pred = int(model.predict([password])[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([password])[0]
        confidence = float(np.max(proba))
        return pred, confidence

    if hasattr(model, "decision_function"):
        score = float(model.decision_function([password])[0])
        confidence = float(1.0 / (1.0 + np.exp(-abs(score))))
        return pred, confidence

    return pred, None


def _show_overview() -> None:
    st.markdown('<div class="harp-hero"><h2 class="harp-hero-title">🪉 H.A.R.P. v2 Control Center</h2><p class="harp-hero-subtitle">Open-source password risk modeling with adaptive search and local execution.</p></div>', unsafe_allow_html=True)

    overview = _dataset_overview(str(COMBINED_CSV))

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Ready", "Yes" if overview.get("exists") else "No")
    col2.metric("Rows", overview.get("rows", 0))
    col3.metric("Model File", "Ready" if MODEL_FILE.exists() else "Missing")

    if overview.get("exists") and "target_ones" in overview:
        fig = px.pie(
            names=["Cracked / Positive", "Synthetic / Negative"],
            values=[overview.get("target_ones", 0), overview.get("target_zeros", 0)],
            title="Class Balance",
            color=["Cracked / Positive", "Synthetic / Negative"],
            color_discrete_map={
                "Cracked / Positive": "#ef476f",
                "Synthetic / Negative": "#118ab2",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="harp-card"><strong>Fast Start</strong><br/>1) Generate dataset<br/>2) Train Adaptive Phase 1 model<br/>3) Score passwords in Predict tab</div>', unsafe_allow_html=True)


def _show_data_generation_tab() -> None:
    st.subheader("Generate Dataset")
    st.write("Build or rebuild combined training data from events + synthetic negatives.")

    factor = st.slider("Negative sample factor", min_value=1, max_value=8, value=3)

    if not EVENTS_CSV.exists():
        st.error(f"Missing input file: {EVENTS_CSV}")
        return

    if st.button("Generate Data", type="primary"):
        with st.spinner("Generating dataset locally..."):
            try:
                logs = _run_data_generation(generated_data_factor=factor)
                st.success("Data generation complete.")
                st.text_area("Generation logs", logs, height=220)
                st.cache_data.clear()
            except Exception as exc:
                st.error(f"Data generation failed: {exc}")


def _show_training_tab() -> None:
    st.subheader("Train Phase 1 (Adaptive HPO)")
    st.write("Two-phase search: quick screening on all combinations, then full CV on top candidates.")

    if not COMBINED_CSV.exists():
        st.warning("Combined dataset not found. Run Data Generation first.")
        return

    top_percent = st.slider("Top candidates to advance", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
    cv_folds = st.selectbox("Full CV folds", options=[3, 4, 5], index=2)

    if st.button("Start Adaptive Training", type="primary"):
        with st.spinner("Training in progress. This may take several minutes..."):
            try:
                result, logs = _run_training(top_percent=top_percent, cv_folds=cv_folds)
                metrics = result["metrics"]
                st.success("Training completed.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Best AUC", f"{metrics['best_auc']:.4f}")
                m2.metric("Screened", metrics["combinations_screened"])
                m3.metric("Fully Evaluated", metrics["combinations_fully_evaluated"])
                time_saved = int(100 * (1 - metrics["combinations_fully_evaluated"] / max(1, metrics["combinations_screened"])))
                m4.metric("Estimated Time Saved", f"{time_saved}%")

                st.markdown("#### Best Configuration")
                st.json(result["best_config"])

                results_df = result["results_df"].copy()
                if not results_df.empty:
                    st.dataframe(results_df.head(25), use_container_width=True)
                    chart = px.bar(
                        results_df.head(15),
                        x="best_auc",
                        y=results_df.head(15).apply(lambda r: f"{r['model']} + {r['preprocessor']}", axis=1),
                        orientation="h",
                        title="Top 15 Model + Preprocessor Combinations",
                        color="best_auc",
                        color_continuous_scale="Viridis",
                    )
                    chart.update_layout(yaxis_title="Combination", xaxis_title="AUC")
                    st.plotly_chart(chart, use_container_width=True)

                st.text_area("Training logs", logs, height=280)
            except Exception as exc:
                st.error(f"Training failed: {exc}")


def _show_predict_tab() -> None:
    st.subheader("Predict Password Risk")

    if not MODEL_FILE.exists():
        st.warning("No trained model found yet. Train Phase 1 first.")
        return

    model = joblib.load(MODEL_FILE)
    password = st.text_input("Password to evaluate", value="")

    if st.button("Predict", type="primary"):
        if not password.strip():
            st.info("Enter a password first.")
            return

        pred, confidence = _predict_with_model(model, password.strip())
        if pred == 1:
            st.error("Prediction: HIGH RISK")
        else:
            st.success("Prediction: LOW RISK")

        if confidence is not None:
            st.write(f"Confidence: {confidence:.1%}")
        else:
            st.write("Confidence: unavailable for this model type.")


def main() -> None:
    st.set_page_config(
        page_title="H.A.R.P. v2 UI",
        page_icon="🪉",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.sidebar.title("🪉 H.A.R.P. v2")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Data Generation", "Adaptive Training", "Predict"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Model path")
    st.sidebar.markdown(f"<div class='mono'>{MODEL_FILE}</div>", unsafe_allow_html=True)

    if page == "Overview":
        _show_overview()
    elif page == "Data Generation":
        _show_data_generation_tab()
    elif page == "Adaptive Training":
        _show_training_tab()
    elif page == "Predict":
        _show_predict_tab()


if __name__ == "__main__":
    main()
