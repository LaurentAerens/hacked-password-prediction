"""H.A.R.P. v2 Streamlit UI.

Run with:
    streamlit run ai-resources/ui_app.py
"""

from __future__ import annotations

import io
import sys
import uuid
import time
import threading
import ctypes
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st

# Try to import torch (optional, for backward compatibility)
try:
    import torch
except ImportError:
    torch = None

# Allow imports from ai-resources when launched from repository root.
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from adaptive_trainer import train_with_adaptive_search
from data_generation import generate_data
from gpu_utils import GPUDetector

# Try to import nn_trainer (optional, requires torch)
try:
    from nn_trainer import PasswordNNTrainer
except ImportError:
    PasswordNNTrainer = None

from shared_lib.telemetry_emitter import TelemetryEmitter
from shared_lib.data_utils import get_data
from shared_lib.realtime_dashboard import RealtimeDashboardState
from shared_lib.control_signal import ControlSignal
from shared_lib.checkpoint_manager import CheckpointManager
from ensemble import EnsemblePredictor
from model_registry import UnifiedModelRegistry
from model_comparison import ModelComparison


DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ROOT_MODELS_DIR = BASE_DIR.parent / "models"
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

            .harp-filter-label {
                font-size: 0.7rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--harp-text-muted);
                margin: 0.6rem 0 0.2rem 0;
                display: block;
            }

            .harp-stat-value {
                font-size: 1rem;
                color: var(--harp-text-muted);
                margin: 0 0 0.75rem 0;
                line-height: 1.4;
            }

            .harp-stat-value strong {
                font-size: 1.25rem;
                color: var(--harp-text);
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def _load_explorer_data(csv_path: str) -> pd.DataFrame | None:
    """Load and enrich dataset for the explorer (cached)."""
    path = Path(csv_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["password"] = df["password"].astype(str)
    df["length"] = df["password"].str.len()
    return df


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


def _run_data_generation(
    generated_data_factor: int,
    source_path: str | None = None,
    password_column: str = "password",
) -> str:
    """Run data generation, writing logs to a string. source_path defaults to EVENTS_CSV."""
    effective_path = source_path if source_path is not None else str(EVENTS_CSV)
    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        generate_data(effective_path, str(COMBINED_CSV), generated_data_factor, password_column)
    return log_stream.getvalue()


def _run_training(top_percent: float, cv_folds: int,
                 run_phase2: bool = False,
                 phase2_top_n: int = 3,
                 phase2_trials_per_model: int = 20,
                 max_parallel_candidates: Optional[int] = None,
                 ram_per_candidate_gb: float = 2.0,
                 cpu_utilization_target: float = 0.9,
                 progress_callback: Optional[Callable] = None,
                 run_id: Optional[str] = None,
                 control_signal: Optional[ControlSignal] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None,
                 fast_mode: bool = False,
                 mini_dataset: bool = False,
                 mini_dataset_size: int = 100,
                 gpu_device: str = 'auto') -> tuple[dict, str]:
    """
    Run training with optional progress callback and control signal for pause/stop.
    
    Args:
        top_percent: Fraction of candidates to advance to phase 1b
        cv_folds: Number of CV folds for phase 1b
        progress_callback: Optional callback to receive ProgressEvent dicts in real-time
        run_id: Optional run identifier for telemetry. Generated if None.
        control_signal: Optional ControlSignal for pause/stop/resume control
        checkpoint_manager: Optional CheckpointManager for saving checkpoints on pause
        run_phase2: Whether to run Optuna Phase 2 fine-tuning
        phase2_top_n: Number of top phase1 candidates to fine-tune
        phase2_trials_per_model: Number of Optuna trials per candidate
        max_parallel_candidates: Optional cap for concurrent model+preprocessor jobs
        ram_per_candidate_gb: Estimated RAM usage per concurrent candidate
        cpu_utilization_target: Fraction of requested CPUs to use
    
    Returns:
        (results_dict, logs_string)
    """
    if run_id is None:
        run_id = str(uuid.uuid4())
    
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
            progress_callback=progress_callback,
            run_id=run_id,
            control_signal=control_signal,
            checkpoint_manager=checkpoint_manager,
            run_phase2=run_phase2,
            phase2_top_n=phase2_top_n,
            phase2_trials_per_model=phase2_trials_per_model,
            max_parallel_candidates=max_parallel_candidates,
            ram_per_candidate_gb=ram_per_candidate_gb,
            cpu_utilization_target=cpu_utilization_target,
            fast_mode=fast_mode,
            mini_dataset=mini_dataset,
            mini_dataset_size=mini_dataset_size,
            gpu_device=gpu_device,
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    results_name = f"phase1_gridsearch_{result['experiment_name']}.csv"
    result["results_df"].to_csv(RESULTS_DIR / results_name, index=False)
    joblib.dump(result["best_model"], MODEL_FILE)

    # Persist Phase 2 artifacts and keep all fine-tuned candidate models.
    if run_phase2 and "phase2_results_df" in result and not result["phase2_results_df"].empty:
        phase2_results_name = f"phase2_optuna_{result['experiment_name']}.csv"
        result["phase2_results_df"].to_csv(RESULTS_DIR / phase2_results_name, index=False)

        phase2_models_dir = MODELS_DIR / "phase2"
        phase2_models_dir.mkdir(parents=True, exist_ok=True)
        for key, model in (result.get("phase2_model_registry") or {}).items():
            safe_name = key.replace("__", "_")
            joblib.dump(model, phase2_models_dir / f"{safe_name}.pkl")

    return result, log_stream.getvalue()


def _training_worker(
    *,
    top_percent: float,
    cv_folds: int,
    run_phase2: bool,
    phase2_top_n: int,
    phase2_trials_per_model: int,
    max_parallel_candidates: Optional[int],
    ram_per_candidate_gb: float,
    cpu_utilization_target: float,
    run_id: str,
    control_signal: ControlSignal,
    checkpoint_manager: CheckpointManager,
    dashboard_state: RealtimeDashboardState,
    job_state: dict,
    fast_mode: bool = False,
    mini_dataset: bool = False,
    mini_dataset_size: int = 100,
    gpu_device: str = 'auto',
) -> None:
    """Background worker that runs training and stores outputs in job_state."""

    def progress_callback(event):
        dashboard_state.on_event(event)

    try:
        result, logs = _run_training(
            top_percent=top_percent,
            cv_folds=cv_folds,
            run_phase2=run_phase2,
            phase2_top_n=phase2_top_n,
            phase2_trials_per_model=phase2_trials_per_model,
            max_parallel_candidates=max_parallel_candidates,
            ram_per_candidate_gb=ram_per_candidate_gb,
            cpu_utilization_target=cpu_utilization_target,
            progress_callback=progress_callback,
            run_id=run_id,
            control_signal=control_signal,
            checkpoint_manager=checkpoint_manager,
            fast_mode=fast_mode,
            mini_dataset=mini_dataset,
            mini_dataset_size=mini_dataset_size,
            gpu_device=gpu_device,
        )
        job_state["result"] = result
        job_state["logs"] = logs
    except Exception as exc:
        job_state["error"] = str(exc)
    finally:
        job_state["done"] = True


def _hard_stop_thread(worker: Optional[threading.Thread]) -> bool:
    """Best-effort hard stop for a Python thread via async exception injection."""
    if worker is None or not worker.is_alive() or worker.ident is None:
        return False

    thread_id = worker.ident
    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id), ctypes.py_object(SystemExit)
    )

    if result == 1:
        return True

    # Revert if multiple threads were affected.
    if result > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
    return False


def _nn_training_worker(
    passwords: pd.Series,
    labels: pd.Series,
    config: dict,
    control_signal: ControlSignal,
    telemetry_emitter: TelemetryEmitter,
    job_state: dict,
) -> None:
    """Background worker for NN training. Runs in thread, updates job_state."""
    import traceback
    try:
        if PasswordNNTrainer is None:
            raise ImportError("torch is required for Neural Network Training. Install pytorch to enable this feature.")
        
        trainer = PasswordNNTrainer()
        
        result = trainer.train(
            X=passwords,
            y=labels,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
            control_signal=control_signal,
            telemetry_emitter=telemetry_emitter,
        )
        
        job_state["result"] = result
        job_state["status"] = "completed"
    except Exception as exc:
        full_traceback = traceback.format_exc()
        print(f"[NN Training Error]\n{full_traceback}", flush=True)
        job_state["error"] = full_traceback
        job_state["status"] = "failed"
    finally:
        job_state["done"] = True



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


def _get_latest_available_models() -> dict[str, str]:
    """Return latest available model paths keyed by display label."""
    available: dict[str, str] = {}

    if MODEL_FILE.exists():
        available["Phase 1 (Adaptive)"] = str(MODEL_FILE)

    model_roots = [MODELS_DIR]
    if ROOT_MODELS_DIR != MODELS_DIR:
        model_roots.append(ROOT_MODELS_DIR)

    for model_root in model_roots:
        try:
            registry = UnifiedModelRegistry(base_dir=str(model_root))
            latest = registry.get_latest_models()
        except Exception:
            continue

        phase2_model = latest.get("phase2", {}).get("path")
        if phase2_model and "Phase 2 (Optuna)" not in available:
            available["Phase 2 (Optuna)"] = str(phase2_model)

        nn_model = latest.get("nn", {}).get("path")
        if nn_model and "Neural Network" not in available:
            available["Neural Network"] = str(nn_model)

    return available


def _predict_with_nn_model(nn_model_path: str, password: str) -> tuple[int, float]:
    """Run single-password prediction with NN via EnsemblePredictor wrapper."""
    ensemble = EnsemblePredictor()
    ensemble.load_nn_model(nn_model_path)

    proba = ensemble.predict_proba(pd.Series([password]))[0]
    hacked_proba = float(proba[1])
    pred = int(hacked_proba >= 0.5)
    confidence = hacked_proba if pred == 1 else float(proba[0])
    return pred, confidence


def _show_nn_training_tab() -> None:
    """Show NN training tab with live progress and controls."""
    st.subheader("Phase 3: Neural Network Classifier")
    st.write("Train independent neural network in parallel with sklearn phases.")

    if not COMBINED_CSV.exists():
        st.warning("Combined dataset not found. Run Data Generation first.")
        return
    
    # Load data
    df = pd.read_csv(COMBINED_CSV)
    
    # Initialize session state keys for NN training
    for key in [
        "nn_training_state", "nn_config", "nn_control_signal", "nn_telemetry_emitter",
        "nn_history", "nn_training_active", "nn_training_job", "nn_training_thread"
    ]:
        if key not in st.session_state:
            if key == "nn_training_state":
                st.session_state[key] = {}
            elif key == "nn_config":
                st.session_state[key] = {}
            elif key == "nn_control_signal":
                st.session_state[key] = None
            elif key == "nn_telemetry_emitter":
                st.session_state[key] = None
            elif key == "nn_history":
                st.session_state[key] = {}
            elif key in ["nn_training_active", "nn_training_job", "nn_training_thread"]:
                st.session_state[key] = None if key != "nn_training_active" else False
    
    # Configuration Section
    with st.expander("🔧 NN Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            nn_config = {
                "num_layers": st.slider("Layers", 2, 4, 2, help="Dense layers count", key="nn_layers"),
                "hidden_dim": st.slider("Hidden Dim", 32, 256, 64, step=32, key="nn_hidden_dim"),
                "dropout": st.slider("Dropout", 0.0, 0.5, 0.2, step=0.1, key="nn_dropout"),
                "learning_rate": st.select_slider(
                    "Learning Rate",
                    [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                    value=1e-3,
                    format_func=lambda x: f"{x:.0e}",
                    key="nn_lr"
                ),
            }
        
        with col2:
            nn_config.update({
                "epochs": st.slider("Epochs", 5, 100, 20, step=5, key="nn_epochs"),
                "batch_size": st.selectbox(
                    "Batch Size",
                    [16, 32, 64, 128],
                    index=1,
                    help="Auto-adjust for GPU/CPU",
                    key="nn_batch_size"
                ),
                "tokenization": st.selectbox(
                    "Tokenization",
                    ["Character-level Embedding"],
                    help="Character-level (MVP), subword/patterns in Wave 3b+",
                    key="nn_tokenization"
                ),
            })
        
        st.session_state.nn_config = nn_config
    
    # Training Controls Section
    st.markdown("### Training Controls")
    col_train, col_pause, col_resume, col_stop = st.columns(4)
    
    with col_train:
        train_nn_button = st.button("▶ Train NN", key="btn_train_nn", use_container_width=True)
    with col_pause:
        pause_nn_button = st.button("⏸ Pause", key="btn_pause_nn", disabled=not st.session_state.nn_training_active, use_container_width=True)
    with col_resume:
        resume_nn_button = st.button("▶ Resume", key="btn_resume_nn", disabled=not st.session_state.nn_training_active, use_container_width=True)
    with col_stop:
        stop_nn_button = st.button("⏹ Stop", key="btn_stop_nn", disabled=not st.session_state.nn_training_active, use_container_width=True)
    
    # Train button logic
    if train_nn_button:
        if not st.session_state.nn_training_active:
            from shared_lib.control_signal import ControlSignal
            
            control_signal = ControlSignal()
            emitter = TelemetryEmitter()
            
            st.session_state.nn_control_signal = control_signal
            st.session_state.nn_telemetry_emitter = emitter
            st.session_state.nn_training_active = True
            st.session_state.nn_training_state = {"status": "starting"}
            
            job_state = {"done": False, "result": None, "error": None, "status": "running"}
            st.session_state.nn_training_job = job_state
            
            worker = threading.Thread(
                target=_nn_training_worker,
                args=(
                    df["password"],
                    df["target"],
                    nn_config,
                    control_signal,
                    emitter,
                    job_state,
                ),
                daemon=True,
            )
            worker.start()
            st.session_state.nn_training_thread = worker
            st.rerun()
    
    # Handle pause/resume/stop buttons
    if st.session_state.nn_training_active:
        control_signal = st.session_state.nn_control_signal
        
        if pause_nn_button and control_signal:
            control_signal.request_pause()
            st.info("Pause requested...")
            st.rerun()
        
        if resume_nn_button and control_signal:
            control_signal.resume()
            st.info("Resuming...")
            st.rerun()
        
        if stop_nn_button and control_signal:
            control_signal.request_stop()
            st.session_state.nn_training_active = False
            st.warning("Stop requested...")
            st.rerun()
    
    # Live Training Progress Section
    st.markdown("### Training Progress")
    
    # Update progress display if training is active
    if st.session_state.nn_training_active:
        emitter = st.session_state.nn_telemetry_emitter
        control_signal = st.session_state.nn_control_signal
        
        if emitter:
            events = emitter.get_events(limit=100)
            
            # Extract latest epoch event
            latest_epoch_event = None
            for event in reversed(events):
                if event.get("event_type") == "nn.epoch.completed":
                    latest_epoch_event = event
                    break
            
            # Status and device
            col_status, col_device = st.columns([0.7, 0.3])
            with col_status:
                state = control_signal.get_state() if control_signal else "STOPPED"
                status_emoji = {"RUNNING": "🟢", "PAUSED": "🟡", "STOPPED": "🔴"}.get(state, "⚪")
                st.metric("Status", f"{status_emoji} {state}")
            with col_device:
                # Use GPU detection utility instead of torch
                is_gpu_available, gpu_type = GPUDetector.detect_gpu_availability()
                device = gpu_type.upper() if is_gpu_available else "CPU"
                st.metric("Device", device)
            
            # Epoch progress bar
            if latest_epoch_event:
                epoch = latest_epoch_event.get("metrics", {}).get("epoch", 0)
                total_epochs = nn_config.get("epochs", 1)
                progress = epoch / total_epochs if total_epochs else 0
                st.progress(progress, text=f"Epoch {epoch + 1}/{total_epochs}")
                
                # Metrics display
                col_m1, col_m2, col_m3 = st.columns(3)
                metrics = latest_epoch_event.get("metrics", {})
                with col_m1:
                    st.metric("Train Loss", f"{metrics.get('train_loss', 0):.4f}")
                with col_m2:
                    st.metric("Val Loss", f"{metrics.get('val_loss', 0):.4f}")
                with col_m3:
                    st.metric("Val Accuracy", f"{metrics.get('val_acc', 0):.2%}")
            
            # Event log
            st.markdown("### Event Log")
            if events and len(events) > 0:
                event_log_text = "\n".join([
                    f"[{e.get('emitted_at', 'N/A')}] {e.get('event_type', 'unknown')} ({e.get('status', '')})"
                    for e in events[-15:]
                ])
                st.code(event_log_text, language="text")
            else:
                st.info("⏳ Waiting for training events...")
            
            # Keep UI live with shorter sleep for responsiveness
            time.sleep(0.5)
            st.rerun()
        else:
            st.info("⏳ Initializing training emitter...")
    
    # Check if training completed
    if st.session_state.get("nn_training_job"):
        job_state = st.session_state.nn_training_job
        if job_state.get("done"):
            st.session_state.nn_training_active = False
            
            if job_state.get("error"):
                st.error(f"❌ Training failed: {job_state['error']}")
            else:
                result = job_state.get("result")
                if result:
                    st.session_state.nn_training_state = {"status": "completed", "result": result}
                    st.rerun()
    
    # Display results after training complete
    if st.session_state.nn_training_state.get("status") == "completed":
        result = st.session_state.nn_training_state.get("result", {})
        
        st.markdown("---")
        st.markdown("### ✅ Training Complete!")
        st.success("✅ Training finished successfully.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Val Loss", f"{result.get('best_metrics', {}).get('val_loss', 0):.4f}")
        with col2:
            st.metric("Val Accuracy", f"{result.get('best_metrics', {}).get('val_acc', 0):.2%}")
        with col3:
            st.metric("Best Epoch", result.get("best_epoch", 0))
        with col4:
            st.metric("Training Time", f"{result.get('training_time_sec', 0):.1f}s")
        
        # Model info
        with st.expander("📊 Model Info"):
            model_info = {
                "Checkpoint": result.get("checkpoint_path", ""),
                "Device": result.get("device", ""),
            }
            st.json(model_info)
        
        # Download model
        if result.get("checkpoint_path") and Path(result["checkpoint_path"]).exists():
            with open(result["checkpoint_path"], "rb") as f:
                st.download_button(
                    label="Download Model",
                    data=f.read(),
                    file_name="nn_model.pt",
                    mime="application/octet-stream"
                )
        
        # Loss/accuracy curves
        history = result.get("history", {})
        if history and "epoch" in history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(history["epoch"], history.get("train_loss", []), label="Train Loss")
            ax1.plot(history["epoch"], history.get("val_loss", []), label="Val Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(history["epoch"], history.get("val_acc", []))
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.grid(True)
            
            st.pyplot(fig)



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
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="harp-card"><strong>Fast Start</strong><br/>1) Generate dataset<br/>2) Train Adaptive Phase 1 model<br/>3) Score passwords in Predict tab</div>', unsafe_allow_html=True)


_SENTINELS = {"-", "<No Pass>", "<Any Pass>", "(none)"}


def _extract_passwords(df: pd.DataFrame, col: str) -> list[str]:
    """Pull clean, non-sentinel password strings from a dataframe column."""
    series = df[col].astype(str).str.strip()
    series = series[~series.isin(_SENTINELS)]
    series = series[series != "nan"]
    series = series[series != ""]
    return series.drop_duplicates().tolist()


def _col_default_index(columns: list[str]) -> int:
    lower = [c.lower() for c in columns]
    if "password" in lower:
        return lower.index("password")
    match = next((i for i, c in enumerate(lower) if "pass" in c), None)
    return match if match is not None else 0


def _source_card(df: pd.DataFrame, label: str, col_key: str) -> tuple[str, bool]:
    """Render a single source card. Returns (selected_column, has_error)."""
    cols = df.columns.tolist()
    lower = [c.lower() for c in cols]
    default = _col_default_index(cols)

    card_left, card_right = st.columns([1, 2])
    with card_left:
        st.markdown(f'<span class="harp-filter-label">{label}</span>', unsafe_allow_html=True)
        chosen_col = st.selectbox(
            "column", cols, index=default,
            label_visibility="collapsed", key=col_key,
        )
        # Inline validation
        has_error = False
        sample_df = df.head(500)
        series = sample_df[chosen_col].astype(str).str.strip()
        valid = series[~series.isin(_SENTINELS) & (series != "nan") & (series != "")]
        null_pct = 100 * (1 - len(valid) / max(1, len(sample_df)))
        if null_pct > 40:
            st.warning(f"⚠️ ~{null_pct:.0f}% of sampled rows are empty/sentinel — only real values will be used.")
        if len(valid) == 0:
            st.error(f"❌ No usable values in '{chosen_col}'.")
            has_error = True
        elif len(valid) < 50:
            st.warning(f"⚠️ Only {len(valid)} usable rows in sample.")
        if df[chosen_col].dtype in ("int64", "float64"):
            st.warning(f"⚠️ '{chosen_col}' looks numeric — are you sure it contains passwords?")

    with card_right:
        samples = df[chosen_col].astype(str).str.strip()
        samples = samples[~samples.isin(_SENTINELS) & (samples != "nan") & (samples != "")].head(6).tolist()
        sample_html = "".join(f'<span class="mono">{v}</span><br/>' for v in samples)
        st.markdown(
            f'<div class="harp-card"><span class="harp-filter-label">SAMPLE VALUES</span><br/>'
            f'{sample_html if sample_html else "<em>No usable values</em>"}</div>',
            unsafe_allow_html=True,
        )
    return chosen_col, has_error


def _show_data_generation_tab() -> None:
    st.markdown(
        '<div class="harp-hero"><h2 class="harp-hero-title">🧬 Data Generation</h2>'
        '<p class="harp-hero-subtitle">'
        'Load one or more CSVs, pick the password column for each, then synthesise negatives — all offline.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    # ── Built-in toggle ────────────────────────────────────────────────────────
    include_builtin = st.toggle(
        f"Include built-in **events.csv**" + (" ✅" if EVENTS_CSV.exists() else " ❌ (file missing)"),
        value=EVENTS_CSV.exists(),
        disabled=not EVENTS_CSV.exists(),
    )

    # ── Multi-file uploader ────────────────────────────────────────────────────
    st.markdown('<span class="harp-filter-label">UPLOAD ADDITIONAL CSV FILES</span>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # ── Parse uploads ──────────────────────────────────────────────────────────
    parsed_uploads: list[tuple[str, pd.DataFrame]] = []   # (display_name, df)
    for uf in uploaded_files:
        try:
            df_up = pd.read_csv(uf)
            if len(df_up) == 0:
                st.error(f"❌ **{uf.name}** has no data rows — skipped.")
            else:
                parsed_uploads.append((uf.name, df_up))
        except Exception as exc:
            st.error(f"❌ Could not parse **{uf.name}**: {exc} — skipped.")

    # ── Nothing to work with ───────────────────────────────────────────────────
    has_any_source = include_builtin or bool(parsed_uploads)
    if not has_any_source:
        st.info("Enable the built-in file or upload at least one CSV to continue.")

    # ── Per-source column pickers ──────────────────────────────────────────────
    selected_columns: dict[str, str] = {}   # source_key → column name
    source_dfs: dict[str, pd.DataFrame] = {}
    any_error = False

    if include_builtin and EVENTS_CSV.exists():
        st.markdown("---")
        st.markdown("**Source 1 — Built-in events.csv**")
        try:
            builtin_df = pd.read_csv(EVENTS_CSV)
            n_rows = len(builtin_df)
        except Exception as exc:
            st.error(f"❌ Could not read events.csv: {exc}")
            include_builtin = False
            builtin_df = pd.DataFrame()
            n_rows = 0
        if include_builtin:
            st.caption(f"📄 {EVENTS_CSV}  ·  {n_rows:,} rows  ·  {len(builtin_df.columns)} columns")
            col, err = _source_card(builtin_df, "PASSWORD COLUMN — events.csv", "dg_col_builtin")
            selected_columns["__builtin__"] = col
            source_dfs["__builtin__"] = builtin_df
            any_error = any_error or err

    for i, (fname, df_up) in enumerate(parsed_uploads, start=2 if include_builtin else 1):
        st.markdown("---")
        st.markdown(f"**Source {i} — {fname}**")
        st.caption(f"📄 {fname}  ·  {len(df_up):,} rows  ·  {len(df_up.columns)} columns")
        col, err = _source_card(df_up, f"PASSWORD COLUMN — {fname}", f"dg_col_upload_{i}")
        source_dfs[fname] = df_up
        selected_columns[fname] = col
        any_error = any_error or err

    # ── Combined summary ───────────────────────────────────────────────────────
    if has_any_source and not any_error:
        all_passwords: list[str] = []
        for src_key, col in selected_columns.items():
            all_passwords.extend(_extract_passwords(source_dfs[src_key], col))
        # Deduplicate across sources
        all_passwords = list(dict.fromkeys(all_passwords))
        n_sources = len(selected_columns)
        st.markdown("---")
        sm1, sm2 = st.columns(2)
        sm1.metric("Total unique passwords across all sources", f"{len(all_passwords):,}")
        sm2.metric("Sources configured", n_sources)
        if len(all_passwords) < 100:
            st.warning("⚠️ Fewer than 100 unique passwords — consider adding more sources.")

    # ── Negative sample factor ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<span class="harp-filter-label">NEGATIVE SAMPLE FACTOR</span>', unsafe_allow_html=True)
    factor = st.slider("Negative sample factor", min_value=1, max_value=8, value=3, label_visibility="collapsed")
    st.caption("Each real password generates N synthetic negatives. Factor 3 = 3× negatives per positive.")

    # ── Generate ───────────────────────────────────────────────────────────────
    generate_blocked = not has_any_source or any_error or (has_any_source and len(selected_columns) == 0)
    if st.button("Generate Data", type="primary", disabled=generate_blocked):
        # Re-collect passwords at click time (session state may have changed)
        merged: list[str] = []
        for src_key, col in selected_columns.items():
            merged.extend(_extract_passwords(source_dfs[src_key], col))
        merged = list(dict.fromkeys(merged))
        if not merged:
            st.error("❌ No usable passwords collected from the selected sources.")
        else:
            log_stream = io.StringIO()
            try:
                with st.spinner(f"Generating dataset from {len(merged):,} unique passwords..."):
                    with redirect_stdout(log_stream):
                        from data_generation import generate_data_from_listofcrackedpasswords
                        COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
                        generate_data_from_listofcrackedpasswords(merged, str(COMBINED_CSV), factor)
                st.success("✅ Data generation complete.")
                if COMBINED_CSV.exists():
                    result_df = pd.read_csv(COMBINED_CSV)
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Rows written", f"{len(result_df):,}")
                    mc2.metric("Positives (cracked)", f"{int((result_df['target'] == 1).sum()):,}")
                    mc3.metric("Negatives (synthetic)", f"{int((result_df['target'] == 0).sum()):,}")
                st.text_area("Generation logs", log_stream.getvalue(), height=220)
                st.cache_data.clear()
            except Exception as exc:
                st.error(f"❌ Data generation failed: {exc}")



def _show_training_tab() -> None:
    st.subheader("Train Phase 1 (Adaptive HPO)")
    st.write("Two-phase search: quick screening on all combinations, then full CV on top candidates.")

    if not COMBINED_CSV.exists():
        st.warning("Combined dataset not found. Run Data Generation first.")
        return

    # ⚡ Quick Training Options (prominent at top)
    st.markdown("### ⚡ Quick Training Options")
    quick_col1, quick_col2 = st.columns(2)
    
    with quick_col1:
        fast_mode = st.checkbox(
            "🚀 Fast Mode",
            value=False,
            help="Use 2-3 values per hyperparameter instead of 3-6. ~2x faster with slightly lower accuracy. Ideal for quick iteration.",
        )
    
    with quick_col2:
        use_mini_dataset = st.checkbox(
            "📊 Mini Dataset",
            value=False,
            help="Use smaller dataset for quick validation. Completes in 30 seconds.",
        )
    
    # Show mini dataset size slider if mini dataset is selected
    if use_mini_dataset:
        mini_dataset_size = st.slider(
            "Mini dataset size (samples)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of balanced samples (50/50 weak/strong passwords) for quick validation.",
        )
        mini_dataset = True
    else:
        mini_dataset_size = 100  # Default size when not using mini dataset
        mini_dataset = False

    # Standard training parameters
    st.markdown("### Training Configuration")
    top_percent = st.slider("Top candidates to advance", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
    cv_folds = st.selectbox("Full CV folds", options=[3, 4, 5], index=2)
    run_phase2 = st.checkbox("Run Phase 2 Optuna fine-tuning", value=True)
    phase2_top_n = st.slider("Phase 2: top candidates", min_value=1, max_value=8, value=3, step=1, disabled=not run_phase2)
    phase2_trials_per_model = st.slider("Phase 2: trials per candidate", min_value=5, max_value=80, value=20, step=5, disabled=not run_phase2)

    with st.expander("Performance tuning (CPU/RAM)"):
        cpu_utilization_target = st.slider(
            "CPU utilization target",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Fraction of available CPUs to actively schedule.",
        )
        ram_per_candidate_gb = st.slider(
            "Estimated RAM per concurrent candidate (GB)",
            min_value=0.5,
            max_value=8.0,
            value=2.0,
            step=0.5,
            help="Higher value is safer for memory-heavy models; lower can increase parallelism.",
        )
        max_parallel_candidates_raw = st.number_input(
            "Max parallel model+preprocessor jobs (0 = auto)",
            min_value=0,
            max_value=256,
            value=0,
            step=1,
            help="Set 0 to let the scheduler choose automatically based on CPU and RAM.",
        )
        max_parallel_candidates = None if int(max_parallel_candidates_raw) == 0 else int(max_parallel_candidates_raw)
        
        # GPU acceleration options
        st.markdown("**GPU Acceleration**")
        
        is_gpu_available, gpu_type = GPUDetector.detect_gpu_availability()
        if is_gpu_available:
            gpu_status = f"✅ GPU Available ({gpu_type.upper()})"
            device_options = ['auto', gpu_type, 'cpu']
        else:
            gpu_status = "❌ No GPU detected (will use CPU)"
            device_options = ['cpu']
        
        st.info(gpu_status)
        
        # Determine default index (always first option: 'auto' or 'cpu')
        default_index = 0
        
        gpu_device = st.selectbox(
            "Device for GPU-accelerated models",
            options=device_options,
            index=default_index,
            help="XGBoost and LightGBM can use GPU for 3-5x speedup. 'auto' detects CUDA/ROCm/MPS automatically.",
        )

    # Initialize realtime dashboard state if not already present
    if "training_run_id" not in st.session_state:
        st.session_state["training_run_id"] = None
    if "training_state" not in st.session_state:
        st.session_state["training_state"] = None
    if "training_logs" not in st.session_state:
        st.session_state["training_logs"] = ""
    if "control_signal" not in st.session_state:
        st.session_state["control_signal"] = None
    if "checkpoint_manager" not in st.session_state:
        st.session_state["checkpoint_manager"] = None
    if "training_active" not in st.session_state:
        st.session_state["training_active"] = False
    if "training_job" not in st.session_state:
        st.session_state["training_job"] = None
    if "training_thread" not in st.session_state:
        st.session_state["training_thread"] = None
    if "stop_confirm_armed" not in st.session_state:
        st.session_state["stop_confirm_armed"] = False

    if st.button("Start Adaptive Training", type="primary", disabled=st.session_state.get("training_active", False)):
        # Initialize control signal and checkpoint manager
        run_id = str(uuid.uuid4())
        control_signal = ControlSignal(run_id)
        checkpoint_manager = CheckpointManager(checkpoint_dir=str(RESULTS_DIR / "checkpoints"))
        
        # Store in session state for button handlers
        st.session_state["training_run_id"] = run_id
        st.session_state["control_signal"] = control_signal
        st.session_state["checkpoint_manager"] = checkpoint_manager
        st.session_state["training_state"] = RealtimeDashboardState(run_id=run_id, buffer_size=500)
        st.session_state["training_active"] = True
        st.session_state["training_result"] = None
        st.session_state["stop_confirm_armed"] = False

        dashboard_state = st.session_state["training_state"]
        job_state = {"done": False, "result": None, "logs": "", "error": None}
        st.session_state["training_job"] = job_state

        worker = threading.Thread(
            target=_training_worker,
            kwargs={
                "top_percent": top_percent,
                "cv_folds": cv_folds,
                "run_phase2": run_phase2,
                "phase2_top_n": phase2_top_n,
                "phase2_trials_per_model": phase2_trials_per_model,
                "max_parallel_candidates": max_parallel_candidates,
                "ram_per_candidate_gb": ram_per_candidate_gb,
                "cpu_utilization_target": cpu_utilization_target,
                "run_id": run_id,
                "control_signal": control_signal,
                "checkpoint_manager": checkpoint_manager,
                "dashboard_state": dashboard_state,
                "job_state": job_state,
                "fast_mode": fast_mode,
                "mini_dataset": mini_dataset,
                "mini_dataset_size": mini_dataset_size,
                "gpu_device": gpu_device,
            },
            daemon=True,
        )
        worker.start()
        st.session_state["training_thread"] = worker
        st.rerun()

    # Collect background training result when worker completes
    if st.session_state.get("training_active"):
        job_state = st.session_state.get("training_job")
        if job_state and job_state.get("done"):
            st.session_state["training_active"] = False
            st.session_state["training_thread"] = None
            st.session_state["training_logs"] = job_state.get("logs", "")
            st.session_state["stop_confirm_armed"] = False

            if job_state.get("error"):
                st.error(f"❌ Training failed: {job_state['error']}")
            else:
                st.session_state["training_result"] = job_state.get("result")

            st.rerun()

    # Display progress and logs continuously (during and after training)
    dashboard_state = st.session_state.get("training_state")
    if dashboard_state:
        status = dashboard_state.get_status_summary() or {}
        phase_label = dashboard_state.get_phase_label()
        phase_prog = status.get('phase_progress') or {}

        st.markdown("---")
        st.markdown("### 📊 Training Progress")

        col1, col2, col3 = st.columns(3)
        current = phase_prog.get('current', 0)
        total = phase_prog.get('total', 1)

        with col1:
            st.metric("Phase", f"Phase {phase_label}" if phase_label else "Pending")
        with col2:
            st.metric("Combinations", f"{current}/{total}")
        with col3:
            pct = int(100 * current / max(1, total)) if total > 0 else 0
            st.metric("Progress", f"{pct}%")

        if total > 0:
            st.progress(pct / 100, text=f"Phase {phase_label}: {current}/{total} combinations ({pct}%)")

        st.markdown("### 🎯 Model Status Board")
        status_board = status.get('status_board') or {}
        if status_board:
            for model, preprocessors in status_board.items():
                for preprocessor, state_info in preprocessors.items():
                    status_val = state_info.get('status') or state_info.get('state') or 'queued'
                    prog_info = state_info.get('progress', {})
                    curr_fold = prog_info.get('current_fold', 0)
                    total_folds = prog_info.get('total_folds', 0)
                    metrics_dict = state_info.get('metrics', {})

                    status_emoji = {
                        'queued': '⏳',
                        'running': '⚙️',
                        'completed': '✅',
                        'failed': '❌',
                    }.get(status_val, '❓')

                    fold_info = f"({curr_fold}/{total_folds})" if total_folds > 0 else ""
                    metric_parts = []
                    for k, v in metrics_dict.items():
                        if isinstance(v, float):
                            metric_parts.append(f"{k}={v:.4f}")
                        elif isinstance(v, int):
                            metric_parts.append(f"{k}={v}")
                    metric_str = " | ".join(metric_parts)
                    line = f"{status_emoji} **{model}** + {preprocessor} {fold_info}"
                    if metric_str:
                        line += f" | {metric_str}"
                    st.write(line)
        else:
            st.info("Waiting for first training events...")

        st.markdown("### 📋 Live Event Log")
        recent_logs = dashboard_state.get_recent_logs(count=30)
        if recent_logs:
            st.code("\n".join(recent_logs), language="text")
        else:
            st.caption("No events yet.")
    
    # Training control buttons (pause, stop, resume)
    control_signal = st.session_state.get("control_signal")
    if control_signal and st.session_state.get("training_active"):
        st.markdown("---")
        st.markdown("### 🎮 Training Controls")
        
        # Get current state
        current_state = control_signal.get_state() if control_signal else "STOPPED"
        
        # Status indicator
        status_emoji_map = {
            "RUNNING": "✅",
            "PAUSE_REQUESTED": "⏳",
            "PAUSED": "⏸️",
            "RESUME_REQUESTED": "▶️",
            "STOP_REQUESTED": "⏳",
            "STOPPED": "🛑",
        }
        status_color_map = {
            "RUNNING": "🟢",
            "PAUSE_REQUESTED": "🟡",
            "PAUSED": "🟡",
            "RESUME_REQUESTED": "🟢",
            "STOP_REQUESTED": "🔴",
            "STOPPED": "🔴",
        }
        
        status_emoji = status_emoji_map.get(current_state, "❓")
        status_color = status_color_map.get(current_state, "⚪")
        st.write(f"**Status**: {status_color} {status_emoji} {current_state}")
        
        # Control buttons
        col_pause, col_stop, col_resume = st.columns(3)
        
        with col_pause:
            if current_state == "RUNNING":
                if st.button("⏸️ PAUSE TRAINING", key="pause_btn", type="secondary"):
                    control_signal.request_pause()
                    st.info("Pause requested. Waiting for current combination to complete...")
            else:
                st.button("⏸️ PAUSE TRAINING", key="pause_btn_disabled", disabled=True, type="secondary")
        
        with col_stop:
            stop_armed = st.session_state.get("stop_confirm_armed", False)
            if stop_armed:
                if st.button("🧨 HARD STOP NOW", key="stop_btn_hard", type="primary"):
                    control_signal.request_stop()
                    worker = st.session_state.get("training_thread")
                    stopped = _hard_stop_thread(worker)

                    if stopped:
                        job_state = st.session_state.get("training_job")
                        if isinstance(job_state, dict):
                            job_state["error"] = "Hard stop requested by user"
                            job_state["done"] = True
                        st.session_state["training_active"] = False
                        st.session_state["training_thread"] = None
                        st.session_state["stop_confirm_armed"] = False
                        st.warning("Hard stop executed. Training was terminated immediately.")
                        st.rerun()
                    else:
                        st.error(
                            "Hard stop could not interrupt the worker instantly. "
                            "Graceful stop is still active and will stop after current step."
                        )
            elif current_state in ["RUNNING", "PAUSED", "PAUSE_REQUESTED", "STOP_REQUESTED"]:
                if not stop_armed:
                    if st.button("🛑 STOP TRAINING", key="stop_btn", type="primary"):
                        control_signal.request_stop()
                        st.session_state["stop_confirm_armed"] = True
                        st.warning("Stop requested. Press the same button again for immediate HARD STOP.")
                        st.rerun()
            else:
                st.button("🛑 STOP TRAINING", key="stop_btn_disabled", disabled=True, type="primary")
        
        with col_resume:
            if current_state == "PAUSED":
                if st.button("▶️ RESUME TRAINING", key="resume_btn", type="secondary"):
                    control_signal.resume()
                    st.session_state["stop_confirm_armed"] = False
                    st.info("Training resumed...")
            else:
                st.button("▶️ RESUME TRAINING", key="resume_btn_disabled", disabled=True, type="secondary")

        # Keep UI live while training thread runs
        time.sleep(0.8)
        st.rerun()
    
    
    # Display final results after training completes
    if st.session_state.get("training_result"):
        result = st.session_state["training_result"]
        st.markdown("---")
        st.markdown("### ✅ Training Complete!")
        st.success("✅ Training finished successfully.")
        
        # Summary metrics
        metrics = result["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Best AUC", f"{metrics['best_auc']:.4f}")
        m2.metric("🔍 Screened", metrics["combinations_screened"])
        m3.metric("🏆 Fully Evaluated", metrics["combinations_fully_evaluated"])
        time_saved = int(100 * (1 - metrics["combinations_fully_evaluated"] / max(1, metrics["combinations_screened"])))
        m4.metric("⚡ Time Saved", f"{time_saved}%")

        # Best configuration
        st.markdown("#### 🏅 Best Configuration")
        st.json(result["best_config"])

        # Results table and chart
        results_df = result["results_df"].copy()
        if not results_df.empty:
            st.dataframe(results_df.head(25), width='stretch')
            
            # Top combinations chart
            chart = px.bar(
                results_df.head(15),
                x="best_auc",
                y=results_df.head(15).apply(lambda r: f"{r['model']} + {r['preprocessor']}", axis=1),
                orientation="h",
                title="🏆 Top 15 Model + Preprocessor Combinations",
                color="best_auc",
                color_continuous_scale="Viridis",
            )
            chart.update_layout(yaxis_title="Combination", xaxis_title="AUC")
            st.plotly_chart(chart, width='stretch')

        phase2_df = result.get("phase2_results_df")
        if phase2_df is not None and not phase2_df.empty:
            st.markdown("#### 🔬 Phase 2 Fine-Tuning Results")
            st.dataframe(phase2_df.head(20), width='stretch')

            best_phase2 = phase2_df.iloc[0]
            st.success(
                f"Phase 2 best: {best_phase2['model']} + {best_phase2['preprocessor']} "
                f"(AUC={best_phase2['phase2_best_auc']:.4f})"
            )
        
        # Final training log
        if dashboard_state and dashboard_state.event_buffer:
            log_lines = dashboard_state.get_recent_logs(count=100)
            log_text = "\n".join(log_lines)
            st.text_area("📝 Full training event log", log_text, height=300)
        else:
            logs_text = st.session_state.get("training_logs", "")
            if logs_text:
                st.text_area("📝 Training logs", logs_text, height=300)


def _show_data_explorer() -> None:
    st.markdown(
        '<div class="harp-hero"><h2 class="harp-hero-title">🔍 Data Explorer</h2>'
        '<p class="harp-hero-subtitle">Filter, search, and inspect the training dataset.</p></div>',
        unsafe_allow_html=True,
    )

    df_full = _load_explorer_data(str(COMBINED_CSV))
    if df_full is None:
        st.warning("Combined dataset not found. Run Data Generation first.")
        return

    total_rows = len(df_full)
    max_len_global = min(64, int(df_full["length"].max()))

    # ── Initialise session state keys ───────────────────────────────────────
    if "de_search" not in st.session_state:
        st.session_state["de_search"] = ""
    if "de_class" not in st.session_state:
        st.session_state["de_class"] = "All"
    if "de_len" not in st.session_state:
        st.session_state["de_len"] = (1, max_len_global)
    if "de_maxrows" not in st.session_state:
        st.session_state["de_maxrows"] = 1_000

    # ── Filter panel + stats ─────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="harp-card">', unsafe_allow_html=True)
        st.markdown('<span class="harp-filter-label">Search</span>', unsafe_allow_html=True)
        search_val = st.text_input(
            "Search passwords", value=st.session_state["de_search"],
            placeholder="substring match, e.g.  admin  or  123",
            label_visibility="collapsed", key="de_search_widget",
        )
        st.markdown('<span class="harp-filter-label">Class</span>', unsafe_allow_html=True)
        class_val = st.selectbox(
            "Class", options=["All", "Cracked (1)", "Synthetic (0)"],
            index=["All", "Cracked (1)", "Synthetic (0)"].index(st.session_state["de_class"]),
            label_visibility="collapsed", key="de_class_widget",
        )
        st.markdown('<span class="harp-filter-label">Password length</span>', unsafe_allow_html=True)
        len_val = st.slider(
            "Password length", min_value=1, max_value=max_len_global,
            value=st.session_state["de_len"],
            label_visibility="collapsed", key="de_len_widget",
        )
        st.markdown('<span class="harp-filter-label">Max rows to display</span>', unsafe_allow_html=True)
        maxrows_val = st.number_input(
            "Max rows", min_value=50, max_value=50_000, value=st.session_state["de_maxrows"],
            step=50, label_visibility="collapsed", key="de_maxrows_widget",
        )
        if st.button("Reset filters", type="secondary", key="de_reset"):
            for k, v in [("de_search", ""), ("de_class", "All"),
                         ("de_len", (1, max_len_global)), ("de_maxrows", 1_000)]:
                st.session_state[k] = v
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply filters (order matters — metrics computed before max-row cap)
    df_filtered = df_full.copy()
    if class_val == "Cracked (1)":
        df_filtered = df_filtered[df_filtered["target"] == 1]
    elif class_val == "Synthetic (0)":
        df_filtered = df_filtered[df_filtered["target"] == 0]
    if search_val.strip():
        df_filtered = df_filtered[
            df_filtered["password"].str.contains(search_val.strip(), case=False, na=False, regex=False)
        ]
    lo, hi = len_val
    df_filtered = df_filtered[df_filtered["length"].between(lo, hi)]

    filtered_count = len(df_filtered)
    n_cracked = int((df_filtered["target"] == 1).sum())
    n_synthetic = int((df_filtered["target"] == 0).sum())

    with right:
        st.markdown('<div class="harp-card">', unsafe_allow_html=True)
        st.markdown(
            f'<p class="harp-stat-value">Showing <strong>{filtered_count:,}</strong>'
            f' of <strong>{total_rows:,}</strong> rows</p>',
            unsafe_allow_html=True,
        )
        bar_fig = px.bar(
            x=[n_cracked, n_synthetic],
            y=["Cracked", "Synthetic"],
            orientation="h",
            color=["Cracked", "Synthetic"],
            color_discrete_map={"Cracked": "#ef476f", "Synthetic": "#118ab2"},
            height=90,
        )
        bar_fig.update_layout(
            showlegend=False, margin=dict(l=70, r=10, t=4, b=4),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False, title=""),
            yaxis=dict(title=""),
        )
        st.plotly_chart(bar_fig, width='stretch')
        if filtered_count > 0:
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("Avg length", round(df_filtered["length"].mean(), 1), delta_color="off")
            lc2.metric("Min length", int(df_filtered["length"].min()), delta_color="off")
            lc3.metric("Max length", int(df_filtered["length"].max()), delta_color="off")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Summary metrics row ──────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Showing", f"{min(int(maxrows_val), filtered_count):,} / {filtered_count:,}")
    pct_c = f"{100 * n_cracked / max(1, filtered_count):.1f}% of filtered"
    pct_s = f"{100 * n_synthetic / max(1, filtered_count):.1f}% of filtered"
    m2.metric("Cracked", f"{n_cracked:,}", delta=pct_c, delta_color="off")
    m3.metric("Synthetic", f"{n_synthetic:,}", delta=pct_s, delta_color="off")

    # ── Table ────────────────────────────────────────────────────────────────
    df_display = df_filtered.head(int(maxrows_val))[["password", "length", "target"]]

    st.dataframe(
        df_display,
        width='stretch',
        height=420,
        hide_index=True,
        column_config={
            "password": st.column_config.TextColumn("Password", width="medium",
                help="Raw password string"),
            "length": st.column_config.NumberColumn("Length", format="%d chars",
                width="small"),
            "target": st.column_config.NumberColumn("Class (0=Synthetic, 1=Cracked)",
                width="small"),
        },
    )

    # ── Download ─────────────────────────────────────────────────────────────
    _, dl_col = st.columns([4, 1])
    with dl_col:
        st.download_button(
            label="Export filtered CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="harp_filtered_export.csv",
            mime="text/csv",
            type="secondary",
        )


def _show_predict_tab() -> None:
    st.subheader("🪉 Predict Password Risk")

    available_models = _get_latest_available_models()
    if not available_models:
        st.warning("No trained models found yet. Train Phase 1, Phase 2, or NN first.")
        return

    model_choice = st.selectbox(
        "Model to use",
        options=list(available_models.keys()),
        index=0,
        key="predict_model_choice",
    )
    selected_model_path = available_models[model_choice]
    st.caption(f"Using: {selected_model_path}")

    password = st.text_input("Password to evaluate", value="")

    if st.button("Predict", type="primary"):
        if not password.strip():
            st.info("Enter a password first.")
            return

        clean_password = password.strip()

        if model_choice == "Neural Network":
            pred, confidence = _predict_with_nn_model(selected_model_path, clean_password)
        else:
            model = joblib.load(selected_model_path)
            pred, confidence = _predict_with_model(model, clean_password)

        if pred == 1:
            st.error("Prediction: HIGH RISK")
        else:
            st.success("Prediction: LOW RISK")

        if confidence is not None:
            st.write(f"Confidence: {confidence:.1%}")
        else:
            st.write("Confidence: unavailable for this model type.")


def _show_overview() -> None:
    """Show overview page."""
    st.markdown(
        '<div class="harp-hero"><h1 class="harp-hero-title">🪉 H.A.R.P. v2</h1>'
        '<p class="harp-hero-subtitle">Hacked Password Risk Prediction</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "This system trains ensemble models to predict whether a password has been compromised in known breaches. "
        "It combines Phase 1 (Adaptive GridSearchCV), Phase 2 (Optuna fine-tuning), and Neural Network approaches."
    )
    st.markdown("Use the navigation menu to explore features.")


def _show_comparison_tab() -> None:
    """Show model comparison and ensemble prediction tab."""
    st.subheader("Model Comparison & Ensemble")
    st.write("Compare Phase 1, Phase 2, and NN model performance side-by-side, and make ensemble predictions.")
    
    # Load registry
    registry = UnifiedModelRegistry(base_dir=str(MODELS_DIR))
    comparison = ModelComparison(registry)
    
    # Build comparison table
    comp_df = comparison.build_comparison_table()
    
    if comp_df.empty:
        st.warning("No trained models available for comparison. Train Phase 1, Phase 2, or NN first.")
    else:
        # Metrics table
        st.markdown("### Performance Metrics")
        st.dataframe(comp_df, width='stretch')
        
        # Best model highlight
        st.markdown("### Best Model")
        best_by_acc = comparison.get_best_model_by_metric("Accuracy")
        if best_by_acc:
            st.success(f"Best by Accuracy: **{best_by_acc}**")
        
        # Comparison chart
        st.markdown("### Metrics Comparison")
        fig = comparison.plot_comparison()
        st.plotly_chart(fig, width='stretch')
        
        # Ensemble configuration
        st.markdown("### Ensemble Configuration")
        
        col_select, col_predict = st.columns([0.8, 0.2])
        
        with col_select:
            st.markdown("Select models to include in ensemble:")
            use_phase1 = st.checkbox("Phase 1 (Adaptive GridSearchCV)", value=True, key="cb_phase1")
            use_phase2 = st.checkbox("Phase 2 (Optuna Fine-Tuning)", value=True, key="cb_phase2")
            use_nn = st.checkbox("Neural Network", value=True, key="cb_nn")
        
        with col_predict:
            st.write("")  # Spacer
            st.write("")  # Spacer
            predict_button = st.button("Make Predictions", key="btn_ensemble_predict", type="primary")
        
        # Ensemble predictions
        if predict_button:
            ensemble = EnsemblePredictor()
            
            if use_phase1:
                try:
                    phase1_model_path = comp_df[comp_df["Phase"] == "PHASE1"]["Path"].iloc[0]
                    ensemble.load_phase1_model(phase1_model_path)
                except (IndexError, FileNotFoundError):
                    pass
            
            if use_phase2:
                try:
                    phase2_model_path = comp_df[comp_df["Phase"] == "PHASE2"]["Path"].iloc[0]
                    ensemble.load_phase2_model(phase2_model_path)
                except (IndexError, FileNotFoundError):
                    pass
            
            if use_nn:
                try:
                    nn_model_path = comp_df[comp_df["Phase"] == "NN"]["Path"].iloc[0]
                    ensemble.load_nn_model(nn_model_path)
                except (IndexError, FileNotFoundError):
                    pass
            
            st.markdown("### Ensemble Results")
            
            # Show ensemble config
            config = ensemble.ensemble_config
            st.metric("Active Models", f"{config['active_count']}/3")
            
            # Load test set
            if COMBINED_CSV.exists():
                df = pd.read_csv(COMBINED_CSV)
                if "test_passwords" not in st.session_state or len(st.session_state["test_passwords"]) == 0:
                    test_idx = np.random.choice(len(df), size=min(100, len(df)), replace=False)
                    st.session_state["test_passwords"] = df.iloc[test_idx]["password"].values
                
                # Predictions on test set
                if ensemble.get_active_model_count() > 0 and len(st.session_state["test_passwords"]) > 0:
                    try:
                        proba = ensemble.predict_proba(pd.Series(st.session_state["test_passwords"]))
                        preds = ensemble.predict(pd.Series(st.session_state["test_passwords"]))
                        
                        # Show sample predictions
                        result_df = pd.DataFrame({
                            "Password": st.session_state["test_passwords"][:10],
                            "Prob Not Hacked": proba[:10, 0],
                            "Prob Hacked": proba[:10, 1],
                            "Prediction": ["Hacked" if p else "Safe" for p in preds[:10]]
                        })
                        
                        st.dataframe(result_df, width='stretch')
                    except Exception as e:
                        st.error(f"Error making predictions: {e}")
                else:
                    st.warning("No active models or test data available")
            else:
                st.warning("Combined dataset not found. Run Data Generation first.")


def main() -> None:
    st.set_page_config(
        page_title="H.A.R.P. v2 UI",
        page_icon="🪉",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    st.sidebar.title("🪉 H.A.R.P. v2")
    
    # Build page list - exclude Neural Network Training if torch is not available
    pages = ["Overview", "Data Explorer", "Data Generation", "Adaptive Training", "Comparison", "Predict"]
    if PasswordNNTrainer is not None:
        pages.insert(4, "Neural Network Training")
    
    page = st.sidebar.radio(
        "Navigate",
        pages,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Model path")
    st.sidebar.markdown(f"<div class='mono'>{MODEL_FILE}</div>", unsafe_allow_html=True)

    if page == "Overview":
        _show_overview()
    elif page == "Data Explorer":
        _show_data_explorer()
    elif page == "Data Generation":
        _show_data_generation_tab()
    elif page == "Adaptive Training":
        _show_training_tab()
    elif page == "Neural Network Training":
        _show_nn_training_tab()
    elif page == "Comparison":
        _show_comparison_tab()
    elif page == "Predict":
        _show_predict_tab()


if __name__ == "__main__":
    main()
