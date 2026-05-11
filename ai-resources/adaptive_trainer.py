"""Adaptive hyperparameter optimization (Phase 1) - like Azure AutoML.

Phase 1a: Quick screening with 2-fold CV on all combinations
Phase 1b: Full 5-fold CV only on top 20% of candidates

This reduces training time by ~70% while maintaining model quality.
"""

import pandas as pd
import uuid
import time
import os
import json
import ctypes
import numpy as np
import optuna
from datetime import datetime
from typing import Dict, Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from modern_trainers.optimizers.sklearn_hpo import SklearnHPOTrainer
from shared_lib.telemetry_emitter import TelemetryEmitter
from shared_lib.control_signal import ControlSignal
from shared_lib.checkpoint_manager import CheckpointManager


def _detect_total_ram_gb() -> Optional[float]:
    """Return total system RAM in GiB, or None if detection fails."""
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) == 0:
                return None
            return float(status.ullTotalPhys) / (1024 ** 3)

        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        return float(page_size * phys_pages) / (1024 ** 3)
    except Exception:
        return None


def _resolve_parallel_plan(
    total_candidates: int,
    n_jobs: int,
    cv_folds: int,
    mean_trials: int,
    max_parallel_candidates: Optional[int],
    ram_per_candidate_gb: float,
    cpu_utilization_target: float,
    allow_candidate_parallelism: bool,
) -> Dict[str, int]:
    """Compute a resource-aware parallel plan for adaptive search.

    Returns:
        dict with keys `cpu_budget`, `outer_workers`, `inner_n_jobs`.
    """
    cpu_total = max(1, os.cpu_count() or 1)
    requested_budget = cpu_total if n_jobs in (-1, None) else max(1, min(cpu_total, int(n_jobs)))
    cpu_target = max(0.2, min(1.0, cpu_utilization_target))
    cpu_budget = max(1, min(cpu_total, int(requested_budget * cpu_target)))

    if total_candidates <= 1 or not allow_candidate_parallelism:
        return {"cpu_budget": cpu_budget, "outer_workers": 1, "inner_n_jobs": cpu_budget}

    total_ram_gb = _detect_total_ram_gb()
    if total_ram_gb is None or ram_per_candidate_gb <= 0:
        mem_limited_workers = total_candidates
    else:
        mem_limited_workers = max(1, int(total_ram_gb / ram_per_candidate_gb))

    cpu_limited_workers = cpu_budget
    if mean_trials <= 8 and cv_folds <= 2:
        cpu_limited_workers = max(1, min(cpu_budget, cpu_budget // 2 or 1))

    outer_cap = min(total_candidates, mem_limited_workers, cpu_limited_workers)
    if max_parallel_candidates is not None:
        outer_cap = min(outer_cap, max(1, int(max_parallel_candidates)))

    outer_workers = max(1, outer_cap)
    inner_n_jobs = max(1, cpu_budget // outer_workers)
    return {
        "cpu_budget": cpu_budget,
        "outer_workers": outer_workers,
        "inner_n_jobs": inner_n_jobs,
    }


def _suggest_finetune_param(trial: optuna.Trial, name: str, best_value, grid_values):
    """Suggest a Phase 2 parameter value around Phase 1 best config.

    Uses expanded local ranges for numeric params and categorical selection for others.
    """
    values = list(grid_values)
    if not values:
        return best_value

    if all(isinstance(v, int) for v in values):
        min_v, max_v = min(values), max(values)
        best = int(best_value) if best_value is not None else int(np.median(values))
        span = max(1, (max_v - min_v) // max(1, len(values) - 1))
        low = max(min_v, best - 2 * span)
        high = min(max(max_v, best + 2 * span), best + 4 * span)
        if low >= high:
            return best
        return trial.suggest_int(name, low, high)

    if all(isinstance(v, (float, int)) for v in values):
        min_v, max_v = float(min(values)), float(max(values))
        best = float(best_value) if best_value is not None else float(np.median(values))
        low = max(min_v / 3.0 if min_v > 0 else min_v, best / 3.0 if best > 0 else min_v)
        high = max(max_v * 1.5, best * 3.0 if best > 0 else max_v)
        if low <= 0 <= high:
            return trial.suggest_float(name, low, high)
        return trial.suggest_float(name, max(low, 1e-9), max(high, 1e-8), log=True)

    # categorical fallback
    return trial.suggest_categorical(name, values)


def _run_phase2_optuna_finetune(
    X,
    y,
    *,
    all_models: Dict,
    all_preprocessors: Dict,
    top_candidates: pd.DataFrame,
    cv_folds: int,
    random_state: int,
    run_id: str,
    emitter: TelemetryEmitter,
    control_signal: Optional[ControlSignal],
    phase2_top_n: int,
    phase2_trials_per_model: int,
) -> Dict:
    """Phase 2: Bayesian fine-tune top candidates with Optuna TPE + Hyperband pruning."""
    results = []
    model_registry: Dict[str, Pipeline] = {}

    top_n = min(max(1, phase2_top_n), len(top_candidates))
    selected = top_candidates.head(top_n)

    emitter.emit_event(
        "training.phase.started",
        "phase_2_optuna_finetune",
        "candidate",
        "started",
        current=0,
        total=top_n,
        message="Phase 2 Optuna fine-tuning started",
        metrics={"trials_per_model": phase2_trials_per_model, "cv_folds": cv_folds},
    )

    X_arr = np.array(X, dtype=object)
    y_arr = np.array(y)

    for candidate_idx, (_, row) in enumerate(selected.iterrows(), 1):
        if control_signal is not None and control_signal.should_stop():
            emitter.emit_event(
                "training.stopped",
                "phase_2_optuna_finetune",
                "candidate",
                "stopped",
                current=candidate_idx - 1,
                total=top_n,
                message="Phase 2 stopped by user",
            )
            break

        model_name = row["model"]
        prep_name = row["preprocessor"]
        base_grid = all_models[model_name]["params"]
        phase1_best_params = row.get("best_params", {}) or {}

        emitter.emit_event(
            "training.candidate.started",
            "phase_2_optuna_finetune",
            "candidate",
            "running",
            model=model_name,
            preprocessor=prep_name,
            current=candidate_idx,
            total=top_n,
            message=f"Phase 2 fine-tuning {model_name} + {prep_name}",
            metrics={"trials": phase2_trials_per_model, "phase1_auc": round(float(row.get("best_auc", row.get("screening_auc", 0))), 4)},
        )

        prep_step = all_preprocessors[prep_name]["step"]
        model_estimator = all_models[model_name]["estimator"]

        storage_path = f"sqlite:///{os.path.abspath('ai-resources/results/optuna_phase2.db')}"
        study_name = f"phase2_{run_id}_{candidate_idx}_{model_name}_{prep_name}"

        sampler = optuna.samplers.TPESampler(seed=random_state, multivariate=True, constant_liar=True)
        pruner = optuna.pruners.HyperbandPruner()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage_path,
            study_name=study_name,
            load_if_exists=True,
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, grid_values in base_grid.items():
                params[param_name] = _suggest_finetune_param(
                    trial,
                    param_name,
                    phase1_best_params.get(param_name),
                    grid_values,
                )

            pipeline = Pipeline([
                ("preprocessing", clone(prep_step)),
                ("model", clone(model_estimator)),
            ])
            pipeline.set_params(**params)

            fold_scores = []
            for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_arr, y_arr), 1):
                X_train, X_valid = X_arr[train_idx].tolist(), X_arr[valid_idx].tolist()
                y_train, y_valid = y_arr[train_idx], y_arr[valid_idx]

                pipeline.fit(X_train, y_train)
                if hasattr(pipeline, "predict_proba"):
                    y_score = pipeline.predict_proba(X_valid)[:, 1]
                else:
                    y_score = pipeline.decision_function(X_valid)

                fold_auc = float(roc_auc_score(y_valid, y_score))
                fold_scores.append(fold_auc)

                running_auc = float(np.mean(fold_scores))
                trial.report(running_auc, step=fold_idx)

                emitter.emit_event(
                    "training.trial.fold.completed",
                    "phase_2_optuna_finetune",
                    "fold",
                    "completed",
                    model=model_name,
                    preprocessor=prep_name,
                    current=fold_idx,
                    total=cv_folds,
                    metrics={
                        "trial": trial.number,
                        "fold_auc": round(fold_auc, 4),
                        "running_auc": round(running_auc, 4),
                    },
                )

                if trial.should_prune():
                    emitter.emit_event(
                        "training.trial.pruned",
                        "phase_2_optuna_finetune",
                        "trial",
                        "pruned",
                        model=model_name,
                        preprocessor=prep_name,
                        metrics={"trial": trial.number, "running_auc": round(running_auc, 4)},
                    )
                    raise optuna.TrialPruned()

            return float(np.mean(fold_scores))

        study.optimize(objective, n_trials=phase2_trials_per_model, n_jobs=1, show_progress_bar=False)

        try:
            best_trial = study.best_trial
            best_params = best_trial.params
            best_auc = float(study.best_value)
        except ValueError:
            best_trial = None
            best_params = phase1_best_params
            best_auc = float(row.get("best_auc", row.get("screening_auc", -1.0)))

        best_pipeline = Pipeline([
            ("preprocessing", clone(prep_step)),
            ("model", clone(model_estimator)),
        ])
        if best_params:
            best_pipeline.set_params(**best_params)
        best_pipeline.fit(X, y)

        registry_key = f"{model_name}__{prep_name}__phase2"
        model_registry[registry_key] = best_pipeline

        results.append({
            "model": model_name,
            "preprocessor": prep_name,
            "phase": "phase_2",
            "phase1_best_auc": float(row.get("best_auc", row.get("screening_auc", -1))),
            "phase2_best_auc": best_auc,
            "best_params": best_params,
            "optuna_trials": len(study.trials),
            "optuna_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
            "optuna_completed": sum(1 for t in study.trials if t.state.name == "COMPLETE"),
            "study_name": study_name,
        })

        emitter.emit_event(
            "training.candidate.completed",
            "phase_2_optuna_finetune",
            "candidate",
            "completed",
            model=model_name,
            preprocessor=prep_name,
            current=candidate_idx,
            total=top_n,
            metrics={
                "phase2_auc": round(best_auc, 4),
                "trials": len(study.trials),
            },
        )

    phase2_df = pd.DataFrame(results).sort_values("phase2_best_auc", ascending=False) if results else pd.DataFrame()
    emitter.emit_event(
        "training.phase.completed",
        "phase_2_optuna_finetune",
        "candidate",
        "completed",
        current=len(results),
        total=top_n,
        metrics={"phase2_candidates": len(results)},
    )

    return {
        "phase2_results_df": phase2_df,
        "phase2_model_registry": model_registry,
    }


def train_with_adaptive_search(X, y, models: Optional[List[str]] = None,
                               preprocessors: Optional[List[str]] = None,
                               cv_folds: int = 5, top_percent: float = 0.20,
                               n_jobs: int = -1, random_state: int = 42,
                               progress_callback: Optional[Callable] = None,
                               run_id: Optional[str] = None,
                               control_signal: Optional[ControlSignal] = None,
                               checkpoint_manager: Optional[CheckpointManager] = None,
                               run_phase2: bool = False,
                               phase2_top_n: int = 3,
                               phase2_trials_per_model: int = 20,
                               max_parallel_candidates: Optional[int] = None,
                               ram_per_candidate_gb: float = 2.0,
                               cpu_utilization_target: float = 0.9) -> Dict:
    """
    Adaptive hyperparameter optimization (like Azure AutoML).
    
    Phase 1a: Quick screening (2-fold CV) on all combinations
    Phase 1b: Full CV only on top 20% of candidates
    
    This dramatically reduces training time (60-80% faster) while maintaining quality.
    
    Args:
        X: Feature matrix
        y: Target labels
        models: List of model names (None = all)
        preprocessors: List of preprocessor names (None = all)
        cv_folds: Number of CV folds for top candidates (default: 5)
        top_percent: Fraction of candidates to advance (default: 0.20)
        n_jobs: Number of parallel jobs
        random_state: Random seed
        progress_callback: Optional callback function to receive progress events (ProgressEvent dict)
        run_id: Optional run identifier for correlating events. Generated if not provided.
        control_signal: Optional ControlSignal for pause/stop/resume. If None, trainer runs without control.
        checkpoint_manager: Optional CheckpointManager for saving/resuming from checkpoints. If None, no checkpoints saved.
        run_phase2: Whether to run Optuna Phase 2 fine-tuning on top candidates.
        phase2_top_n: Number of top Phase 1 candidates to fine-tune in Phase 2.
        phase2_trials_per_model: Number of Optuna trials per candidate in Phase 2.
        max_parallel_candidates: Optional cap for concurrent model+preprocessor jobs.
        ram_per_candidate_gb: Estimated RAM budget per concurrent candidate.
        cpu_utilization_target: Fraction of requested CPUs to actively use (0.2-1.0).
        
    Returns:
        Dictionary with best_model, results_df, experiment_name, metrics
    """
    trainer = SklearnHPOTrainer(random_state=random_state, n_jobs=n_jobs)
    all_models = trainer._get_models()
    all_preprocessors = trainer._get_preprocessors()

    # Default ordering favors faster models first for quicker interactive feedback.
    model_speed_priority = {
        'LogisticRegression': 1,
        'MultinomialNB': 2,
        'LinearSVM': 3,
        'KNeighbors': 4,
        'DecisionTree': 5,
        'SVM': 6,
        'AdaBoost': 7,
        'Bagging': 8,
        'ExtraTrees': 9,
        'RandomForest': 10,
        'GradientBoosting': 11,
        'XGBoost': 12,
    }

    if models is None:
        models = list(all_models.keys())
        models = sorted(models, key=lambda name: model_speed_priority.get(name, 999))

    preprocessors = preprocessors or list(all_preprocessors.keys())

    # Password datasets are text by default, so keep only text-compatible preprocessors.
    is_text_input = len(X) > 0 and isinstance(X[0], str)
    if is_text_input:
        preprocessors = [p for p in preprocessors if all_preprocessors[p].get('requires_text', False)]
    
    # Initialize telemetry emitter
    if run_id is None:
        run_id = str(uuid.uuid4())
    emitter = TelemetryEmitter(run_id=run_id)
    if progress_callback is not None:
        emitter.subscribe(progress_callback)
    
    experiment_name = f"sklearn_hpo_adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    total_combinations = len(models) * len(preprocessors)
    top_count = max(1, int(total_combinations * top_percent))
    
    # Emit run started event
    emitter.emit_event(
        "training.run.started",
        "init",
        "run",
        "started",
        total=total_combinations,
        cv_folds=2
    )
    
    print(f"\n{'='*70}")
    print(f"PHASE 1a: Quick Screening (2-fold CV)")
    print(f"{'='*70}")
    print(f"Total combinations: {total_combinations}")
    print(f"Will advance top {top_count} ({top_percent*100:.0f}%) to full CV\n")
    
    # Emit phase 1a started
    emitter.emit_event(
        "training.phase.started",
        "phase_1a_screening",
        "combination",
        "started",
        current=0,
        total=total_combinations,
        cv_folds=2,
        top_percent=top_percent
    )
    
    screening_results = []
    combination_count = 0

    phase1_candidates = []
    total_trials = 0
    for prep_name in preprocessors:
        incompatible_models = all_preprocessors[prep_name].get('incompatible_models', [])
        for model_name in models:
            if model_name in incompatible_models:
                continue  # Skip incompatible model+preprocessor combinations
            trials = len(list(ParameterGrid(all_models[model_name]['params'])))
            total_trials += trials
            phase1_candidates.append((prep_name, model_name, trials))

    mean_trials = max(1, total_trials // max(1, len(phase1_candidates)))
    parallel_plan = _resolve_parallel_plan(
        total_candidates=len(phase1_candidates),
        n_jobs=n_jobs,
        cv_folds=2,
        mean_trials=mean_trials,
        max_parallel_candidates=max_parallel_candidates,
        ram_per_candidate_gb=ram_per_candidate_gb,
        cpu_utilization_target=cpu_utilization_target,
        allow_candidate_parallelism=(control_signal is None),
    )

    emitter.emit_event(
        "training.parallel.plan",
        "phase_1a_screening",
        "scheduler",
        "planned",
        metrics={
            "cpu_budget": parallel_plan["cpu_budget"],
            "outer_workers": parallel_plan["outer_workers"],
            "inner_n_jobs": parallel_plan["inner_n_jobs"],
            "mean_trials": mean_trials,
            "ram_per_candidate_gb": ram_per_candidate_gb,
            "cpu_utilization_target": cpu_utilization_target,
        },
    )

    print(
        f"Parallel plan: workers={parallel_plan['outer_workers']} | "
        f"grid_n_jobs={parallel_plan['inner_n_jobs']} | cpu_budget={parallel_plan['cpu_budget']}"
    )
    
    def _run_phase1_candidate(prep_name: str, model_name: str) -> Dict:
        preprocessor = all_preprocessors[prep_name]
        prep_step = preprocessor['step']
        model_config = all_models[model_name]
        model_estimator = model_config['estimator']
        param_grid = model_config['params']

        pipeline = Pipeline([
            ('preprocessing', clone(prep_step)),
            ('model', clone(model_estimator))
        ])

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=parallel_plan['inner_n_jobs'],
            verbose=0,
            pre_dispatch='2*n_jobs'
        )

        grid_search.fit(X, y)
        return {
            'model': model_name,
            'preprocessor': prep_name,
            'screening_auc': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }

    for prep_name, model_name, n_trials in phase1_candidates:
        combination_count += 1
        emitter.emit_event(
            "training.candidate.started",
            "phase_1a_screening",
            "combination",
            "running",
            model=model_name,
            preprocessor=prep_name,
            current=combination_count,
            total=total_combinations,
            message=f"Fine-tuning {model_name} + {prep_name}",
            metrics={"cv_folds": 2, "trials": n_trials}
        )

    completed_count = 0
    if parallel_plan['outer_workers'] == 1:
        for prep_name, model_name, _ in phase1_candidates:
            try:
                candidate_result = _run_phase1_candidate(prep_name, model_name)
                screening_results.append(candidate_result)
                completed_count += 1
                auc_score = candidate_result['screening_auc']
                print(f"  {model_name:20} + {prep_name:15} -> AUC: {auc_score:.4f}")
                emitter.emit_event(
                    "training.candidate.completed",
                    "phase_1a_screening",
                    "combination",
                    "completed",
                    model=model_name,
                    preprocessor=prep_name,
                    current=completed_count,
                    total=total_combinations,
                    metrics={"auc": round(auc_score, 4)}
                )
            except Exception as e:
                error_msg = str(e)[:50]
                print(f"  {model_name:20} + {prep_name:15} -> FAILED: {error_msg}")
                emitter.emit_event(
                    "training.candidate.failed",
                    "phase_1a_screening",
                    "combination",
                    "failed",
                    model=model_name,
                    preprocessor=prep_name,
                    current=completed_count,
                    total=total_combinations,
                    error_type=type(e).__name__,
                    error_message=error_msg
                )

            if checkpoint_manager is not None:
                checkpoint_manager.save_checkpoint(
                    run_id=run_id,
                    phase="phase_1a",
                    combination_index=completed_count,
                    total_combinations=total_combinations,
                    results_df=pd.DataFrame(screening_results),
                    best_model=None,
                    best_auc=-1,
                    best_config={}
                )

            if control_signal is not None:
                if control_signal.should_pause():
                    emitter.emit_event("training.paused", "phase_1a_screening", "combination", "paused")
                    while control_signal.get_state() == "PAUSED":
                        time.sleep(0.1)
                    emitter.emit_event("training.resumed", "phase_1a_screening", "combination", "resumed")

                if control_signal.should_stop():
                    emitter.emit_event("training.stopped", "phase_1a_screening", "combination", "stopped")
                    cv_results = pd.DataFrame(screening_results)
                    return {
                        'best_model': None,
                        'best_config': None,
                        'results_df': cv_results,
                        'experiment_name': experiment_name,
                        'metrics': {
                            'best_auc': -1,
                            'combinations_screened': len(screening_results),
                            'combinations_fully_evaluated': 0,
                            'cv_folds': cv_folds,
                            'screening_folds': 2,
                            'top_percent': top_percent
                        }
                    }
    else:
        with ThreadPoolExecutor(max_workers=parallel_plan['outer_workers']) as executor:
            futures = {
                executor.submit(_run_phase1_candidate, prep_name, model_name): (prep_name, model_name)
                for prep_name, model_name, _ in phase1_candidates
            }

            for future in as_completed(futures):
                prep_name, model_name = futures[future]
                try:
                    candidate_result = future.result()
                    screening_results.append(candidate_result)
                    completed_count += 1
                    auc_score = candidate_result['screening_auc']
                    print(f"  {model_name:20} + {prep_name:15} -> AUC: {auc_score:.4f}")
                    emitter.emit_event(
                        "training.candidate.completed",
                        "phase_1a_screening",
                        "combination",
                        "completed",
                        model=model_name,
                        preprocessor=prep_name,
                        current=completed_count,
                        total=total_combinations,
                        metrics={"auc": round(auc_score, 4)}
                    )
                except Exception as e:
                    completed_count += 1
                    error_msg = str(e)[:50]
                    print(f"  {model_name:20} + {prep_name:15} -> FAILED: {error_msg}")
                    emitter.emit_event(
                        "training.candidate.failed",
                        "phase_1a_screening",
                        "combination",
                        "failed",
                        model=model_name,
                        preprocessor=prep_name,
                        current=completed_count,
                        total=total_combinations,
                        error_type=type(e).__name__,
                        error_message=error_msg
                    )

                if checkpoint_manager is not None:
                    checkpoint_manager.save_checkpoint(
                        run_id=run_id,
                        phase="phase_1a",
                        combination_index=completed_count,
                        total_combinations=total_combinations,
                        results_df=pd.DataFrame(screening_results),
                        best_model=None,
                        best_auc=-1,
                        best_config={}
                    )
    
    # Rank by screening AUC and select top candidates
    screening_df = pd.DataFrame(screening_results).sort_values('screening_auc', ascending=False)
    top_candidates = screening_df.head(top_count)
    
    # Emit phase 1a completed
    emitter.emit_event(
        "training.phase.completed",
        "phase_1a_screening",
        "combination",
        "completed",
        current=total_combinations,
        total=total_combinations,
        metrics={"combinations_screened": len(screening_results)}
    )
    
    print(f"\n{'='*70}")
    print(f"PHASE 1b: Top {top_count} Candidates (full {cv_folds}-fold CV)")
    print(f"{'='*70}\n")
    
    # Emit phase 1b started
    emitter.emit_event(
        "training.phase.started",
        "phase_1b_full_cv",
        "candidate",
        "started",
        current=0,
        total=top_count,
        cv_folds=cv_folds,
        top_percent=top_percent
    )
    
    # Phase 1b: Full CV on top candidates only
    final_results = []
    best_auc = -1
    best_pipeline = None
    best_config = None
    
    for candidate_idx, (idx, row) in enumerate(top_candidates.iterrows(), 1):
        model_name = row['model']
        prep_name = row['preprocessor']
        model_config = all_models[model_name]
        n_trials = len(list(ParameterGrid(model_config['params'])))

        # Emit "currently tuning" event before full CV fit starts.
        emitter.emit_event(
            "training.candidate.started",
            "phase_1b_full_cv",
            "candidate",
            "running",
            model=model_name,
            preprocessor=prep_name,
            current=candidate_idx,
            total=top_count,
            message=f"Fine-tuning {model_name} + {prep_name}",
            metrics={"cv_folds": cv_folds, "trials": n_trials, "screening_auc": round(float(row['screening_auc']), 4)}
        )
        
        try:
            preprocessor = all_preprocessors[prep_name]
            prep_step = preprocessor['step']
            model_estimator = model_config['estimator']
            param_grid = model_config['params']
            
            pipeline = Pipeline([
                ('preprocessing', prep_step),
                ('model', model_estimator)
            ])
            
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=n_jobs,
                verbose=0
            )
            
            grid_search.fit(X, y)
            
            best_idx = grid_search.best_index_
            cv_row = grid_search.cv_results_
            
            auc_score = grid_search.best_score_
            result = {
                'model': model_name,
                'preprocessor': prep_name,
                'screening_auc': row['screening_auc'],
                'best_params': grid_search.best_params_,
                'best_auc': auc_score,
                'mean_auc': cv_row['mean_test_score'][best_idx],
                'std_auc': cv_row['std_test_score'][best_idx],
                'fit_time': cv_row['mean_fit_time'][best_idx]
            }
            
            final_results.append(result)
            
            if auc_score > best_auc:
                best_auc = auc_score
                best_pipeline = grid_search.best_estimator_
                best_config = {
                    'model': model_name,
                    'preprocessor': prep_name,
                    'params': grid_search.best_params_,
                    'auc': auc_score
                }
            
            print(f"  {model_name:20} + {prep_name:15} -> AUC: {auc_score:.4f}")
            
            # Emit per-candidate progress with best_auc update
            emitter.emit_event(
                "training.candidate.completed",
                "phase_1b_full_cv",
                "candidate",
                "completed",
                model=model_name,
                preprocessor=prep_name,
                current=candidate_idx,
                total=top_count,
                metrics={"auc": round(auc_score, 4), "fit_time": round(result['fit_time'], 2)}
            )
            
            # Save checkpoint after combination completes
            if checkpoint_manager is not None:
                checkpoint_manager.save_checkpoint(
                    run_id=run_id,
                    phase="phase_1b",
                    combination_index=candidate_idx,
                    total_combinations=top_count,
                    results_df=pd.DataFrame(final_results),
                    best_model=best_pipeline,
                    best_auc=best_auc,
                    best_config=best_config or {}
                )
            
            # Check control signal
            if control_signal is not None:
                # Check for pause
                if control_signal.should_pause():
                    emitter.emit_event(
                        "training.paused",
                        "phase_1b_full_cv",
                        "candidate",
                        "paused"
                    )
                    # Wait until resume is called
                    while control_signal.get_state() == "PAUSED":
                        time.sleep(0.1)
                    # Resumed
                    emitter.emit_event(
                        "training.resumed",
                        "phase_1b_full_cv",
                        "candidate",
                        "resumed"
                    )
                
                # Check for stop
                if control_signal.should_stop():
                    emitter.emit_event(
                        "training.stopped",
                        "phase_1b_full_cv",
                        "candidate",
                        "stopped"
                    )
                    # Return partial results
                    cv_results = pd.DataFrame(final_results)
                    return {
                        'best_model': best_pipeline,
                        'best_config': best_config,
                        'results_df': cv_results,
                        'experiment_name': experiment_name,
                        'metrics': {
                            'best_auc': float(best_auc) if best_auc != -1 else -1,
                            'combinations_screened': len(screening_results) if 'screening_results' in locals() else 0,
                            'combinations_fully_evaluated': len(final_results),
                            'cv_folds': cv_folds,
                            'screening_folds': 2,
                            'top_percent': top_percent
                        }
                    }
            
        except Exception as e:
            error_msg = str(e)[:50]
            print(f"  {model_name:20} + {prep_name:15} -> FAILED: {error_msg}")
            
            # Emit failure event
            emitter.emit_event(
                "training.candidate.failed",
                "phase_1b_full_cv",
                "candidate",
                "failed",
                model=model_name,
                preprocessor=prep_name,
                current=candidate_idx,
                total=top_count,
                error_type=type(e).__name__,
                error_message=error_msg
            )
            continue
    
    cv_results = pd.DataFrame(final_results).sort_values('best_auc', ascending=False).reset_index(drop=True)
    
    time_saved = int(100 * (1 - len(final_results) / max(1, len(screening_results))))
    
    # Emit phase 1b completed
    emitter.emit_event(
        "training.phase.completed",
        "phase_1b_full_cv",
        "candidate",
        "completed",
        current=len(final_results),
        total=top_count,
        metrics={"combinations_fully_evaluated": len(final_results)}
    )
    
    print(f"\n{'='*70}")
    print(f"Phase 1 Complete (Adaptive)")
    print(f"{'='*70}")
    print(f"Best Model: {best_config['model']} + {best_config['preprocessor']}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Combinations Screened: {len(screening_results)}")
    print(f"Combinations Fully Evaluated: {len(final_results)}")
    print(f"Time Saved: ~{time_saved}%")

    phase2_results_df = pd.DataFrame()
    phase2_model_registry = {}
    if run_phase2 and not cv_results.empty:
        print(f"\n{'='*70}")
        print("PHASE 2: Optuna Bayesian Fine-Tuning")
        print(f"{'='*70}")
        phase2_payload = _run_phase2_optuna_finetune(
            X,
            y,
            all_models=all_models,
            all_preprocessors=all_preprocessors,
            top_candidates=cv_results,
            cv_folds=cv_folds,
            random_state=random_state,
            run_id=run_id,
            emitter=emitter,
            control_signal=control_signal,
            phase2_top_n=phase2_top_n,
            phase2_trials_per_model=phase2_trials_per_model,
        )
        phase2_results_df = phase2_payload.get("phase2_results_df", pd.DataFrame())
        phase2_model_registry = phase2_payload.get("phase2_model_registry", {})

        if not phase2_results_df.empty:
            top_phase2 = phase2_results_df.iloc[0]
            print(
                f"Phase 2 Best: {top_phase2['model']} + {top_phase2['preprocessor']} "
                f"-> AUC: {top_phase2['phase2_best_auc']:.4f}"
            )
    
    # Emit run completed
    emitter.emit_event(
        "training.run.completed",
        "completion",
        "run",
        "completed",
        metrics={
            "best_auc": round(float(best_auc), 4),
            "combinations_screened": len(screening_results),
            "combinations_fully_evaluated": len(final_results),
            "cv_folds": cv_folds,
            "screening_folds": 2,
            "top_percent": top_percent,
            "time_saved_percent": time_saved,
            "phase2_enabled": run_phase2,
            "phase2_candidates": int(len(phase2_results_df)) if phase2_results_df is not None else 0
        }
    )
    
    return {
        'best_model': best_pipeline,
        'best_config': best_config,
        'results_df': cv_results,
        'phase2_results_df': phase2_results_df,
        'phase2_model_registry': phase2_model_registry,
        'experiment_name': experiment_name,
        'metrics': {
            'best_auc': float(best_auc),
            'combinations_screened': len(screening_results),
            'combinations_fully_evaluated': len(final_results),
            'cv_folds': cv_folds,
            'screening_folds': 2,
            'top_percent': top_percent,
            'phase2_enabled': run_phase2,
            'phase2_candidates': int(len(phase2_results_df)) if phase2_results_df is not None else 0
        }
    }
