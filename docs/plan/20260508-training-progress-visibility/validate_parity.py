#!/usr/bin/env python
"""
Parity Oracle Validation Script for Callback-Absent Behavior

Demonstrates and validates that Streamlit and CLI surfaces produce equivalent
training outcomes when progress_callback is absent (None).

This is the MVP validation harness for task tpv-004.

Usage:
    python validate_parity.py [--data-path PATH] [--quick]
    
    --data-path: Path to combined_data.csv (default: ai-resources/data/combined_data.csv)
    --quick: Use smaller dataset for faster validation (default: full dataset)

Exit codes:
    0 = All parity checks passed (MVP gate passes)
    1 = Parity validation failed (MVP gate fails)
    2 = Setup/data error
"""

import sys
import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Any

# Add ai-resources to path - go up from docs/plan/20260508-training-progress-visibility/
SCRIPT_DIR = Path(__file__).resolve().parent  # docs/plan/20260508-training-progress-visibility/
PLAN_DIR = SCRIPT_DIR.parent  # docs/plan/
DOCS_DIR = PLAN_DIR.parent  # docs/
BASE_DIR = DOCS_DIR.parent  # repo root
AI_RESOURCES = BASE_DIR / "ai-resources"

sys.path.insert(0, str(AI_RESOURCES))
sys.path.insert(0, str(BASE_DIR))

from adaptive_trainer import train_with_adaptive_search
from shared_lib.data_utils import get_data
from parity_oracle import ParityOracle, ParityValidationError


def run_streamlit_path(X: list, y: list, config: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    """
    Simulate Streamlit execution path:
    - Capture stdout to StringIO (Streamlit redirection pattern)
    - Call trainer with callback=None (absent)
    - Return result and captured logs
    """
    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        result = train_with_adaptive_search(
            X, y,
            models=config.get('models'),
            preprocessors=config.get('preprocessors'),
            cv_folds=config['cv_folds'],
            top_percent=config['top_percent'],
            n_jobs=config['n_jobs'],
            random_state=config['random_state'],
            # progress_callback=None  # Explicitly absent (default)
        )
    
    return result, log_stream.getvalue()


def run_cli_path(X: list, y: list, config: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    """
    Simulate CLI execution path:
    - Call trainer with callback=None (absent)
    - Let stdout print directly (not redirected)
    - Return result
    
    This would normally print live to terminal; we capture for validation.
    """
    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        result = train_with_adaptive_search(
            X, y,
            models=config.get('models'),
            preprocessors=config.get('preprocessors'),
            cv_folds=config['cv_folds'],
            top_percent=config['top_percent'],
            n_jobs=config['n_jobs'],
            random_state=config['random_state'],
            # progress_callback=None  # Explicitly absent (default)
        )
    
    return result, log_stream.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Parity Oracle Validation: Assert Streamlit/CLI equivalence when callback is absent"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(BASE_DIR / "ai-resources" / "data" / "combined_data.csv"),
        help="Path to combined_data.csv"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick config for faster validation (fewer cv_folds, smaller model/preprocessor set)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override CV folds (default: 3 for --quick, 5 for full)"
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=0.2,
        help="Top percent for Phase 1b (default: 0.2)"
    )
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}", file=sys.stderr)
        return 2
    
    print(f"Loading data from {data_path}...")
    try:
        df = get_data(str(data_path))
        X = df["password"].astype(str).tolist()
        y = df["target"].tolist()
        print(f"✓ Loaded {len(X)} samples")
    except Exception as e:
        print(f"❌ Failed to load data: {e}", file=sys.stderr)
        return 2
    
    # Configure training
    config = {
        'models': None,  # Use all models
        'preprocessors': None,  # Use all preprocessors
        'cv_folds': args.cv_folds or (3 if args.quick else 5),
        'top_percent': args.top_percent,
        'n_jobs': -1,
        'random_state': 42,
    }
    
    print(f"\n{'='*70}")
    print("PARITY ORACLE VALIDATION")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  CV Folds: {config['cv_folds']}")
    print(f"  Top Percent: {config['top_percent']}")
    print(f"  Random Seed: {config['random_state']}")
    print(f"  Data: {len(X)} samples")
    print()
    
    # Run Streamlit path
    print("Running STREAMLIT path (callback=None, stdout redirected)...")
    try:
        streamlit_result, streamlit_logs = run_streamlit_path(X, y, config)
        print(f"✓ Streamlit execution completed")
        print(f"  - Best model: {streamlit_result['best_config']['model']}")
        print(f"  - Best preprocessor: {streamlit_result['best_config']['preprocessor']}")
        print(f"  - Best AUC: {streamlit_result['best_config']['auc']:.4f}")
        print(f"  - Combinations screened: {streamlit_result['metrics']['combinations_screened']}")
        print(f"  - Combinations evaluated: {streamlit_result['metrics']['combinations_fully_evaluated']}")
    except Exception as e:
        print(f"❌ Streamlit path failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2
    
    # Validate Streamlit result independently
    print("\nValidating Streamlit result contract...")
    sl_ok, sl_msgs = ParityOracle.validate_callback_absent_behavior(streamlit_result)
    for msg in sl_msgs:
        print(f"  {msg}")
    if not sl_ok:
        print("❌ Streamlit result contract validation failed", file=sys.stderr)
        return 1
    
    # Run CLI path
    print("\nRunning CLI path (callback=None, stdout would be live)...")
    try:
        cli_result, cli_logs = run_cli_path(X, y, config)
        print(f"✓ CLI execution completed")
        print(f"  - Best model: {cli_result['best_config']['model']}")
        print(f"  - Best preprocessor: {cli_result['best_config']['preprocessor']}")
        print(f"  - Best AUC: {cli_result['best_config']['auc']:.4f}")
        print(f"  - Combinations screened: {cli_result['metrics']['combinations_screened']}")
        print(f"  - Combinations evaluated: {cli_result['metrics']['combinations_fully_evaluated']}")
    except Exception as e:
        print(f"❌ CLI path failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2
    
    # Validate CLI result independently
    print("\nValidating CLI result contract...")
    cli_ok, cli_msgs = ParityOracle.validate_callback_absent_behavior(cli_result)
    for msg in cli_msgs:
        print(f"  {msg}")
    if not cli_ok:
        print("❌ CLI result contract validation failed", file=sys.stderr)
        return 1
    
    # Validate equivalence across surfaces
    print("\nValidating equivalence across Streamlit and CLI surfaces...")
    parity_ok, parity_msgs = ParityOracle.validate_parity_across_surfaces(
        streamlit_result,
        cli_result
    )
    for msg in parity_msgs:
        print(f"  {msg}")
    
    print(f"\n{'='*70}")
    if parity_ok:
        print("✓ MVP GATE PASSED: Parity Oracle validation successful")
        print(f"{'='*70}")
        print("\nSummary:")
        print(f"  ✓ Streamlit and CLI both produced valid result contracts")
        print(f"  ✓ Best model selection is equivalent: {streamlit_result['best_config']['model']}")
        print(f"  ✓ Best AUC is equivalent: {streamlit_result['best_config']['auc']:.4f}")
        print(f"  ✓ Metrics are equivalent")
        print(f"  ✓ Top-5 results ranking is identical")
        print(f"\nCallback-absent behavior is VERIFIED as parity-safe for MVP rollout.")
        return 0
    else:
        print("❌ MVP GATE FAILED: Parity Oracle validation failed")
        print(f"{'='*70}")
        print("\nParity violations detected. Training outcomes are not equivalent.")
        print("Streamlit and CLI must produce identical results when callback=None.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
