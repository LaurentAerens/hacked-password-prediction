"""
H.A.R.P. - Hacked Account Risk Predictor v2
Open-source ML framework with GridSearchCV and Optuna HPO
"""

import os
import sys
import subprocess
from pathlib import Path

from .shared_lib.data_utils import get_data
from .adaptive_trainer import train_with_adaptive_search

ASCII_ART = r""" Welcome to H.A.R.P. - Hacked Account Risk Predictor v2
/ ',        ,--,
`;, '------'  @
 //'\-\-\-'\.'/
|| \ \ \ \,' /
||  \ \ \'  /
 \\  \,\   /
  \\,'    /
   `-----'          Credit Ascii: Tostig
   
🚀 v2: Open-source ML, no cloud credentials needed
"""

def show_menu():
    """Display main menu."""
    print(ASCII_ART)
    print("\n" + "="*60)
    print("PHASE 1: Modern ML Framework (GridSearchCV + 5-fold CV)")
    print("="*60)
    print("\nMenu:")
    print("1. Generate Data (if needed)")
    print("2. Train Phase 1 (Adaptive HPO, ~5-10 min, quick + quality)")
    print("3. Train Phase 2 (Optuna HPO, ~1-2 hours, production)")
    print("4. Use Trained Models (make predictions)")
    print("5. View MLflow Results (metrics & experiments)")
    print("6. Exit")
    print("\n" + "="*60)

def phase1_training():
    """Phase 1: GridSearchCV training."""
    print("\n📊 Loading data...")
    try:
        # Use absolute path relative to script location
        data_path = Path(__file__).parent.parent.parent / 'data' / 'combined_data.csv'
        df = get_data(str(data_path))
        X = df['password'].astype(str).tolist()
        y = df['target'].tolist()
        print(f"✅ Loaded {len(X)} samples")
    except FileNotFoundError:
        print("❌ Data not found. Run option 1 to generate data first.")
        return
    
    print("\n🔍 Starting Phase 1 (Adaptive Hyperparameter Optimization)...")
    print("   • Phase 1a: Quick screening (2-fold CV) of all 96 combinations")
    print("   • Phase 1b: Full 5-fold CV on top 20% performers (~20 models)")
    print("   • 70% time reduction vs exhaustive search")
    print("   • Parallel execution (uses all CPU cores)")
    print("\n⏱️  Estimated time: 5-15 minutes (CPU dependent)")
    
    try:
        result = train_with_adaptive_search(
            X, y,
            models=None,  # Use all 12 models
            preprocessors=None,  # Use all 8 preprocessors
            cv_folds=5,
            top_percent=0.20,  # Advance top 20% to full CV
            n_jobs=-1,  # All cores
            random_state=42
        )
        
        print("\n✅ Phase 1 Complete!")
        print(f"   Best Model: {result['best_config']['model']} + {result['best_config']['preprocessor']}")
        print(f"   Best AUC: {result['best_config']['auc']:.4f}")
        print(f"   Combinations screened: {result['metrics']['combinations_screened']}")
        print(f"   Combinations fully evaluated: {result['metrics']['combinations_fully_evaluated']}")
        print(f"   Time saved: ~{int(100 * (1 - result['metrics']['combinations_fully_evaluated'] / result['metrics']['combinations_screened']))}%")
        
        # Save results (use absolute path relative to script, 3 levels up to repo root)
        results_dir = Path(__file__).parent.parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        results_csv = results_dir / f"phase1_gridsearch_{result['experiment_name']}.csv"
        result['results_df'].to_csv(str(results_csv), index=False)
        print(f"   📁 Results saved: {results_csv}")
        
        # Save best model (use absolute path relative to script, 3 levels up to repo root)
        models_dir = Path(__file__).parent.parent.parent / 'models'
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / 'phase1_best_model.pkl'
        import joblib
        joblib.dump(result['best_model'], str(model_path))
        print(f"   📁 Model saved: {model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

def phase2_training():
    """Phase 2: Optuna Bayesian fine-tuning on top Phase 1 candidates."""
    print("\n🔬 Phase 2: Optuna Bayesian Hyperparameter Optimization")

    try:
        data_path = Path(__file__).parent.parent.parent / 'data' / 'combined_data.csv'
        df = get_data(str(data_path))
        X = df['password'].astype(str).tolist()
        y = df['target'].tolist()
        print(f"✅ Loaded {len(X)} samples")
    except FileNotFoundError:
        print("❌ Data not found. Run option 1 to generate data first.")
        return

    print("\n⚙️ Running Phase 1 + Phase 2 pipeline...")
    print("   • Phase 1: adaptive screening + full CV")
    print("   • Phase 2: Optuna TPE + Hyperband on top candidates")

    try:
        result = train_with_adaptive_search(
            X, y,
            models=None,
            preprocessors=None,
            cv_folds=5,
            top_percent=0.20,
            n_jobs=-1,
            random_state=42,
            run_phase2=True,
            phase2_top_n=3,
            phase2_trials_per_model=20,
        )

        results_dir = Path(__file__).parent.parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        models_dir = Path(__file__).parent.parent.parent / 'models'
        models_dir.mkdir(exist_ok=True)

        phase1_csv = results_dir / f"phase1_gridsearch_{result['experiment_name']}.csv"
        result['results_df'].to_csv(str(phase1_csv), index=False)

        import joblib
        phase1_model = models_dir / 'phase1_best_model.pkl'
        joblib.dump(result['best_model'], str(phase1_model))

        phase2_df = result.get('phase2_results_df')
        if phase2_df is not None and not phase2_df.empty:
            phase2_csv = results_dir / f"phase2_optuna_{result['experiment_name']}.csv"
            phase2_df.to_csv(str(phase2_csv), index=False)

            phase2_models_dir = models_dir / 'phase2'
            phase2_models_dir.mkdir(exist_ok=True)
            for key, model in (result.get('phase2_model_registry') or {}).items():
                safe_name = key.replace('__', '_')
                joblib.dump(model, str(phase2_models_dir / f"{safe_name}.pkl"))

            best_phase2 = phase2_df.iloc[0]
            print("\n✅ Phase 2 Complete!")
            print(
                f"   Best Phase 2: {best_phase2['model']} + {best_phase2['preprocessor']} "
                f"(AUC={best_phase2['phase2_best_auc']:.4f})"
            )
            print(f"   📁 Phase 2 results saved: {phase2_csv}")
            print(f"   📁 Phase 2 models saved: {phase2_models_dir}")
        else:
            print("\nℹ️ Phase 2 enabled, but no candidates completed.")

    except Exception as e:
        print(f"❌ Phase 2 training failed: {str(e)}")
        import traceback
        traceback.print_exc()

def data_generation():
    """Generate training data."""
    print("\n📥 Generating data...")
    script = Path(__file__).parent / "data_generation.py"
    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(Path(__file__).parent))
        print("✅ Data generation complete")
    except subprocess.CalledProcessError:
        print("❌ Data generation failed")
    except FileNotFoundError:
        print("❌ data_generation.py not found")

def use_models():
    """Use trained models for predictions."""
    print("\n🔮 Model Prediction (uses Phase 1 best model)")
    
    # Use absolute path relative to script location
    model_path = Path(__file__).parent / 'models' / 'phase1_best_model.pkl'
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Run Phase 1 training first (option 2)")
        return
    
    import joblib
    model = joblib.load(str(model_path))
    
    print("\n✅ Model loaded successfully")
    print("   Enter passwords to check (or 'quit' to exit):\n")
    
    while True:
        password = input("Enter password: ").strip()
        if password.lower() == 'quit':
            break
        
        try:
            # Vectorize input (matches Phase 1 preprocessing)
            from sklearn.feature_extraction.text import CountVectorizer
            # Note: In production, use the same vectorizer from pipeline
            pred = model.predict([password])[0]
            prob = model.predict_proba([password])[0]
            
            print(f"   Prediction: {'HIGH RISK' if pred == 1 else 'LOW RISK'}")
            print(f"   Confidence: {max(prob):.1%}")
            print()
        except Exception as e:
            print(f"   Error: {str(e)}\n")

def mlflow_ui():
    """Launch MLflow UI to view experiments."""
    print("\n📊 Launching MLflow UI...")
    print("   Open browser: http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    try:
        subprocess.run([sys.executable, "-m", "mlflow", "ui", "--host", "127.0.0.1"], check=True)
    except FileNotFoundError:
        print("❌ MLflow not installed. Run: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n✅ MLflow UI closed")

def main():
    """Main CLI loop."""
    while True:
        show_menu()
        choice = input("Enter your choice (1-6): ").strip()
        print("\n")
        
        if choice == '1':
            data_generation()
        elif choice == '2':
            phase1_training()
        elif choice == '3':
            phase2_training()
        elif choice == '4':
            use_models()
        elif choice == '5':
            mlflow_ui()
        elif choice == '6':
            print("👋 Exiting H.A.R.P.\n")
            break
        else:
            print("❌ Invalid choice. Please enter 1-6.\n")
        
        input("Press Enter to continue...")

if __name__ == "__main__":
    main()