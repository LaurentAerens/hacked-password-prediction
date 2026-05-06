"""
H.A.R.P. - Hacked Account Risk Predictor v2
Open-source ML framework with GridSearchCV and Optuna HPO
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_generation import get_data
from modern_trainers.optimizers.sklearn_hpo import train_with_gridsearch
from modern_trainers.registries.mlflow_registry import MLflowRegistry

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
    print("2. Train Phase 1 (GridSearchCV, ~30 min, best for MVP)")
    print("3. Train Phase 2 (Optuna HPO, ~1-2 hours, production)")
    print("4. Use Trained Models (make predictions)")
    print("5. View MLflow Results (metrics & experiments)")
    print("6. Exit")
    print("\n" + "="*60)

def phase1_training():
    """Phase 1: GridSearchCV training."""
    print("\n📊 Loading data...")
    try:
        X, y = get_data('data/combined_data.csv')
        print(f"✅ Loaded {len(X)} samples")
    except FileNotFoundError:
        print("❌ Data not found. Run option 1 to generate data first.")
        return
    
    print("\n🔍 Starting Phase 1 (GridSearchCV)...")
    print("   • 12 models × 8 preprocessors = 96 combinations")
    print("   • 5-fold stratified cross-validation")
    print("   • Parallel execution (uses all CPU cores)")
    print("\n⏱️  Estimated time: 20-40 minutes (CPU dependent)")
    
    try:
        result = train_with_gridsearch(
            X, y,
            models=None,  # Use all 12 models
            preprocessors=None,  # Use all 8 preprocessors
            cv_folds=5,
            n_jobs=-1,  # All cores
            random_state=42
        )
        
        print("\n✅ Phase 1 Complete!")
        print(f"   Best Model: {result['best_config']['model']} + {result['best_config']['preprocessor']}")
        print(f"   Best AUC: {result['best_config']['auc']:.4f}")
        print(f"   Combinations tested: {result['metrics']['combinations_tested']}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_csv = f"results/phase1_gridsearch_{result['experiment_name']}.csv"
        result['results_df'].to_csv(results_csv, index=False)
        print(f"   📁 Results saved: {results_csv}")
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/phase1_best_model.pkl'
        import joblib
        joblib.dump(result['best_model'], model_path)
        print(f"   📁 Model saved: {model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

def phase2_training():
    """Phase 2: Optuna Bayesian HPO (placeholder for now)."""
    print("\n🔬 Phase 2: Optuna Bayesian Hyperparameter Optimization")
    print("   ⏳ Implementation in progress (Wave 3)")
    print("   Estimated release: 1-2 weeks")
    print("\n   Phase 2 will provide:")
    print("   • Bayesian optimization (vs exhaustive GridSearch)")
    print("   • Adaptive trial pruning (skip bad trials early)")
    print("   • ~50% fewer trials needed for same quality")
    print("   • Full parallelization with joblib")

def data_generation():
    """Generate training data."""
    print("\n📥 Generating data...")
    try:
        subprocess.run(["python", "data_generation.py"], check=True)
        print("✅ Data generation complete")
    except subprocess.CalledProcessError:
        print("❌ Data generation failed")
    except FileNotFoundError:
        print("❌ data_generation.py not found")

def use_models():
    """Use trained models for predictions."""
    print("\n🔮 Model Prediction (uses Phase 1 best model)")
    
    model_path = 'models/phase1_best_model.pkl'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Run Phase 1 training first (option 2)")
        return
    
    import joblib
    model = joblib.load(model_path)
    
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
        subprocess.run(["mlflow", "ui", "--host", "127.0.0.1"], check=True)
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