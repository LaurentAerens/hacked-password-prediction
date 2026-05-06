"""
MLflow-based experiment tracking and model registry.

Provides centralized storage for trained models, hyperparameters, and metrics
using local file store (no cloud backend required).
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional, Any

import mlflow
import pandas as pd
import joblib


class MLflowRegistry:
    """MLflow-based model registry with local storage."""
    
    def __init__(self, experiment_name: str = "default", tracking_uri: str = "./mlruns"):
        """
        Initialize MLflow registry.
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: Path to local MLflow storage (default: ./mlruns)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Configure MLflow
        mlflow.set_tracking_uri(f"file:///{tracking_uri}")
        mlflow.set_experiment(experiment_name)
        
        # Create experiment if doesn't exist
        try:
            mlflow.create_experiment(experiment_name)
        except:
            pass  # Experiment already exists
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)
    
    def end_run(self):
        """End current MLflow run."""
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except:
                # MLflow has param size/count limits; skip if too large
                pass
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics (AUC, precision, recall, etc.)."""
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
    
    def log_model(self, model, model_name: str = "model"):
        """Log trained model."""
        model_path = f"models/{model_name}"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
    
    def log_artifact(self, file_path: str):
        """Log artifact (file, plot, report, etc.)."""
        mlflow.log_artifact(file_path)
    
    def log_dataframe_as_csv(self, df: pd.DataFrame, filename: str):
        """Log DataFrame as CSV artifact."""
        df.to_csv(filename, index=False)
        mlflow.log_artifact(filename)
        os.remove(filename)
    
    def get_run_id(self) -> str:
        """Get current run ID."""
        return mlflow.active_run().info.run_id if mlflow.active_run() else None
    
    def search_experiments(self, metric_name: str = "auc", top_k: int = 5) -> pd.DataFrame:
        """Search for best runs by metric."""
        experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=top_k
        )
        return runs
    
    def get_best_model(self, metric_name: str = "auc") -> Optional[Dict]:
        """Retrieve best model metadata."""
        runs = self.search_experiments(metric_name=metric_name, top_k=1)
        if len(runs) > 0:
            best_run = runs.iloc[0]
            return {
                'run_id': best_run['run_id'],
                'metric_value': best_run[f'metrics.{metric_name}'],
                'params': {k: v for k, v in best_run.items() if k.startswith('params.')},
                'timestamp': best_run['start_time']
            }
        return None
    
    def log_gridseaarch_results(self, results_df: pd.DataFrame, experiment_name: str):
        """Log GridSearchCV results from p1-trainer-refactor."""
        # Flatten best config
        if len(results_df) > 0:
            best_row = results_df.iloc[0]
            
            # Log top-level metrics
            mlflow.log_metric('best_auc', float(best_row['best_auc']))
            mlflow.log_metric('best_model_index', 0)
            mlflow.log_param('best_model', best_row['model'])
            mlflow.log_param('best_preprocessor', best_row['preprocessor'])
            
            # Log results CSV
            csv_path = f"results/{experiment_name}_gridsearch.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            results_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)


def create_registry(experiment_name: str = "hacked_password_v2") -> MLflowRegistry:
    """Create and return an MLflow registry instance."""
    return MLflowRegistry(experiment_name=experiment_name)


if __name__ == '__main__':
    print("mlflow_registry.py loaded successfully")
