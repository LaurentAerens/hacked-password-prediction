"""ModelComparison: Compare metrics across Phase 1, Phase 2, NN models."""

import pandas as pd
import numpy as np
from typing import Optional
from .model_registry import UnifiedModelRegistry


class ModelComparison:
    """
    Compare metrics across Phase 1, Phase 2, NN models.
    """
    
    def __init__(self, registry: UnifiedModelRegistry):
        self.registry = registry
        self.models = registry.get_latest_models()
        self.comparison_df = None
    
    def build_comparison_table(self) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        [Phase, Model, Accuracy, Precision, Recall, F1, Params, Training_Time]
        """
        rows = []
        
        for phase, model_info in self.models.items():
            metadata = self.registry.get_model_metadata(phase, model_info["run_id"])
            
            rows.append({
                "Phase": phase.upper(),
                "Run ID": model_info["run_id"],
                "Path": model_info["path"],
                "Accuracy": metadata.get("metrics", {}).get("val_acc", 0),
                "Precision": metadata.get("metrics", {}).get("precision", 0),
                "Recall": metadata.get("metrics", {}).get("recall", 0),
                "F1": metadata.get("metrics", {}).get("f1", 0),
                "Params": metadata.get("architecture", {}).get("param_count", 0),
                "Training Time": metadata.get("training_time_sec", 0),
            })
        
        self.comparison_df = pd.DataFrame(rows)
        return self.comparison_df
    
    def get_best_model_by_metric(self, metric: str = "accuracy") -> Optional[str]:
        """Return phase name of best model by metric."""
        if self.comparison_df is None:
            self.build_comparison_table()
        
        if self.comparison_df.empty:
            return None
        
        metric_col = metric.capitalize()
        if metric_col not in self.comparison_df.columns:
            return None
        
        best_idx = self.comparison_df[metric_col].idxmax()
        return self.comparison_df.loc[best_idx, "Phase"]
    
    def plot_comparison(self):
        """Return Plotly figure comparing models."""
        if self.comparison_df is None:
            self.build_comparison_table()
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        metrics = ["Accuracy", "Precision", "Recall", "F1"]
        for metric in metrics:
            if metric in self.comparison_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=self.comparison_df["Phase"],
                        y=self.comparison_df[metric],
                        name=metric
                    )
                )
        
        fig.update_layout(
            title="Model Comparison",
            xaxis_title="Phase",
            yaxis_title="Score",
            barmode="group"
        )
        
        return fig
