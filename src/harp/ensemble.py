"""EnsemblePredictor: Combines predictions from Phase 1, Phase 2, and NN models."""

import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from typing import Optional


class EnsemblePredictor:
    """
    Combines predictions from Phase 1 (sklearn), Phase 2 (sklearn), NN (PyTorch).
    Soft voting: average probabilities across all available models.
    """
    
    def __init__(self):
        self.phase1_model = None
        self.phase2_model = None
        self.nn_model = None
        self.tokenizer = None
        self.enabled_models = {
            "phase1": False,
            "phase2": False,
            "nn": False,
        }
    
    def load_phase1_model(self, checkpoint_path: str):
        """Load Phase 1 best model (sklearn pickle)."""
        self.phase1_model = joblib.load(checkpoint_path)
        self.enabled_models["phase1"] = True
    
    def load_phase2_model(self, checkpoint_path: str):
        """Load Phase 2 best model (sklearn pickle)."""
        self.phase2_model = joblib.load(checkpoint_path)
        self.enabled_models["phase2"] = True
    
    def load_nn_model(self, checkpoint_path: str):
        """Load NN best model (PyTorch state_dict)."""
        from .nn_models import PasswordCNN
        from .nn_tokenizer import PasswordTokenizer
        
        self.nn_model = PasswordCNN()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.nn_model.load_state_dict(state_dict)
        self.nn_model.eval()
        
        self.tokenizer = PasswordTokenizer()
        self.enabled_models["nn"] = True
    
    def predict_proba(
        self,
        passwords: pd.Series,
        ensemble_method: str = "soft_vote"
    ) -> np.ndarray:
        """
        Predict probabilities for passwords using available models.
        
        Returns: (n_samples, 2) array [prob_not_hacked, prob_hacked]
        """
        probas = []
        weights = []
        
        # Phase 1 prediction (if enabled)
        if self.enabled_models["phase1"] and self.phase1_model:
            try:
                proba = self.phase1_model.predict_proba(passwords)
                probas.append(proba)
                weights.append(1.0)
            except Exception as e:
                pass
        
        # Phase 2 prediction (if enabled)
        if self.enabled_models["phase2"] and self.phase2_model:
            try:
                proba = self.phase2_model.predict_proba(passwords)
                probas.append(proba)
                weights.append(1.0)
            except Exception as e:
                pass
        
        # NN prediction (if enabled)
        if self.enabled_models["nn"] and self.nn_model:
            try:
                with torch.no_grad():
                    x = self.tokenizer.batch_encode(passwords.tolist())
                    logits = self.nn_model(x)
                    probs = torch.sigmoid(logits).numpy()
                    # Expand to (n, 2) format: [prob_0, prob_1]
                    proba = np.hstack([1 - probs, probs])
                    probas.append(proba)
                    weights.append(1.0)
            except Exception as e:
                pass
        
        if not probas:
            raise ValueError("No models enabled for ensemble")
        
        # Soft voting: weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        ensemble_proba = np.zeros_like(probas[0], dtype=float)
        for proba, weight in zip(probas, weights):
            ensemble_proba += weight * proba
        
        return ensemble_proba
    
    def predict(
        self,
        passwords: pd.Series,
        ensemble_method: str = "soft_vote",
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict binary labels (0/1).
        threshold: probability >= threshold → label 1 (hacked)
        """
        proba = self.predict_proba(passwords, ensemble_method)
        return (proba[:, 1] >= threshold).astype(int)
    
    def get_active_model_count(self) -> int:
        """Return number of enabled models."""
        return sum(self.enabled_models.values())
    
    @property
    def ensemble_config(self) -> dict:
        """Return config dict for UI display."""
        return {
            "phase1": self.enabled_models["phase1"],
            "phase2": self.enabled_models["phase2"],
            "nn": self.enabled_models["nn"],
            "method": "soft_vote",
            "active_count": self.get_active_model_count(),
        }
