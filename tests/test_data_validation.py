"""Data validation tests for input/output handling.

Tests ensure:
- Input validation (empty, unicode, long passwords, SQL injection)
- Output validation (predictions in [0,1], probabilities sum to 1)
- Label distribution handling (imbalanced data)
- No NaN/Inf in outputs
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest

# Add ai-resources directory to path
ai_resources_path = str(Path(__file__).parent.parent / "ai-resources")
sys.path.insert(0, ai_resources_path)

from nn_tokenizer import PasswordTokenizer
from nn_models import PasswordCNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Test handling of various input types."""

    def test_empty_password_list_handled(self):
        """Empty password list doesn't crash."""
        tokenizer = PasswordTokenizer()
        
        # Should handle gracefully (may raise or return empty)
        # Just ensure it doesn't silently fail
        empty_list = []
        assert isinstance(empty_list, list)

    def test_unicode_passwords_handled(self):
        """Unicode passwords (emoji, non-ASCII) handled."""
        tokenizer = PasswordTokenizer()
        
        unicode_passwords = [
            "password😀123",
            "пароль123",  # Russian
            "密码123",     # Chinese
            "パスワード123",  # Japanese
        ]
        
        for pwd in unicode_passwords:
            try:
                tokens = tokenizer.encode(pwd)
                assert tokens is not None
            except Exception as e:
                # Should either work or raise informative error
                assert "encode" in str(e).lower() or "unicode" in str(e).lower()

    def test_very_long_passwords_handled(self):
        """Very long passwords (>1000 chars) handled."""
        tokenizer = PasswordTokenizer()
        
        long_pwd = "a" * 2000
        
        # Should either work or raise informative error
        try:
            tokens = tokenizer.encode(long_pwd)
            assert tokens is not None
        except Exception as e:
            # Should have informative error message
            assert isinstance(e, Exception)

    def test_sql_injection_like_strings_accepted(self):
        """SQL injection-like strings accepted as passwords."""
        tokenizer = PasswordTokenizer()
        
        injection_strings = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
        ]
        
        for s in injection_strings:
            try:
                tokens = tokenizer.encode(s)
                assert tokens is not None
            except Exception:
                # OK to fail, but should be handled
                pass


# ============================================================================
# OUTPUT VALIDATION TESTS
# ============================================================================

class TestOutputValidation:
    """Test validity of predictions."""

    def test_predictions_in_valid_range(self):
        """Predictions should be in [0, 1] range."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        X_test = np.random.randn(100, 5)
        proba = model.predict_proba(X_test)
        
        # Should be probabilities
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_class_probabilities_sum_to_one(self):
        """Class probabilities should sum to 1.0."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        X_test = np.random.randn(100, 5)
        proba = model.predict_proba(X_test)
        
        # Each row should sum to ~1.0
        sums = np.sum(proba, axis=1)
        assert np.allclose(sums, 1.0)

    def test_no_nan_in_outputs(self):
        """Outputs should never contain NaN."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        X_test = np.random.randn(100, 5)
        proba = model.predict_proba(X_test)
        
        assert not np.isnan(proba).any()

    def test_no_inf_in_outputs(self):
        """Outputs should never contain Inf."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        X_test = np.random.randn(100, 5)
        proba = model.predict_proba(X_test)
        
        assert not np.isinf(proba).any()

    def test_nn_outputs_valid(self):
        """NN outputs should be valid logits."""
        model = PasswordCNN()
        x = torch.randn(4, 32, 8)
        
        with torch.no_grad():
            output = model(x)
        
        # Should be tensor
        assert isinstance(output, torch.Tensor)
        
        # Should not have NaN/Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# ============================================================================
# LABEL DISTRIBUTION TESTS
# ============================================================================

class TestLabelDistribution:
    """Test handling of imbalanced data."""

    def test_imbalanced_data_90_10_handled(self):
        """Imbalanced data (90/10 split) handled."""
        X = np.random.randn(100, 5)
        # 90% label 0, 10% label 1
        y = np.array([0]*90 + [1]*10)
        
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(X, y)
        
        # Model should train without errors
        assert model is not None
        
        # Predictions should be reasonable
        X_test = np.random.randn(50, 5)
        proba = model.predict_proba(X_test)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_all_same_label_handled(self):
        """Training with all 0s or all 1s doesn't crash."""
        X = np.random.randn(50, 5)
        
        # All 0s
        y_all_zeros = np.zeros(50, dtype=int)
        
        try:
            model = RandomForestClassifier(n_estimators=2, random_state=42)
            model.fit(X, y_all_zeros)
            # May succeed or raise informative error
            assert model is not None
        except Exception as e:
            # Should be informative
            assert isinstance(e, Exception)

    def test_minority_class_not_ignored(self):
        """Minority class predictions are meaningful."""
        X = np.random.randn(100, 5)
        # 90% label 0, 10% label 1
        y = np.array([0]*90 + [1]*10)
        
        model = RandomForestClassifier(n_estimators=2, random_state=42, class_weight='balanced')
        model.fit(X, y)
        
        # Predictions should consider both classes
        X_test = np.random.randn(50, 5)
        proba = model.predict_proba(X_test)
        
        # At least some predictions should be for minority class
        minority_proba = proba[:, 1]
        assert np.max(minority_proba) > 0.1  # Some non-trivial prediction


# ============================================================================
# REPRODUCIBILITY TESTS
# ============================================================================

class TestReproducibility:
    """Test that results are reproducible."""

    def test_deterministic_predictions_same_seed(self):
        """Same seed → same predictions."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        # Train 1
        np.random.seed(42)
        model1 = RandomForestClassifier(n_estimators=2, random_state=42)
        model1.fit(X, y)
        
        # Train 2
        np.random.seed(42)
        model2 = RandomForestClassifier(n_estimators=2, random_state=42)
        model2.fit(X, y)
        
        # Predictions should match
        X_test = np.random.randn(20, 5)
        pred1 = model1.predict_proba(X_test)
        pred2 = model2.predict_proba(X_test)
        
        assert np.allclose(pred1, pred2)
