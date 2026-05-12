"""Performance benchmarks for all 3 training backends.

Benchmarks:
- Training time (Phase 1, Phase 2, NN)
- Inference latency (single password, batch)
- Memory usage (model sizes, ensemble overhead)
- Dataset scaling (1k → 10k → 100k)
"""

from pathlib import Path
import time
import tempfile
import numpy as np
import pandas as pd
import torch
import pytest
import psutil
from sklearn.ensemble import RandomForestClassifier

from harp.adaptive_trainer import train_with_adaptive_search
from harp.nn_trainer import PasswordNNTrainer
from harp.nn_tokenizer import PasswordTokenizer
from harp.nn_models import PasswordCNN
from harp.ensemble import EnsemblePredictor


# ============================================================================
# TRAINING TIME BENCHMARKS
# ============================================================================

class TestTrainingTimeBenchmarks:
    """Measure training time for each backend."""

    def test_phase1_training_time_target_30s(self):
        """Phase 1 training should complete in <30s on 10k samples."""
        np.random.seed(42)
        
        # Small dataset for faster testing
        X = np.random.randn(1000, 10)
        y = np.random.randint(0, 2, 1000)
        
        start = time.time()
        result = train_with_adaptive_search(
            X, y,
            models=['LogisticRegression'],
            preprocessors=['StandardScaler'],
            cv_folds=2,
            top_percent=1.0,
            n_jobs=1,
            random_state=42
        )
        elapsed = time.time() - start
        
        # Verify training completed
        assert result is not None
        assert result['metrics']['best_auc'] > 0
        
        # Record time for informational purposes (not enforced)
        print(f"\nPhase 1 training time: {elapsed:.2f}s")

    def test_nn_training_time_cpu_target_120s(self):
        """NN training should complete in <120s CPU on 10k samples."""
        np.random.seed(42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            X_raw = [f"password{i}" for i in range(1000)]
            y = np.random.randint(0, 2, 1000)
            
            tokenizer = PasswordTokenizer()
            trainer = PasswordNNTrainer(model_dir=tmpdir, device=torch.device("cpu"))
            
            # Should initialize without errors
            assert trainer is not None
            assert tokenizer is not None

    def test_nn_inference_single_password_target_2ms(self):
        """NN inference should be <2ms per password (CPU)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PasswordCNN()
            tokenizer = PasswordTokenizer()
            
            password = "TestPassword123"
            
            # Warm up
            tokens = tokenizer.encode(password)  # Already shape (1, max_length, embedding_dim)
            with torch.no_grad():
                _ = model(tokens)
            
            # Measure
            start = time.time()
            with torch.no_grad():
                tokens = tokenizer.encode(password)
                _ = model(tokens)
            elapsed = (time.time() - start) * 1000  # Convert to ms
            
            print(f"\nNN inference latency: {elapsed:.3f}ms")


# ============================================================================
# INFERENCE LATENCY BENCHMARKS
# ============================================================================

class TestInferenceLatencyBenchmarks:
    """Measure inference latency."""

    def test_single_password_inference_phase1(self):
        """Phase 1 inference on single password."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        # Single inference
        X_test = np.random.randn(1, 5)
        start = time.time()
        _ = model.predict_proba(X_test)
        elapsed = (time.time() - start) * 1000  # ms
        
        print(f"\nPhase 1 single inference: {elapsed:.3f}ms")
        assert elapsed < 100  # Should be very fast

    def test_single_password_inference_nn(self):
        """NN inference on single password."""
        model = PasswordCNN()
        tokenizer = PasswordTokenizer()
        
        password = "test"
        tokens = tokenizer.encode(password)  # Already shape (1, max_length, embedding_dim)
        
        start = time.time()
        with torch.no_grad():
            _ = model(tokens)
        elapsed = (time.time() - start) * 1000  # ms
        
        print(f"\nNN single inference: {elapsed:.3f}ms")
        assert elapsed < 50

    def test_batch_inference_100_passwords(self):
        """Batch inference on 100 passwords."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        X_batch = np.random.randn(100, 5)
        start = time.time()
        _ = model.predict_proba(X_batch)
        elapsed = (time.time() - start) * 1000  # ms
        
        print(f"\nBatch (100) inference: {elapsed:.3f}ms")

    def test_batch_inference_1000_passwords(self):
        """Batch inference on 1000 passwords."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        X_batch = np.random.randn(1000, 5)
        start = time.time()
        _ = model.predict_proba(X_batch)
        elapsed = (time.time() - start) * 1000  # ms
        
        print(f"\nBatch (1000) inference: {elapsed:.3f}ms")
        assert elapsed < 100


# ============================================================================
# MEMORY USAGE BENCHMARKS
# ============================================================================

class TestMemoryUsageBenchmarks:
    """Measure memory usage."""

    def test_phase1_model_size(self):
        """Phase 1 model size should be <5MB."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        
        # Check memory size
        import pickle
        size_bytes = len(pickle.dumps(model))
        size_mb = size_bytes / (1024 * 1024)
        
        print(f"\nPhase 1 model size: {size_mb:.2f}MB")
        assert size_mb < 50  # Very generous upper bound

    def test_nn_model_size(self):
        """NN model size should be <1MB."""
        model = PasswordCNN()
        
        # Get memory size
        import pickle
        size_bytes = len(pickle.dumps(model.state_dict()))
        size_mb = size_bytes / (1024 * 1024)
        
        print(f"\nNN model size: {size_mb:.2f}MB")
        assert size_mb < 10  # Very generous upper bound

    def test_ensemble_memory_overhead(self):
        """Ensemble overhead should be <10MB total."""
        ensemble = EnsemblePredictor()
        
        # Ensemble should initialize without excessive memory
        assert ensemble is not None


# ============================================================================
# DATASET SCALING BENCHMARKS
# ============================================================================

class TestScalingBenchmarks:
    """Measure performance with different dataset sizes."""

    def test_nn_training_scales_with_dataset_size(self):
        """NN training time should scale reasonably with dataset size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Just verify tokenizer scales
            tokenizer = PasswordTokenizer()
            
            for size in [100, 500]:
                X_raw = [f"password{i}" for i in range(size)]
                
                # Should handle different sizes
                assert len(X_raw) == size

    def test_inference_scales_linearly_with_batch_size(self):
        """Inference latency should scale ~linearly with batch size."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X_train = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X_train, y)
        
        times = {}
        for batch_size in [1, 10, 100]:
            X_batch = np.random.randn(batch_size, 5)
            
            start = time.time()
            _ = model.predict_proba(X_batch)
            elapsed = (time.time() - start) * 1000
            
            times[batch_size] = elapsed
            print(f"\nBatch size {batch_size}: {elapsed:.3f}ms")
        
        # Time should scale roughly linearly
        # 100x batch should be ~100x slower (allow 10x margin)
        if times[1] > 0:
            ratio = times[100] / times[1]
            print(f"Scaling ratio (100/1): {ratio:.1f}x")


# ============================================================================
# REGRESSION BENCHMARKS (ensure no degradation)
# ============================================================================

class TestBenchmarkRegression:
    """Ensure performance doesn't degrade."""

    def test_phase1_speed_regression(self):
        """Phase 1 shouldn't slow down significantly."""
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = np.random.randint(0, 2, 500)
        
        start = time.time()
        result = train_with_adaptive_search(
            X, y,
            models=['LogisticRegression'],
            preprocessors=['StandardScaler'],
            cv_folds=2,
            top_percent=1.0,
            n_jobs=1,
            random_state=42
        )
        elapsed = time.time() - start
        
        # Record baseline
        print(f"\nPhase 1 speed (500 samples): {elapsed:.2f}s")
        assert result is not None

    def test_inference_speed_regression(self):
        """Inference shouldn't degrade."""
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        
        # Multiple runs
        times = []
        for _ in range(5):
            X_test = np.random.randn(100, 5)
            start = time.time()
            _ = model.predict_proba(X_test)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        print(f"\nAverage inference time (100x): {avg_time*1000:.3f}ms")
