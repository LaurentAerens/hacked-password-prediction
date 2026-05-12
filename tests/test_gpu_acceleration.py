#!/usr/bin/env python3
"""
GPU Acceleration Demo - Shows XGBoost/LightGBM speedup on GPU vs CPU.

Requirements: XGBoost, LightGBM with GPU support
"""

import time
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification

from harp.gpu_utils import GPUDetector, GPUTrainerConfig, print_gpu_info


def benchmark_xgboost():
    """Benchmark XGBoost on GPU vs CPU."""
    try:
        import xgboost as xgb
    except ImportError:
        print("❌ XGBoost not installed")
        return
    
    print("\n" + "=" * 80)
    print("XGBoost Benchmark: GPU vs CPU")
    print("=" * 80)
    
    # Generate synthetic data
    X, y = make_classification(n_samples=100000, n_features=100, n_informative=50,
                               random_state=42)
    
    # CPU training
    print("\n🔵 CPU Training (hist tree method)...")
    start = time.time()
    cpu_model = xgb.XGBClassifier(
        n_estimators=100,
        tree_method='hist',
        random_state=42,
        verbosity=0
    )
    cpu_model.fit(X, y)
    cpu_time = time.time() - start
    cpu_score = cpu_model.score(X, y)
    print(f"   ✅ CPU completed in {cpu_time:.2f}s (accuracy: {cpu_score:.4f})")
    
    # GPU training (if available)
    gpu_config = GPUTrainerConfig(device='auto')
    if gpu_config.is_gpu_available:
        print(f"\n🔴 GPU Training ({gpu_config.actual_device} gpu_hist tree method)...")
        start = time.time()
        gpu_model = xgb.XGBClassifier(
            n_estimators=100,
            tree_method='gpu_hist',
            gpu_id=0,
            device='cuda',
            random_state=42,
            verbosity=0
        )
        gpu_model.fit(X, y)
        gpu_time = time.time() - start
        gpu_score = gpu_model.score(X, y)
        print(f"   ✅ GPU completed in {gpu_time:.2f}s (accuracy: {gpu_score:.4f})")
        
        speedup = cpu_time / gpu_time
        print(f"\n🚀 Speedup: {speedup:.1f}x faster on GPU!")
        print(f"   CPU: {cpu_time:.2f}s | GPU: {gpu_time:.2f}s | Saved: {cpu_time - gpu_time:.2f}s")
    else:
        print("   GPU not available, skipping GPU training")


def benchmark_lightgbm():
    """Benchmark LightGBM on GPU vs CPU."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("❌ LightGBM not installed")
        return
    
    print("\n" + "=" * 80)
    print("LightGBM Benchmark: GPU vs CPU")
    print("=" * 80)
    
    # Generate synthetic data
    X, y = make_classification(n_samples=100000, n_features=100, n_informative=50,
                               random_state=42)
    
    # CPU training
    print("\n🔵 CPU Training...")
    start = time.time()
    cpu_model = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=42,
        verbose=-1
    )
    cpu_model.fit(X, y)
    cpu_time = time.time() - start
    cpu_score = cpu_model.score(X, y)
    print(f"   ✅ CPU completed in {cpu_time:.2f}s (accuracy: {cpu_score:.4f})")
    
    # GPU training (if available)
    gpu_config = GPUTrainerConfig(device='auto')
    if gpu_config.is_gpu_available:
        print(f"\n🔴 GPU Training ({gpu_config.actual_device})...")
        start = time.time()
        gpu_model = lgb.LGBMClassifier(
            n_estimators=100,
            device='gpu',
            gpu_platform_id=0,
            gpu_device_id=0,
            random_state=42,
            verbose=-1
        )
        gpu_model.fit(X, y)
        gpu_time = time.time() - start
        gpu_score = gpu_model.score(X, y)
        print(f"   ✅ GPU completed in {gpu_time:.2f}s (accuracy: {gpu_score:.4f})")
        
        speedup = cpu_time / gpu_time
        print(f"\n🚀 Speedup: {speedup:.1f}x faster on GPU!")
        print(f"   CPU: {cpu_time:.2f}s | GPU: {gpu_time:.2f}s | Saved: {cpu_time - gpu_time:.2f}s")
    else:
        print("   GPU not available, skipping GPU training")


def show_expected_improvements():
    """Show expected improvements with GPU acceleration."""
    print("\n" + "=" * 80)
    print("Expected Performance Improvements with GPU Acceleration")
    print("=" * 80)
    
    print("""
On RTX 4090 (24GB VRAM):
- XGBoost: 5-10x faster training on large datasets (100K+ samples)
- LightGBM: 3-8x faster training on large datasets
- GridSearchCV with GPU models: 3-5x faster hyperparameter search

Training Time Estimates on Full Dataset (300K+ samples):
┌──────────────────────────────────────────────────────────┐
│ Scenario                          │ CPU (16 cores) │ GPU  │
├─────────────────────────────────────────────────────────┤
│ XGBoost GridSearch (100 evals)   │ 4-6 hours     │ 30-45 min │
│ LightGBM GridSearch (100 evals)  │ 3-5 hours     │ 20-30 min │
│ Mixed models + GPU-optimized     │ 2-4 hours     │ 15-25 min │
│ Fast mode + Mini dataset         │ 30-60 min     │ 5-10 min  │
└──────────────────────────────────────────────────────────┘

Combined Optimization Stack:
- GPU acceleration: 3-10x on GPU-capable models
- Fast mode (reduced grids): 2x on all models
- Loky backend (CPU parallelism): 3-5x on multi-core
- Combined effect: 15-30x faster than original single-core threading!

Recommendation:
1. Start with mini_dataset=True, gpu_device='auto' → 5-10 sec validation
2. Use fast_mode=True for quick iteration on full data → 20-30 min
3. Run full search with fast_mode=False, gpu_device='cuda' → 20-30 min (GPU models)
    """)


if __name__ == '__main__':
    print_gpu_info()
    
    try:
        benchmark_xgboost()
    except Exception as e:
        print(f"⚠️  XGBoost benchmark failed: {e}")
    
    try:
        benchmark_lightgbm()
    except Exception as e:
        print(f"⚠️  LightGBM benchmark failed: {e}")
    
    show_expected_improvements()
    
    print("\n✅ GPU acceleration demo completed!")
