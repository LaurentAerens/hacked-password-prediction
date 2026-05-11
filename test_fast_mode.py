#!/usr/bin/env python3
"""
Quick test of fast mode + mini dataset for rapid iteration.
Demonstrates speedup compared to normal training.
"""
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "ai-resources"))

from adaptive_trainer import train_with_adaptive_search

print("=" * 80)
print("PERFORMANCE COMPARISON: Normal vs Fast Mode")
print("=" * 80)

# Test 1: Mini dataset + Fast mode (should take ~30 seconds)
print("\n🚀 TEST 1: Mini Dataset + Fast Mode (expect ~30 sec)")
print("-" * 80)
start = time.time()
result_fast = train_with_adaptive_search(
    [], [],  # X, y will be generated (empty = use mini dataset)
    mini_dataset=True,
    fast_mode=True,
    cv_folds=2,
    top_percent=0.2,
    n_jobs=-1,
    random_state=42
)
time_fast = time.time() - start
print(f"✅ Fast mode completed in {time_fast:.1f} seconds")
print(f"   Best model: {result_fast['best_config']['model']} + {result_fast['best_config']['preprocessor']}")
print(f"   AUC: {result_fast['best_config']['auc']:.4f}")

# Test 2: Mini dataset without fast mode (should take ~60 seconds)
print("\n⏱️ TEST 2: Mini Dataset + Normal Mode (expect ~60 sec)")
print("-" * 80)
start = time.time()
result_normal = train_with_adaptive_search(
    [], [],  # X, y will be generated
    mini_dataset=True,
    fast_mode=False,
    cv_folds=2,
    top_percent=0.2,
    n_jobs=-1,
    random_state=42
)
time_normal = time.time() - start
print(f"✅ Normal mode completed in {time_normal:.1f} seconds")
print(f"   Best model: {result_normal['best_config']['model']} + {result_normal['best_config']['preprocessor']}")
print(f"   AUC: {result_normal['best_config']['auc']:.4f}")

# Summary
print("\n" + "=" * 80)
print("SPEEDUP SUMMARY")
print("=" * 80)
speedup = time_normal / time_fast
print(f"Fast mode speedup: {speedup:.1f}x faster ({time_normal - time_fast:.1f} sec saved)")
print(f"  Normal mode: {time_normal:.1f} sec")
print(f"  Fast mode:   {time_fast:.1f} sec")

print("\n💡 RECOMMENDATIONS FOR RYZEN 9 7950X:")
print("  1. Use mini_dataset=True for quick validation during development")
print("  2. Use fast_mode=True to iterate quickly on hyperparameter search")
print("  3. On full dataset, expect:")
print("     - Normal mode: 2-4 hours for comprehensive search")
print("     - Fast mode: 30-60 minutes for focused search")
print("  4. System will automatically use deep parallelism (exhaustive_parallel strategy)")
print("     on high-core systems, utilizing all 16 cores efficiently")

print("\n✅ Tests completed successfully!")
