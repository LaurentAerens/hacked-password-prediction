#!/usr/bin/env python3
"""Test neural network training with device placement fix"""

import torch
import numpy as np
import pandas as pd
from harp.nn_trainer import PasswordNNTrainer

print("[Testing] Creating dummy data...")
X = pd.Series([' '.join(chr(np.random.randint(65, 91)) for _ in range(12)) for _ in range(100)])
y = pd.Series(np.random.randint(0, 2, (100,)))

print("[Testing] Initializing NN trainer...")
trainer = PasswordNNTrainer(model_dir="models/nn", device=None)

print("[Testing] Training model...")
try:
    trainer.train(X, y, epochs=2, batch_size=16, learning_rate=0.001, val_split=0.2)
    print("\n[SUCCESS] NN training completed without device mismatch errors!")
    print(f"[Device Info] Model device: {trainer.device}")
    print(f"[Device Info] CUDA available: {torch.cuda.is_available()}")
except RuntimeError as e:
    if "device" in str(e).lower():
        print(f"\n[FAILED] Device mismatch error: {e}")
    else:
        raise
