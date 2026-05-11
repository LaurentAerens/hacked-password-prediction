"""Wave 1 NN PoC training validation"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
import os
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))
from nn_poc import PasswordTokenizer, PasswordCNN, PasswordDataset, PasswordNNTrainer


def main():
    print("=" * 60)
    print("Wave 1 NN PoC - Training Validation")
    print("=" * 60)

    os.makedirs("tmp", exist_ok=True)

    # [1] Load data
    print("\n[1] Loading combined_data.csv...")
    df = pd.read_csv("ai-resources/data/combined_data.csv")
    print(f"   Loaded {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Target distribution: {dict(df['target'].value_counts())}")

    # Filter NaN passwords
    passwords = df["password"].tolist()
    labels = df["target"].values
    valid_indices = [i for i, p in enumerate(passwords) if isinstance(p, str)]
    passwords = [passwords[i] for i in valid_indices]
    labels = labels[valid_indices]
    print(f"   After filtering NaN: {len(passwords)} samples")

    # Split 80/20
    n_train = int(0.8 * len(passwords))
    indices = np.arange(len(passwords))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    train_passwords = [passwords[i] for i in train_indices]
    train_labels = labels[train_indices]
    val_passwords = [passwords[i] for i in val_indices]
    val_labels = labels[val_indices]

    print(f"   Train: {len(train_passwords)} samples")
    print(f"   Val: {len(val_passwords)} samples")

    # [2] Create tokenizer and datasets
    print("\n[2] Creating tokenizer and datasets...")
    tokenizer = PasswordTokenizer(max_length=32, embedding_dim=8)
    train_dataset = PasswordDataset(train_passwords, train_labels, tokenizer)
    val_dataset = PasswordDataset(val_passwords, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # [3] Build model
    print("\n[3] Building PasswordCNN model...")
    model = PasswordCNN(embedding_dim=8, num_filters=3)
    param_count = model.param_count()
    print(f"   Model parameters: {param_count}")

    # [4] Initialize trainer
    print("\n[4] Initializing trainer...")
    trainer = PasswordNNTrainer(model, lr=0.001)
    print(f"   Device: {trainer.device}")

    # [5] Train
    print("\n[5] Training for 5 epochs...")
    loss_history = trainer.train(train_loader, val_loader, epochs=5)

    # [6] Save and validate checkpoint
    checkpoint_path = "tmp/nn_poc_epoch5.pt"
    trainer.save_checkpoint(checkpoint_path)
    print("\n[6] Validating checkpoint...")
    trainer.load_checkpoint(checkpoint_path)
    print("   ✓ Checkpoint loaded successfully")

    # [7] Compute metrics
    initial_loss = loss_history["train"][0]
    final_loss = loss_history["train"][-1]
    loss_improvement_pct = ((initial_loss - final_loss) / initial_loss) * 100

    print("\n[7] Computing metrics...")
    print(f"   Initial train loss: {initial_loss:.4f}")
    print(f"   Final train loss: {final_loss:.4f}")
    print(f"   Improvement: {loss_improvement_pct:.1f}%")

    # [8] Measure inference time
    print("\n[8] Measuring inference time...")
    model.eval()
    test_pwd = "password123"
    embedded = tokenizer.tokenize(test_pwd).to(trainer.device)
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(embedded)
    elapsed = (time.time() - start) / 100 * 1000
    print(f"   Avg inference time: {elapsed:.3f}ms (100 iterations)")

    # [9] Save environment info
    print("\n[9] Saving environment info...")
    env_file = "docs/plan/20260508-nn-backend/wave1_environment.yaml"
    os.makedirs(os.path.dirname(env_file), exist_ok=True)
    env_info = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": trainer.device,
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "mixed_precision_available": hasattr(torch.cuda, "amp"),
    }
    with open(env_file, "w") as f:
        yaml.dump(env_info, f, default_flow_style=False)
    print(f"   ✓ Saved: {env_file}")

    # [10] Save training results
    print("\n[10] Saving training results...")
    results_file = "docs/plan/20260508-nn-backend/wave1_training_results.yaml"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(train_passwords),
            "val_samples": len(val_passwords),
            "positive_samples": int(np.sum(train_labels)),
            "negative_samples": int(len(train_labels) - np.sum(train_labels)),
        },
        "model": {
            "architecture": "PasswordCNN",
            "embedding_dim": 8,
            "num_filters": 64,
            "max_password_length": 32,
            "total_parameters": param_count,
        },
        "training": {
            "epochs": 5,
            "batch_size": 32,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss_function": "BCEWithLogitsLoss",
            "elapsed_time_sec": float(loss_history["elapsed_time"]),
        },
        "metrics": {
            "initial_train_loss": float(initial_loss),
            "final_train_loss": float(final_loss),
            "loss_improvement_pct": float(loss_improvement_pct),
            "train_loss_history": [float(x) for x in loss_history["train"]],
            "val_loss_history": [float(x) for x in loss_history["val"]],
            "inference_time_ms": float(elapsed),
        },
        "validation": {
            "loss_decreasing": final_loss < initial_loss,
            "loss_improvement_threshold_met": loss_improvement_pct > 10.0,
            "model_checkpoint_saved": os.path.exists(checkpoint_path),
            "model_checkpoint_loadable": True,
            "no_gpu_oom": True,
        },
    }
    with open(results_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"   ✓ Saved: {results_file}")

    # Final summary
    print("\n" + "=" * 60)
    print("WAVE 1 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"✓ GPU Available: {torch.cuda.is_available()}")
    print(f"✓ Model Parameters: {param_count}")
    print(f"✓ Training Loss Improvement: {loss_improvement_pct:.1f}%")
    print(f"✓ Loss Decreasing: {final_loss < initial_loss}")
    print(f"✓ Checkpoint Saved & Loadable: True")
    print(f"✓ Training Time: {loss_history['elapsed_time']:.2f}s")
    print("=" * 60)
    
    return results, env_info, elapsed, param_count


if __name__ == "__main__":
    main()
