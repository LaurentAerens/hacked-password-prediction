"""Quick debug script for training loop."""

import sys
from pathlib import Path
import torch
import pandas as pd
import tempfile

# Add ai-resources directory to path
ai_resources_path = str(Path(__file__).parent / "ai-resources")
sys.path.insert(0, ai_resources_path)

from nn_trainer import PasswordNNTrainer
from nn_tokenizer import PasswordTokenizer
from nn_models import PasswordCNN

# Test 1: Tokenizer
print("=== Testing Tokenizer ===")
tokenizer = PasswordTokenizer()
encoded = tokenizer.encode("test")
print(f"Encoded shape: {encoded.shape}, requires_grad: {encoded.requires_grad}")

# Test 2: Model
print("\n=== Testing Model ===")
model = PasswordCNN()
x = torch.randn(4, 32, 8, requires_grad=True)
print(f"Input requires_grad: {x.requires_grad}")

output = model(x)
print(f"Output shape: {output.shape}, requires_grad: {output.requires_grad}")

# Test 3: Loss and backward
print("\n=== Testing Loss ===")
y = torch.randn(4, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
loss = loss_fn(output, y)
print(f"Loss: {loss.item()}, requires_grad: {loss.requires_grad}")

try:
    loss.backward()
    print("Backward successful!")
except Exception as e:
    print(f"Backward failed: {e}")

# Test 4: Full training
print("\n=== Testing Full Training ===")
# Create minimal dataset
passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 4
labels = [0, 1] * 10
X = pd.Series(passwords)
y = pd.Series(labels)

print(f"Data shape: X={X.shape}, y={y.shape}")

# Train
with tempfile.TemporaryDirectory() as tmpdir:
    trainer = PasswordNNTrainer(model_dir=tmpdir)
    try:
        result = trainer.train(X, y, epochs=1, batch_size=4)
        print("Training successful!")
        print(f"Result keys: {result.keys()}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
