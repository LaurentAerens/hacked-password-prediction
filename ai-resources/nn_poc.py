"""
Wave 1 NN PoC: Minimal PyTorch CNN for password classification
- PasswordTokenizer: char-level embedding (8-dim for 256 ASCII)
- PasswordCNN: multi-branch conv1d (~5k params)
- PasswordNNTrainer: training interface with loss tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict
import time


class PasswordTokenizer:
    """Char-level tokenizer with fixed-length embedding."""

    def __init__(self, max_length: int = 32, embedding_dim: int = 8):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        # Create embedding layer for ASCII chars (0-255)
        self.embedding = nn.Embedding(256, embedding_dim, padding_idx=0)

    def tokenize(self, password: str) -> torch.Tensor:
        """
        Convert password string to embedded tensor.
        
        Args:
            password: password string
            
        Returns:
            Embedded tensor of shape (1, max_length, embedding_dim)
        """
        # Convert to ASCII codes, truncate/pad to max_length
        ascii_codes = [min(ord(c), 255) for c in password[: self.max_length]]
        # Pad with 0s
        ascii_codes += [0] * (self.max_length - len(ascii_codes))
        # Convert to tensor
        token_tensor = torch.tensor(ascii_codes, dtype=torch.long)
        # Embed
        embedded = self.embedding(token_tensor)  # (max_length, embedding_dim)
        return embedded.unsqueeze(0)  # (1, max_length, embedding_dim)

    def tokenize_batch(self, passwords: List[str]) -> torch.Tensor:
        """
        Tokenize batch of passwords.
        
        Args:
            passwords: list of password strings
            
        Returns:
            Tensor of shape (batch_size, max_length, embedding_dim)
        """
        batch = []
        for pwd in passwords:
            batch.append(self.tokenize(pwd).squeeze(0))
        return torch.stack(batch, dim=0)


class PasswordCNN(nn.Module):
    """Multi-branch CNN with ~5k parameters."""

    def __init__(self, embedding_dim: int = 8, num_filters: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters

        # Multi-branch conv1d (kernels 2, 3, 4)
        self.conv1_k2 = nn.Sequential(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=2, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv1_k3 = nn.Sequential(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv1_k4 = nn.Sequential(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=4, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Compute flattened size after conv + maxpool
        # Input: (batch, 32, 8) -> after conv: (batch, num_filters, 16) -> after maxpool: (batch, num_filters, 8)
        flattened_size = num_filters * 3 * 8  # 3 branches * num_filters * 8

        # Adaptive pooling to ensure consistent output size across branches
        self.adaptive_pool = nn.AdaptiveMaxPool1d(8)

        # Dense layers
        flattened_size = num_filters * 3 * 8  # 3 branches * num_filters * 8
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, max_length, embedding_dim)
            
        Returns:
            logits: (batch_size, 1)
        """
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)  # (batch, embedding_dim, max_length)

        # Multi-branch conv
        out_k2 = self.conv1_k2(x)  # (batch, num_filters, 8)
        out_k3 = self.conv1_k3(x)  # (batch, num_filters, 8)
        out_k4 = self.conv1_k4(x)  # (batch, num_filters, 8)
        
        # Apply adaptive pooling to ensure consistent sizes
        out_k2 = self.adaptive_pool(out_k2)  # (batch, num_filters, 8)
        out_k3 = self.adaptive_pool(out_k3)  # (batch, num_filters, 8)
        out_k4 = self.adaptive_pool(out_k4)  # (batch, num_filters, 8)

        # Concatenate and flatten
        out = torch.cat([out_k2, out_k3, out_k4], dim=1)  # (batch, 3*num_filters, 8)
        out = out.view(out.size(0), -1)  # (batch, 3*num_filters*8)

        # Dense
        logits = self.fc(out)  # (batch, 1)
        return logits

    def param_count(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PasswordDataset(Dataset):
    """PyTorch Dataset for password classification."""

    def __init__(self, passwords: List[str], labels: np.ndarray, tokenizer: PasswordTokenizer):
        self.passwords = passwords
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        pwd = self.passwords[idx]
        label = self.labels[idx]
        embedded = self.tokenizer.tokenize(pwd).squeeze(0)
        return embedded, torch.tensor(label, dtype=torch.float32)


class PasswordNNTrainer:
    """Trainer for PasswordCNN with loss tracking."""

    def __init__(self, model: PasswordCNN, device: str = None, lr: float = 0.001):
        """
        Args:
            model: PasswordCNN model
            device: 'cpu' or 'cuda' (auto-detect if None)
            lr: learning rate for Adam
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Train model for specified epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: number of epochs
            
        Returns:
            Dictionary with loss_history (train + val if available)
        """
        loss_history = {"train": [], "val": []}
        start_time = time.time()

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.loss_fn(logits.squeeze(-1), batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            loss_history["train"].append(train_loss)

            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        logits = self.model(batch_x)
                        loss = self.loss_fn(logits.squeeze(-1), batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                loss_history["val"].append(val_loss)

                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")

        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f}s")
        loss_history["elapsed_time"] = elapsed_time

        return loss_history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Checkpoint loaded: {path}")
