"""PasswordCNN model: Multi-branch CNN for password classification."""

import torch
import torch.nn as nn
from typing import List, Optional


class PasswordCNN(nn.Module):
    """
    Multi-branch CNN for password classification (~5k parameters, frozen design).
    - Input: (batch_size, seq_len, embedding_dim) = (B, 32, 8)
    - 3 parallel conv1d branches: kernels [2, 3, 4]
    - Each branch: Conv1d(8 in, 3 out) → BatchNorm1d → ReLU → MaxPool1d
    - Concatenate: 3 branches × 3 filters × 8 features = 72
    - Dense: 72 → hidden_dim → 1
    - Total params: ~5k (frozen per Wave 2 design)
    """

    def __init__(self, embedding_dim: int = 8, hidden_dim: int = 64, dropout: float = 0.3):
        """
        Initialize PasswordCNN (frozen architecture, 3 filters per kernel).
        
        Args:
            embedding_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension in dense stack
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        num_filters = 3  # Frozen per Wave 2 design
        kernel_sizes = [2, 3, 4]  # Frozen per Wave 2 design
        
        # Conv1d branches (3 parallel kernels)
        self.conv_branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size, padding=kernel_size - 1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )
            self.conv_branches.append(branch)
        
        # After maxpool, we need to calculate the output size
        # Input: (B, 8, 32) for conv1d
        # After conv: (B, 3, 32)
        # After maxpool(2): (B, 3, 16)
        # We'll use adaptive pooling to normalize this
        self.adaptive_pool = nn.AdaptiveMaxPool1d(8)
        
        # Dense layers
        # Input: 3 branches * 3 filters * 8 = 72
        dense_input_dim = len(kernel_sizes) * num_filters * 8
        self.dense = nn.Sequential(
            nn.Linear(dense_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, max_length, embedding_dim)
            
        Returns:
            Logits tensor (B, 1) - NOT probabilities (use BCEWithLogitsLoss)
        """
        # Transpose for Conv1d: (B, max_length, embedding_dim) → (B, embedding_dim, max_length)
        x = x.transpose(1, 2)
        
        # Apply conv branches in parallel
        branch_outputs = []
        for branch in self.conv_branches:
            out = branch(x)  # (B, num_filters, variable_length)
            # Normalize length with adaptive pooling
            out = self.adaptive_pool(out)  # (B, num_filters, 8)
            branch_outputs.append(out)
        
        # Concatenate all branches
        concatenated = torch.cat(branch_outputs, dim=1)  # (B, 3*3=9, 8)
        # Flatten
        flattened = concatenated.view(concatenated.size(0), -1)  # (B, 72)
        
        # Dense layers
        logits = self.dense(flattened)  # (B, 1)
        return logits

    @property
    def num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


class PasswordCNNConfigurable(nn.Module):
    """
    Variant with configurable dense layer count (2-4 layers).
    Conv architecture is frozen (3 filters per kernel [2,3,4]).
    For Wave 4 UI: user can select hidden layer configuration.
    """

    def __init__(self, embedding_dim: int = 8, hidden_dims: Optional[List[int]] = None, dropout: float = 0.3):
        """
        Initialize configurable PasswordCNN.
        
        Args:
            embedding_dim: Input embedding dimension
            hidden_dims: List of hidden layer dimensions (e.g., [64, 32] for 2-layer dense)
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout_rate = dropout
        
        num_filters = 3  # Frozen per Wave 2 design
        kernel_sizes = [2, 3, 4]  # Frozen per Wave 2 design
        
        # Conv1d branches (3 parallel kernels)
        self.conv_branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size, padding=kernel_size - 1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )
            self.conv_branches.append(branch)
        
        # Adaptive pooling to normalize conv output
        self.adaptive_pool = nn.AdaptiveMaxPool1d(8)
        
        # Build configurable dense stack
        dense_input_dim = len(kernel_sizes) * num_filters * 8  # 72
        layers = []
        
        current_dim = dense_input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(current_dim, 1))
        
        self.dense = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, max_length, embedding_dim)
            
        Returns:
            Logits tensor (B, 1)
        """
        # Transpose for Conv1d
        x = x.transpose(1, 2)
        
        # Apply conv branches
        branch_outputs = []
        for branch in self.conv_branches:
            out = branch(x)
            out = self.adaptive_pool(out)
            branch_outputs.append(out)
        
        # Concatenate and flatten
        concatenated = torch.cat(branch_outputs, dim=1)
        flattened = concatenated.view(concatenated.size(0), -1)
        
        # Dense layers
        logits = self.dense(flattened)
        return logits

    @property
    def num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())
