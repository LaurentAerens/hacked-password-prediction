"""PasswordCNN model: Multi-branch CNN for password classification."""

import torch
import torch.nn as nn
from typing import List, Optional, Union

_VALID_ACTIVATIONS = ("relu", "tanh", "sigmoid", "leaky_relu", "elu")


def _resolve_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name == "elu":
        return nn.ELU()
    raise ValueError(
        f"Unknown activation '{name}'. Valid options: {', '.join(_VALID_ACTIVATIONS)}"
    )


def _coerce_layer_entry(entry: Union[int, dict]) -> dict:
    """Coerce int or dict entry to canonical dict form."""
    if isinstance(entry, int):
        return {"units": entry, "activation": "relu"}
    if isinstance(entry, dict):
        if "units" not in entry:
            raise ValueError(f"Layer dict missing required key 'units': {entry}")
        return entry
    raise ValueError(f"Layer entry must be int or dict, got {type(entry).__name__}: {entry}")


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

    def __init__(self, embedding_dim: int = 8, hidden_dims=None, dropout: float = 0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout

        # Normalize hidden_dims: each entry → {"units": int, "activation": str, optional "dropout": float}
        raw_dims = hidden_dims or [64, 32]
        normalized: List[dict] = [_coerce_layer_entry(e) for e in raw_dims]
        # Validate activations up-front so errors surface at init time
        for entry in normalized:
            _resolve_activation(entry.get("activation", "relu"))
        self.hidden_dims = normalized

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
        for entry in self.hidden_dims:
            units = entry["units"]
            act = entry.get("activation", "relu")
            layer_dropout = entry.get("dropout", dropout)
            layers.append(nn.Linear(current_dim, units))
            layers.append(_resolve_activation(act))
            layers.append(nn.Dropout(layer_dropout))
            current_dim = units

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


def validate_layer_spec(layer_spec) -> tuple:
    """
    Validate a layer spec without raising.

    Returns (True, "") if valid, (False, "<reason>") otherwise.
    """
    try:
        if not isinstance(layer_spec, list):
            return (False, f"layer_spec must be a list, got {type(layer_spec).__name__}")
        if len(layer_spec) == 0:
            return (False, "layer_spec must be a non-empty list")
        for i, entry in enumerate(layer_spec):
            if isinstance(entry, int):
                if entry <= 0:
                    return (False, f"Entry {i}: units must be > 0, got {entry}")
            elif isinstance(entry, dict):
                if "units" not in entry:
                    return (False, f"Entry {i}: dict missing required key 'units'")
                units = entry["units"]
                if not isinstance(units, int) or units <= 0:
                    return (False, f"Entry {i}: units must be a positive int, got {units!r}")
                act = entry.get("activation", "relu")
                if act not in _VALID_ACTIVATIONS:
                    return (False, f"Entry {i}: unknown activation '{act}'. Valid: {', '.join(_VALID_ACTIVATIONS)}")
            else:
                return (False, f"Entry {i}: must be int or dict, got {type(entry).__name__}")
        return (True, "")
    except Exception as exc:  # pragma: no cover — defensive catch
        return (False, str(exc))


def build_model_from_spec(
    layer_spec: list,
    dropout: float = 0.2,
    device=None,
) -> "PasswordCNNConfigurable":
    """
    Build a PasswordCNNConfigurable from a layer spec.

    Validates, normalizes (int entries → dict), then constructs the model.
    Moves to `device` if provided.
    """
    if not isinstance(layer_spec, list):
        raise ValueError(f"layer_spec must be a list, got {type(layer_spec).__name__}")
    if len(layer_spec) == 0:
        raise ValueError("layer_spec must be a non-empty list")
    # Validate units before construction
    for i, entry in enumerate(layer_spec):
        units = entry if isinstance(entry, int) else entry.get("units")
        if units is None:
            raise ValueError(f"Entry {i}: dict missing required key 'units'")
        if not isinstance(units, int) or units <= 0:
            raise ValueError(f"Entry {i}: units must be a positive int, got {units!r}")

    model = PasswordCNNConfigurable(hidden_dims=layer_spec, dropout=dropout)
    if device is not None:
        model = model.to(device)
    return model

