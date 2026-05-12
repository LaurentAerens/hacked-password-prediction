"""PasswordTokenizer: Character-level embedding for password inputs."""

import torch
import torch.nn as nn
from typing import List


class PasswordTokenizer:
    """
    Converts password strings to embedded tensors.
    - Input: password string (e.g., "P@ssw0rd!")
    - Char-level: map each char to 8-dim embedding
    - Padding: fixed max_length=32 chars (pad shorter, truncate longer)
    - Output: torch.Tensor (batch_size, max_length, 8)
    """

    def __init__(self, max_length: int = 32, embedding_dim: int = 8):
        """
        Initialize tokenizer.
        
        Args:
            max_length: Maximum password length (chars beyond this are truncated)
            embedding_dim: Embedding dimension per character
        """
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.device = torch.device('cpu')
        # Create embedding layer for ASCII codes (0-255)
        self.embedding_layer = nn.Embedding(256, embedding_dim, padding_idx=0)

    def encode(self, password: str) -> torch.Tensor:
        """
        Encode single password to tensor.
        
        Args:
            password: Password string
            
        Returns:
            Tensor of shape (1, max_length, embedding_dim)
        """
        # Handle NaN and non-string types
        if password is None or (isinstance(password, float) and password != password):  # NaN check
            password = ""
        else:
            password = str(password)
        
        # Convert password to ASCII codes, truncate/pad to max_length
        ascii_codes = [min(ord(c), 255) for c in password[:self.max_length]]
        # Pad with 0s to reach max_length
        ascii_codes += [0] * (self.max_length - len(ascii_codes))
        
        # Convert to tensor and embed
        token_tensor = torch.tensor(ascii_codes, dtype=torch.long, device=self.device)
        embedded = self.embedding_layer(token_tensor)  # (max_length, embedding_dim)
        return embedded.unsqueeze(0)  # (1, max_length, embedding_dim)

    def batch_encode(self, passwords: List[str]) -> torch.Tensor:
        """
        Encode batch of passwords.
        
        Args:
            passwords: List of password strings or items
            
        Returns:
            Tensor of shape (batch_size, max_length, embedding_dim)
        """
        if len(passwords) == 0:
            return torch.empty((0, self.max_length, self.embedding_dim), dtype=torch.float32)
        
        # Clean batch: convert NaN and non-string types to empty strings
        cleaned_passwords = []
        for password in passwords:
            if password is None or (isinstance(password, float) and password != password):  # NaN check
                cleaned_passwords.append("")
            else:
                cleaned_passwords.append(str(password))
        
        batch = []
        for password in cleaned_passwords:
            # Use encode and squeeze the batch dimension
            encoded = self.encode(password).squeeze(0)  # (max_length, embedding_dim)
            batch.append(encoded)
        
        # Stack all into batch
        return torch.stack(batch, dim=0)  # (batch_size, max_length, embedding_dim)

    @property
    def num_embedding_params(self) -> int:
        """Return embedding parameter count (256 * embedding_dim)."""
        return 256 * self.embedding_dim

    def to(self, device):
        """Move embedding layer to device."""
        self.device = device
        self.embedding_layer = self.embedding_layer.to(device)
        return self
