"""Tests for PasswordTokenizer (character-level embedding)."""

from pathlib import Path
import torch
import pytest

from harp.nn_tokenizer import PasswordTokenizer


class TestPasswordTokenizer:
    """Unit tests for PasswordTokenizer."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        tokenizer = PasswordTokenizer()
        assert tokenizer.max_length == 32
        assert tokenizer.embedding_dim == 8

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        tokenizer = PasswordTokenizer(max_length=64, embedding_dim=16)
        assert tokenizer.max_length == 64
        assert tokenizer.embedding_dim == 16

    def test_encode_single_password(self):
        """Test encoding a single password."""
        tokenizer = PasswordTokenizer()
        password = "P@ssw0rd!"
        result = tokenizer.encode(password)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 32, 8)  # (batch=1, max_length, embedding_dim)
        assert result.dtype == torch.float32

    def test_encode_padding_short_password(self):
        """Test that short passwords are padded."""
        tokenizer = PasswordTokenizer()
        password = "abc"
        result = tokenizer.encode(password)
        
        # Should be padded to 32
        assert result.shape == (1, 32, 8)

    def test_encode_truncation_long_password(self):
        """Test that long passwords are truncated."""
        tokenizer = PasswordTokenizer()
        password = "a" * 50  # Longer than max_length=32
        result = tokenizer.encode(password)
        
        assert result.shape == (1, 32, 8)

    def test_batch_encode(self):
        """Test batch encoding multiple passwords."""
        tokenizer = PasswordTokenizer()
        passwords = ["P@ssw0rd!", "simple", "VeryLong1!@#$%^&*()[]{}"]
        result = tokenizer.batch_encode(passwords)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 8)  # (batch_size=3, max_length, embedding_dim)
        assert result.dtype == torch.float32

    def test_batch_encode_empty_list(self):
        """Test batch encoding with empty list."""
        tokenizer = PasswordTokenizer()
        passwords = []
        result = tokenizer.batch_encode(passwords)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (0, 32, 8)

    def test_num_embedding_params_property(self):
        """Test num_embedding_params property."""
        tokenizer = PasswordTokenizer()
        # 256 ASCII codes * 8-dim embeddings = 2,048
        assert tokenizer.num_embedding_params == 256 * 8

    def test_device_cpu(self):
        """Test that tokenizer can be moved to CPU."""
        tokenizer = PasswordTokenizer()
        tokenizer_cpu = tokenizer.to("cpu")
        
        password = "test123"
        result = tokenizer_cpu.encode(password)
        assert result.device.type == "cpu"

    def test_device_cuda(self):
        """Test that tokenizer can be moved to GPU (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        tokenizer = PasswordTokenizer()
        tokenizer_gpu = tokenizer.to("cuda")
        
        password = "test123"
        result = tokenizer_gpu.encode(password)
        assert result.device.type == "cuda"

    def test_consistency_same_password(self):
        """Test that same password produces same embedding."""
        tokenizer = PasswordTokenizer()
        password = "P@ssw0rd!"
        
        result1 = tokenizer.encode(password)
        result2 = tokenizer.encode(password)
        
        assert torch.allclose(result1, result2)

    def test_different_passwords_different_embeddings(self):
        """Test that different passwords produce different embeddings."""
        tokenizer = PasswordTokenizer()
        
        result1 = tokenizer.encode("password1")
        result2 = tokenizer.encode("password2")
        
        # At least some values should differ
        assert not torch.allclose(result1, result2)

    def test_batch_encode_consistency_vs_single(self):
        """Test that batch encoding matches individual encoding."""
        tokenizer = PasswordTokenizer()
        passwords = ["pass1", "pass2", "pass3"]
        
        # Batch encode
        batch_result = tokenizer.batch_encode(passwords)
        
        # Individual encodes
        for i, pwd in enumerate(passwords):
            single_result = tokenizer.encode(pwd)
            # Compare batch[i] with squeeze(0) of single result
            assert torch.allclose(batch_result[i:i+1], single_result)
