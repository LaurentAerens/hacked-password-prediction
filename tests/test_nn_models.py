"""Tests for PasswordCNN model."""

import sys
from pathlib import Path
import torch
import pytest

# Add ai-resources directory to path
ai_resources_path = str(Path(__file__).parent.parent / "ai-resources")
sys.path.insert(0, ai_resources_path)

from nn_models import PasswordCNN, PasswordCNNConfigurable


class TestPasswordCNN:
    """Unit tests for PasswordCNN model."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        model = PasswordCNN()
        assert model is not None

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        model = PasswordCNN(embedding_dim=16, hidden_dim=128, dropout=0.5)
        assert model is not None

    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        model = PasswordCNN()
        batch_size = 4
        seq_len = 32
        embedding_dim = 8
        
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = model(x)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 1)  # (batch_size, 1) for binary classification

    def test_forward_pass_dtype(self):
        """Test forward pass returns float tensors."""
        model = PasswordCNN()
        x = torch.randn(4, 32, 8)
        output = model(x)
        
        assert output.dtype == torch.float32

    def test_forward_pass_logits_not_probabilities(self):
        """Test that output is logits, not probabilities."""
        model = PasswordCNN()
        x = torch.randn(4, 32, 8)
        output = model(x)
        
        # Logits can be outside [0, 1]
        # Just verify it's produced without sigmoid normalization
        assert output.min().item() < 0 or output.max().item() > 1

    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        model = PasswordCNN()
        x = torch.randn(1, 32, 8)
        output = model(x)
        
        assert output.shape == (1, 1)

    def test_forward_pass_large_batch(self):
        """Test forward pass with large batch."""
        model = PasswordCNN()
        x = torch.randn(128, 32, 8)
        output = model(x)
        
        assert output.shape == (128, 1)

    def test_num_params_property(self):
        """Test num_params property returns ~5k."""
        model = PasswordCNN()
        num_params = model.num_params
        
        # Should be approximately 5000 (frozen design)
        assert 4800 <= num_params <= 5200, f"Param count {num_params} not in expected range"

    def test_num_params_matches_manual_count(self):
        """Test num_params property matches sum of all parameters."""
        model = PasswordCNN()
        
        # Manual count
        manual_count = sum(p.numel() for p in model.parameters())
        # Property count
        property_count = model.num_params
        
        assert manual_count == property_count

    def test_model_to_device_cpu(self):
        """Test model can be moved to CPU."""
        model = PasswordCNN()
        model_cpu = model.to("cpu")
        
        # Check all parameters are on CPU
        for param in model_cpu.parameters():
            assert param.device.type == "cpu"

    def test_model_to_device_cuda(self):
        """Test model can be moved to GPU (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = PasswordCNN()
        model_gpu = model.to("cuda")
        
        # Check all parameters are on GPU
        for param in model_gpu.parameters():
            assert param.device.type == "cuda"

    def test_forward_pass_differentiable(self):
        """Test that forward pass is differentiable (has gradients)."""
        model = PasswordCNN()
        x = torch.randn(4, 32, 8, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None

    def test_forward_pass_batch_independence(self):
        """Test that batch samples don't leak into each other."""
        model = PasswordCNN()
        model.eval()
        
        x1 = torch.randn(1, 32, 8)
        x2 = torch.randn(1, 32, 8)
        batch = torch.cat([x1, x2], dim=0)
        
        with torch.no_grad():
            out1_single = model(x1)
            out2_single = model(x2)
            out_batch = model(batch)
        
        # Outputs from single samples should match batch outputs
        assert torch.allclose(out1_single, out_batch[0:1], atol=1e-6)
        assert torch.allclose(out2_single, out_batch[1:2], atol=1e-6)


class TestPasswordCNNConfigurable:
    """Unit tests for PasswordCNNConfigurable model."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        model = PasswordCNNConfigurable()
        assert model is not None

    def test_init_custom_hidden_dims(self):
        """Test initialization with custom hidden dims."""
        model = PasswordCNNConfigurable(hidden_dims=[128, 64, 32])
        assert model is not None

    def test_init_single_hidden_layer(self):
        """Test initialization with single hidden layer."""
        model = PasswordCNNConfigurable(hidden_dims=[64])
        assert model is not None

    def test_init_four_hidden_layers(self):
        """Test initialization with four hidden layers."""
        model = PasswordCNNConfigurable(hidden_dims=[128, 96, 64, 32])
        assert model is not None

    def test_forward_pass_shape_two_layers(self):
        """Test forward pass with 2-layer dense stack."""
        model = PasswordCNNConfigurable(hidden_dims=[64, 32])
        x = torch.randn(4, 32, 8)
        output = model(x)
        
        assert output.shape == (4, 1)

    def test_forward_pass_shape_four_layers(self):
        """Test forward pass with 4-layer dense stack."""
        model = PasswordCNNConfigurable(hidden_dims=[128, 96, 64, 32])
        x = torch.randn(4, 32, 8)
        output = model(x)
        
        assert output.shape == (4, 1)

    def test_num_params_increases_with_hidden_dims(self):
        """Test that adding more hidden layers increases param count."""
        model_2layers = PasswordCNNConfigurable(hidden_dims=[64, 32])
        model_4layers = PasswordCNNConfigurable(hidden_dims=[128, 96, 64, 32])
        
        params_2 = model_2layers.num_params
        params_4 = model_4layers.num_params
        
        # More layers = more params
        assert params_4 > params_2

    def test_forward_pass_differentiable_configurable(self):
        """Test that forward pass is differentiable."""
        model = PasswordCNNConfigurable(hidden_dims=[64, 32])
        x = torch.randn(4, 32, 8, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None


class TestPasswordCNNConfigurableDictHiddenDims:
    """Tests for hidden_dims as list[dict] and mixed list inputs."""

    def test_dict_hidden_dims_basic(self):
        """List of dicts is accepted and produces correct output shape."""
        from nn_models import PasswordCNNConfigurable
        model = PasswordCNNConfigurable(hidden_dims=[{"units": 64}, {"units": 32}])
        x = torch.randn(4, 32, 8)
        assert model(x).shape == (4, 1)

    def test_dict_hidden_dims_with_activation(self):
        """Dict with explicit activation key is applied correctly."""
        from nn_models import PasswordCNNConfigurable
        model = PasswordCNNConfigurable(hidden_dims=[{"units": 64, "activation": "tanh"}])
        x = torch.randn(4, 32, 8)
        assert model(x).shape == (4, 1)

    def test_dict_hidden_dims_all_activations(self):
        """All supported activation strings are accepted."""
        from nn_models import PasswordCNNConfigurable
        for act in ("relu", "tanh", "sigmoid", "leaky_relu", "elu"):
            model = PasswordCNNConfigurable(hidden_dims=[{"units": 32, "activation": act}])
            x = torch.randn(2, 32, 8)
            assert model(x).shape == (2, 1), f"Failed for activation={act}"

    def test_dict_hidden_dims_unknown_activation_raises(self):
        """Unknown activation string raises ValueError with clear message."""
        from nn_models import PasswordCNNConfigurable
        with pytest.raises(ValueError, match="swish"):
            PasswordCNNConfigurable(hidden_dims=[{"units": 32, "activation": "swish"}])

    def test_dict_hidden_dims_per_layer_dropout(self):
        """Dict entry with 'dropout' overrides class-level dropout for that layer."""
        from nn_models import PasswordCNNConfigurable
        model = PasswordCNNConfigurable(
            hidden_dims=[{"units": 64, "dropout": 0.5}, {"units": 32}],
            dropout=0.1,
        )
        x = torch.randn(4, 32, 8)
        assert model(x).shape == (4, 1)

    def test_mixed_list_int_and_dict(self):
        """Mixed list of ints and dicts is coerced correctly."""
        from nn_models import PasswordCNNConfigurable
        model = PasswordCNNConfigurable(hidden_dims=[64, {"units": 32, "activation": "elu"}])
        x = torch.randn(4, 32, 8)
        assert model(x).shape == (4, 1)

    def test_int_list_backward_compat(self):
        """Plain int list still works (backward compat)."""
        from nn_models import PasswordCNNConfigurable
        model = PasswordCNNConfigurable(hidden_dims=[128, 64])
        x = torch.randn(4, 32, 8)
        assert model(x).shape == (4, 1)

    def test_dict_missing_units_raises(self):
        """Dict entry without 'units' key raises ValueError."""
        from nn_models import PasswordCNNConfigurable
        with pytest.raises((ValueError, KeyError)):
            PasswordCNNConfigurable(hidden_dims=[{"activation": "relu"}])


class TestBuildModelFromSpec:
    """Tests for module-level build_model_from_spec function."""

    def test_basic_spec_returns_model(self):
        from nn_models import build_model_from_spec, PasswordCNNConfigurable
        model = build_model_from_spec([{"units": 64}, {"units": 32}])
        assert isinstance(model, PasswordCNNConfigurable)

    def test_int_entries_normalized(self):
        """Int entries in spec are coerced to dict."""
        from nn_models import build_model_from_spec, PasswordCNNConfigurable
        model = build_model_from_spec([64, 32])
        assert isinstance(model, PasswordCNNConfigurable)
        x = torch.randn(4, 32, 8)
        assert model(x).shape == (4, 1)

    def test_empty_spec_raises(self):
        from nn_models import build_model_from_spec
        with pytest.raises(ValueError, match="non-empty"):
            build_model_from_spec([])

    def test_non_list_spec_raises(self):
        from nn_models import build_model_from_spec
        with pytest.raises(ValueError):
            build_model_from_spec(None)

    def test_zero_units_raises(self):
        from nn_models import build_model_from_spec
        with pytest.raises(ValueError, match="units"):
            build_model_from_spec([{"units": 0}])

    def test_negative_units_raises(self):
        from nn_models import build_model_from_spec
        with pytest.raises(ValueError, match="units"):
            build_model_from_spec([{"units": -8}])

    def test_custom_dropout_applied(self):
        from nn_models import build_model_from_spec
        model = build_model_from_spec([{"units": 32}], dropout=0.5)
        assert model(torch.randn(2, 32, 8)).shape == (2, 1)

    def test_device_cpu_moved(self):
        from nn_models import build_model_from_spec
        import torch
        model = build_model_from_spec([{"units": 32}], device=torch.device("cpu"))
        for p in model.parameters():
            assert p.device.type == "cpu"


class TestValidateLayerSpec:
    """Tests for module-level validate_layer_spec function."""

    def test_valid_dict_spec(self):
        from nn_models import validate_layer_spec
        ok, msg = validate_layer_spec([{"units": 64}, {"units": 32}])
        assert ok is True
        assert msg == ""

    def test_valid_int_spec(self):
        from nn_models import validate_layer_spec
        ok, msg = validate_layer_spec([64, 32])
        assert ok is True

    def test_empty_list_invalid(self):
        from nn_models import validate_layer_spec
        ok, msg = validate_layer_spec([])
        assert ok is False
        assert msg != ""

    def test_none_invalid(self):
        from nn_models import validate_layer_spec
        ok, msg = validate_layer_spec(None)
        assert ok is False

    def test_zero_units_invalid(self):
        from nn_models import validate_layer_spec
        ok, msg = validate_layer_spec([{"units": 0}])
        assert ok is False
        assert "units" in msg.lower()

    def test_negative_units_invalid(self):
        from nn_models import validate_layer_spec
        ok, msg = validate_layer_spec([{"units": -1}])
        assert ok is False

    def test_does_not_raise(self):
        """validate_layer_spec must never raise — always returns tuple."""
        from nn_models import validate_layer_spec
        result = validate_layer_spec("not a list")
        assert isinstance(result, tuple)
        assert result[0] is False

    def test_unknown_activation_invalid(self):
        from nn_models import validate_layer_spec
        ok, msg = validate_layer_spec([{"units": 32, "activation": "swish"}])
        assert ok is False
        assert "swish" in msg
