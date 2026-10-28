import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2Config:
    def test_default_config(self):
        from rafnet.fusion.gateway2 import Gateway2Config
        config = Gateway2Config()
        
        assert config.input_dim == 3
        assert config.hidden_dim == 32
        assert config.gate_type == "sigmoid"
        assert config.dropout == 0.1
        assert config.num_layers == 2
        assert config.gate_bias_init == 0.0
        assert config.log_gate_stats is True
        assert config.enable_classical is True
        assert config.enable_quantum is True
        assert config.ablation_mode is None
    
    def test_custom_config(self):
        from rafnet.fusion.gateway2 import Gateway2Config
        config = Gateway2Config(
            input_dim=3,
            hidden_dim=64,
            gate_type="softmax",
            num_layers=3,
            gate_bias_init=0.5,
            log_gate_stats=False,
            enable_classical=False,
            enable_quantum=True,
            ablation_mode="fixed_equal",
        )
        
        assert config.input_dim == 3
        assert config.hidden_dim == 64
        assert config.gate_type == "softmax"
        assert config.num_layers == 3
        assert config.gate_bias_init == 0.5
        assert config.log_gate_stats is False
        assert config.enable_classical is False
        assert config.enable_quantum is True
        assert config.ablation_mode == "fixed_equal"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2Forward:
    def test_basic_forward(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.2, 0.8])
        p_q = torch.tensor([0.3, 0.7])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert p_fused.shape == (2,)
        assert "beta" in info
        assert info["beta"].shape == (2,)
        assert "disagreement" in info
        assert info["disagreement"].shape == (2,)
        assert "p_c" in info
        assert "p_q" in info
        assert torch.all(info["beta"] >= 0) and torch.all(info["beta"] <= 1)
        assert torch.all(info["p_c"] >= 0) and torch.all(info["p_c"] <= 1)
        assert torch.all(info["p_q"] >= 0) and torch.all(info["p_q"] <= 1)
    
    def test_forward_2d_inputs(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([[0.2], [0.8]])
        p_q = torch.tensor([[0.3], [0.7]])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert p_fused.shape == (2,)
        assert info["beta"].shape == (2,)
    
    def test_return_gate(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.2, 0.8])
        p_q = torch.tensor([0.3, 0.7])
        
        p_fused, info, beta = gateway(p_c, p_q, return_gate=True)
        
        assert beta.shape == (2, 1)
        assert torch.allclose(info["beta"], beta.squeeze(-1))
    
    def test_gate_values_in_range(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        for _ in range(100):
            p_c = torch.rand(10)
            p_q = torch.rand(10)
            
            p_fused, info = gateway(p_c, p_q)
            
            assert torch.all(info["beta"] >= 0)
            assert torch.all(info["beta"] <= 1)
            assert torch.all(info["p_c"] >= 0) and torch.all(info["p_c"] <= 1)
            assert torch.all(info["p_q"] >= 0) and torch.all(info["p_q"] <= 1)
            assert torch.all(info["disagreement"] >= 0)
            assert torch.all(info["disagreement"] <= 1)
    
    def test_different_batch_sizes(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        for batch_size in [1, 2, 8, 16, 32, 64]:
            p_c = torch.rand(batch_size)
            p_q = torch.rand(batch_size)
            
            p_fused, info = gateway(p_c, p_q)
            
            assert p_fused.shape == (batch_size,)
            assert info["beta"].shape == (batch_size,)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2BranchControl:
    def test_disable_classical_branch(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config(enable_classical=False, enable_quantum=True)
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.9, 0.1])
        p_q = torch.tensor([0.2, 0.8])
        
        p_fused, info = gateway(p_c, p_q)
        
        # Should trust quantum when classical disabled
        assert torch.allclose(p_fused, p_q)
        assert torch.all(info["beta"] == 0.0)
        assert info.get("classical_disabled") is True
    
    def test_disable_quantum_branch(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config(enable_classical=True, enable_quantum=False)
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.9, 0.1])
        p_q = torch.tensor([0.2, 0.8])
        
        p_fused, info = gateway(p_c, p_q)
        
        # Should trust classical when quantum disabled
        assert torch.allclose(p_fused, p_c)
        assert torch.all(info["beta"] == 1.0)
        assert info.get("quantum_disabled") is True
    
    def test_enable_disable_methods(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.9, 0.1])
        p_q = torch.tensor([0.2, 0.8])
        
        # Initially both enabled
        p_fused1, _ = gateway(p_c, p_q)
        
        # Disable classical
        gateway.enable_classical_branch(False)
        p_fused2, info2 = gateway(p_c, p_q)
        assert torch.allclose(p_fused2, p_q)
        assert torch.all(info2["beta"] == 0.0)
        
        # Re-enable classical, disable quantum
        gateway.enable_classical_branch(True)
        gateway.enable_quantum_branch(False)
        p_fused3, info3 = gateway(p_c, p_q)
        assert torch.allclose(p_fused3, p_c)
        assert torch.all(info3["beta"] == 1.0)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2AblationModes:
    def test_fixed_classical_ablation(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config(ablation_mode="fixed_classical")
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.2, 0.8])
        p_q = torch.tensor([0.9, 0.1])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert torch.allclose(p_fused, p_c)
        assert torch.all(info["beta"] == 1.0)
        assert info.get("ablation") == "fixed_classical"
    
    def test_fixed_quantum_ablation(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config(ablation_mode="fixed_quantum")
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.2, 0.8])
        p_q = torch.tensor([0.9, 0.1])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert torch.allclose(p_fused, p_q)
        assert torch.all(info["beta"] == 0.0)
        assert info.get("ablation") == "fixed_quantum"
    
    def test_fixed_equal_ablation(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config(ablation_mode="fixed_equal")
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.2, 0.8])
        p_q = torch.tensor([0.9, 0.1])
        
        p_fused, info = gateway(p_c, p_q)
        
        expected = 0.5 * (p_c + p_q)
        assert torch.allclose(p_fused, expected)
        assert torch.all(info["beta"] == 0.5)
        assert info.get("ablation") == "fixed_equal"
    
    def test_set_ablation_mode_method(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.2, 0.8])
        p_q = torch.tensor([0.9, 0.1])
        
        # Test switching modes
        gateway.set_ablation_mode("fixed_classical")
        p_fused1, info1 = gateway(p_c, p_q)
        assert torch.allclose(p_fused1, p_c)
        
        gateway.set_ablation_mode("fixed_quantum")
        p_fused2, info2 = gateway(p_c, p_q)
        assert torch.allclose(p_fused2, p_q)
        
        gateway.set_ablation_mode(None)
        p_fused3, info3 = gateway(p_c, p_q)
        # Should be learnable now
        assert not torch.allclose(p_fused3, p_c) or not torch.allclose(p_fused3, p_q)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2BoundaryValues:
    def test_p_c_zero(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.0, 0.0])
        p_q = torch.tensor([0.5, 0.8])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert not torch.isnan(p_fused).any()
        assert torch.all(info["p_c"] == 0.0)
        assert torch.all(info["disagreement"] == p_q)
    
    def test_p_c_one(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([1.0, 1.0])
        p_q = torch.tensor([0.2, 0.5])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert not torch.isnan(p_fused).any()
        assert torch.all(info["p_c"] == 1.0)
        assert torch.all(info["disagreement"] == 1.0 - p_q)
    
    def test_p_q_zero(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.3, 0.7])
        p_q = torch.tensor([0.0, 0.0])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert not torch.isnan(p_fused).any()
        assert torch.all(info["p_q"] == 0.0)
        assert torch.all(info["disagreement"] == p_c)
    
    def test_p_q_one(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.3, 0.7])
        p_q = torch.tensor([1.0, 1.0])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert not torch.isnan(p_fused).any()
        assert torch.all(info["p_q"] == 1.0)
        assert torch.all(info["disagreement"] == 1.0 - p_c)
    
    def test_both_zero(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([0.0, 0.0])
        p_q = torch.tensor([0.0, 0.0])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert not torch.isnan(p_fused).any()
        assert torch.all(p_fused == 0.0)
        assert torch.all(info["disagreement"] == 0.0)
    
    def test_both_one(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([1.0, 1.0])
        p_q = torch.tensor([1.0, 1.0])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert not torch.isnan(p_fused).any()
        assert torch.all(p_fused == 1.0)
        assert torch.all(info["disagreement"] == 0.0)
    
    def test_out_of_bounds_clamped(self):
        """Test that out-of-bounds inputs are clamped."""
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.tensor([-0.5, 1.5])
        p_q = torch.tensor([1.5, -0.5])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert not torch.isnan(p_fused).any()
        assert torch.all(info["p_c"] >= 0) and torch.all(info["p_c"] <= 1)
        assert torch.all(info["p_q"] >= 0) and torch.all(info["p_q"] <= 1)
        assert torch.all(info["p_c"] == torch.tensor([0.0, 1.0]))
        assert torch.all(info["p_q"] == torch.tensor([1.0, 0.0]))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2ExtremeDisagreement:
    def test_high_disagreement(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        # Extreme disagreement
        p_c = torch.tensor([0.05, 0.95])
        p_q = torch.tensor([0.95, 0.05])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert info["disagreement"][0] > 0.85
        assert info["disagreement"][1] > 0.85
    
    def test_identical_predictions(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        # Identical predictions
        p_c = torch.tensor([0.3, 0.7, 0.5])
        p_q = torch.tensor([0.3, 0.7, 0.5])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert torch.all(info["disagreement"] == 0.0)
        # When identical, any beta gives same result
        assert torch.allclose(p_fused, p_c)
    
    def test_moderate_disagreement(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        # Moderate disagreement
        p_c = torch.tensor([0.4, 0.6])
        p_q = torch.tensor([0.6, 0.4])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert torch.allclose(info["disagreement"], torch.tensor([0.2, 0.2]))
        # Fused should be between them
        assert torch.all(p_fused >= torch.minimum(p_c, p_q))
        assert torch.all(p_fused <= torch.maximum(p_c, p_q))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2Statistics:
    def test_gate_statistics_tracking(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config(log_gate_stats=True)
        gateway = Gateway2(config)
        gateway.train()
        
        for _ in range(10):
            p_c = torch.rand(8)
            p_q = torch.rand(8)
            gateway(p_c, p_q)
        
        stats = gateway.get_gate_statistics()
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "num_samples" in stats
        assert stats["num_samples"] > 0
    
    def test_gate_statistics_reset(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config(log_gate_stats=True)
        gateway = Gateway2(config)
        gateway.train()
        
        p_c = torch.rand(8)
        p_q = torch.rand(8)
        gateway(p_c, p_q)
        
        gateway.reset_statistics()
        stats = gateway.get_gate_statistics()
        
        assert stats["num_samples"] == 0
    
    def test_conditional_statistics(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config(log_gate_stats=True)
        gateway = Gateway2(config)
        gateway.train()
        
        # Feed cases where classical is more confident
        for _ in range(5):
            p_c = torch.tensor([0.9, 0.1])
            p_q = torch.tensor([0.2, 0.8])
            gateway(p_c, p_q)
        
        # Feed cases where quantum is more confident
        for _ in range(5):
            p_c = torch.tensor([0.2, 0.8])
            p_q = torch.tensor([0.9, 0.1])
            gateway(p_c, p_q)
        
        stats = gateway.get_gate_statistics()
        
        assert "beta_when_c_confident_mean" in stats
        assert "beta_when_q_confident_mean" in stats


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2GradientFlow:
    def test_gradient_to_gate_network(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.randn(4, requires_grad=True)
        p_q = torch.randn(4, requires_grad=True)
        
        p_fused, _ = gateway(p_c, p_q)
        loss = p_fused.sum()
        loss.backward()
        
        # Check gate network has gradients
        for param in gateway.gate_network.parameters():
            assert param.grad is not None
    
    def test_gradient_to_inputs(self):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        
        p_c = torch.randn(4, requires_grad=True)
        p_q = torch.randn(4, requires_grad=True)
        
        p_fused, _ = gateway(p_c, p_q)
        loss = p_fused.sum()
        loss.backward()
        
        assert p_c.grad is not None
        assert p_q.grad is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2GateBehavior:
    def test_gate_responds_to_inputs(self):
        """Test that gate output changes with different inputs."""
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        config = Gateway2Config()
        gateway = Gateway2(config)
        gateway.eval()
        
        # Same inputs should give same gate values
        p_c1 = torch.tensor([0.3, 0.7])
        p_q1 = torch.tensor([0.4, 0.6])
        
        _, info1 = gateway(p_c1, p_q1)
        _, info2 = gateway(p_c1, p_q1)
        
        assert torch.allclose(info1["beta"], info2["beta"])
        
        # Different inputs should give different gate values
        p_c2 = torch.tensor([0.8, 0.2])
        p_q2 = torch.tensor([0.1, 0.9])
        
        _, info3 = gateway(p_c2, p_q2)
        
        diff = (info1["beta"] - info3["beta"]).abs().mean().item()
        assert diff > 1e-5
    
    def test_beta_bias_initialization(self):
        """Test that gate_bias_init affects initial gate values."""
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        
        # Positive bias should favor classical
        config_pos = Gateway2Config(gate_bias_init=5.0)
        gateway_pos = Gateway2(config_pos)
        
        p_c = torch.randn(100)
        p_q = torch.randn(100)
        
        _, info_pos = gateway_pos(p_c, p_q)
        
        # Negative bias should favor quantum
        config_neg = Gateway2Config(gate_bias_init=-5.0)
        gateway_neg = Gateway2(config_neg)
        
        _, info_neg = gateway_neg(p_c, p_q)
        
        assert info_pos["beta"].mean() > info_neg["beta"].mean()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestFixedFusionGateway2:
    def test_classical_only_mode(self):
        from rafnet.fusion.gateway2 import FixedFusionGateway2
        
        gateway = FixedFusionGateway2(mode="classical_only")
        
        p_c = torch.tensor([0.3, 0.7])
        p_q = torch.tensor([0.9, 0.1])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert torch.allclose(p_fused, p_c)
        assert torch.all(info["beta"] == 1.0)
        assert info["ablation_mode"] == "classical_only"
    
    def test_quantum_only_mode(self):
        from rafnet.fusion.gateway2 import FixedFusionGateway2
        
        gateway = FixedFusionGateway2(mode="quantum_only")
        
        p_c = torch.tensor([0.3, 0.7])
        p_q = torch.tensor([0.9, 0.1])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert torch.allclose(p_fused, p_q)
        assert torch.all(info["beta"] == 0.0)
        assert info["ablation_mode"] == "quantum_only"
    
    def test_equal_mode(self):
        from rafnet.fusion.gateway2 import FixedFusionGateway2
        
        gateway = FixedFusionGateway2(mode="equal")
        
        p_c = torch.tensor([0.3, 0.7])
        p_q = torch.tensor([0.9, 0.1])
        
        p_fused, info = gateway(p_c, p_q)
        
        expected = 0.5 * (p_c + p_q)
        assert torch.allclose(p_fused, expected)
        assert torch.all(info["beta"] == 0.5)
        assert info["ablation_mode"] == "equal"
    
    def test_max_mode(self):
        from rafnet.fusion.gateway2 import FixedFusionGateway2
        
        gateway = FixedFusionGateway2(mode="max")
        
        p_c = torch.tensor([0.3, 0.7])
        p_q = torch.tensor([0.9, 0.1])
        
        p_fused, info = gateway(p_c, p_q)
        
        expected = torch.maximum(p_c, p_q)
        assert torch.allclose(p_fused, expected)
        assert info["ablation_mode"] == "max"
    
    def test_min_mode(self):
        from rafnet.fusion.gateway2 import FixedFusionGateway2
        
        gateway = FixedFusionGateway2(mode="min")
        
        p_c = torch.tensor([0.3, 0.7])
        p_q = torch.tensor([0.9, 0.1])
        
        p_fused, info = gateway(p_c, p_q)
        
        expected = torch.minimum(p_c, p_q)
        assert torch.allclose(p_fused, expected)
        assert info["ablation_mode"] == "min"
    
    def test_invalid_mode_raises(self):
        from rafnet.fusion.gateway2 import FixedFusionGateway2
        
        with pytest.raises(ValueError):
            FixedFusionGateway2(mode="invalid_mode")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2IntegrationWithModel:
    def test_gateway2_in_model(self):
        from rafnet.config.schema import RAFNetConfig, AudioConfig, NetworkConfig, Gateway1Config, Gateway2Config, ClassicalConfig
        from rafnet.model import RAFNet
        
        config = RAFNetConfig(
            device="cpu",
            audio=AudioConfig(wavlm_freeze=True, use_handcrafted=False, projection_dim=64),
            network=NetworkConfig(input_dim=32, output_dim=64, hidden_dims=[64]),
            gateway1=Gateway1Config(audio_dim=64, network_dim=64, hidden_dim=32),
            classical=ClassicalConfig(model_type="logistic"),
            gateway2=Gateway2Config(
                hidden_dim=32,
                num_layers=2,
                enable_classical=True,
                enable_quantum=True,
            ),
        )
        
        model = RAFNet(config)
        
        # Fit classical branch first
        z_fused_dummy = torch.randn(10, 64)
        labels_dummy = torch.randint(0, 2, (10,))
        model.fit_classical(z_fused_dummy, labels_dummy)
        
        # Fit quantum branch PCA
        model.fit_quantum(z_fused_dummy)
        
        audio = torch.randn(2, 16000)
        network = torch.randn(2, 32)
        
        output = model(audio, network)
        
        assert output.beta is not None
        assert output.gate2_info is not None
        assert "p_c" in output.gate2_info
        assert "p_q" in output.gate2_info
        assert "disagreement" in output.gate2_info
        # p_c and p_q should be tensors now (not None)
        assert output.gate2_info["p_c"] is not None
        assert output.gate2_info["p_q"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])