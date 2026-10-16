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
class TestGateway1Config:
    def test_default_config(self):
        from rafnet.fusion.gateway1 import Gateway1Config
        config = Gateway1Config()
        
        assert config.audio_dim == 256
        assert config.network_dim == 256
        assert config.hidden_dim == 128
        assert config.gate_type == "sigmoid"
        assert config.reliability_aware is True
        assert config.dropout == 0.1
        assert config.num_layers == 2
        assert config.use_residual_fusion is False
        assert config.residual_weight == 0.1
        assert config.gate_bias_init == 0.0
        assert config.log_gate_stats is True
        assert config.ablation_mode is None
    
    def test_custom_config(self):
        from rafnet.fusion.gateway1 import Gateway1Config
        config = Gateway1Config(
            audio_dim=128,
            network_dim=128,
            hidden_dim=64,
            gate_type="softmax",
            num_layers=3,
            use_residual_fusion=True,
            residual_weight=0.2,
            gate_bias_init=0.5,
            ablation_mode="fixed_equal",
        )
        
        assert config.audio_dim == 128
        assert config.network_dim == 128
        assert config.hidden_dim == 64
        assert config.gate_type == "softmax"
        assert config.num_layers == 3
        assert config.use_residual_fusion is True
        assert config.residual_weight == 0.2
        assert config.gate_bias_init == 0.5
        assert config.ablation_mode == "fixed_equal"
    
    def test_input_dim_property(self):
        from rafnet.fusion.gateway1 import Gateway1Config
        config = Gateway1Config(audio_dim=256, network_dim=256, context_dim=32)
        assert config.input_dim == 256 + 256 + 2 * 32  # 640
        
        config2 = Gateway1Config(audio_dim=128, network_dim=128, context_dim=0)
        assert config2.input_dim == 256


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestReliabilityFeatureExtractor:
    def test_basic_extraction(self):
        from rafnet.fusion.gateway1 import ReliabilityFeatureExtractor
        
        extractor = ReliabilityFeatureExtractor(
            audio_dim=256,
            network_dim=256,
            hidden_dim=128,
            context_dim=0
        )
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        c_a, c_n = extractor(z_a, z_n)
        
        assert c_a.shape == (4, 1)
        assert c_n.shape == (4, 1)
        assert torch.all(c_a >= 0) and torch.all(c_a <= 1)
        assert torch.all(c_n >= 0) and torch.all(c_n <= 1)
    
    def test_context_dim_extraction(self):
        from rafnet.fusion.gateway1 import ReliabilityFeatureExtractor
        
        extractor = ReliabilityFeatureExtractor(
            audio_dim=256,
            network_dim=256,
            hidden_dim=128,
            context_dim=32
        )
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        c_a, c_n = extractor(z_a, z_n)
        
        assert c_a.shape == (4, 32)
        assert c_n.shape == (4, 32)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestReliabilityGate:
    def test_sigmoid_gate(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        
        config = Gateway1Config(gate_type="sigmoid")
        gate = ReliabilityGate(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        alpha, alpha_raw, info = gate(z_a, z_n)
        
        assert alpha.shape == (4, 1)
        assert alpha_raw.shape == (4, 1)
        assert torch.all(alpha >= 0) and torch.all(alpha <= 1)
        assert "alpha_raw_mean" in info
        assert "alpha_raw_std" in info
    
    def test_softmax_gate(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        
        config = Gateway1Config(
            gate_type="softmax",
            audio_dim=256,
            network_dim=256,
            context_dim=1,
        )
        gate = ReliabilityGate(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        c_a = torch.randn(4, 1)
        c_n = torch.randn(4, 1)
        
        alpha, alpha_raw, info = gate(z_a, z_n, c_a, c_n)
        
        assert alpha.shape == (4, 1)
        assert torch.all(alpha >= 0) and torch.all(alpha <= 1)
    
    def test_gate_with_context(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        
        config = Gateway1Config(context_dim=16)
        gate = ReliabilityGate(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        c_a = torch.randn(4, 16)
        c_n = torch.randn(4, 16)
        
        alpha, alpha_raw, info = gate(z_a, z_n, c_a, c_n)
        
        assert alpha.shape == (4, 1)
        assert torch.all(alpha >= 0) and torch.all(alpha <= 1)
    
    def test_gate_bias_init(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        
        # Test positive bias initialization
        config = Gateway1Config(gate_bias_init=5.0)
        gate = ReliabilityGate(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        alpha, alpha_raw, _ = gate(z_a, z_n)
        
        # With positive bias, gate should initially favor audio (alpha > 0.5)
        assert alpha.mean() > 0.5
    
    def test_gate_statistics_tracking(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        
        config = Gateway1Config(log_gate_stats=True)
        gate = ReliabilityGate(config)
        gate.train()
        
        for _ in range(5):
            z_a = torch.randn(4, 256)
            z_n = torch.randn(4, 256)
            gate(z_a, z_n)
        
        stats = gate.get_gate_statistics()
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "num_samples" in stats
        assert stats["num_samples"] > 0
    
    def test_gate_statistics_reset(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        
        config = Gateway1Config(log_gate_stats=True)
        gate = ReliabilityGate(config)
        gate.train()
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        gate(z_a, z_n)
        
        gate.reset_statistics()
        stats = gate.get_gate_statistics()
        
        assert stats["num_samples"] == 0
        assert stats["mean"] == 0.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway1AblationModes:
    def test_fixed_audio_ablation(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        
        config = Gateway1Config(ablation_mode="fixed_audio")
        gate = ReliabilityGate(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        alpha, alpha_raw, info = gate(z_a, z_n)
        
        assert torch.all(alpha == 1.0)
        assert info.get("ablation") == "fixed_audio"
    
    def test_fixed_network_ablation(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        
        config = Gateway1Config(ablation_mode="fixed_network")
        gate = ReliabilityGate(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        alpha, alpha_raw, info = gate(z_a, z_n)
        
        assert torch.all(alpha == 0.0)
        assert info.get("ablation") == "fixed_network"
    
    def test_fixed_equal_ablation(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        
        config = Gateway1Config(ablation_mode="fixed_equal")
        gate = ReliabilityGate(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        alpha, alpha_raw, info = gate(z_a, z_n)
        
        assert torch.all(alpha == 0.5)
        assert info.get("ablation") == "fixed_equal"
    
    def test_gateway1_fixed_audio_ablation(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(ablation_mode="fixed_audio")
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert torch.allclose(z_fused, z_a)
        assert torch.all(info["alpha"] == 1.0)
    
    def test_gateway1_fixed_network_ablation(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(ablation_mode="fixed_network")
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert torch.allclose(z_fused, z_n)
        assert torch.all(info["alpha"] == 0.0)
    
    def test_gateway1_fixed_equal_ablation(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(ablation_mode="fixed_equal")
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        expected = 0.5 * z_a + 0.5 * z_n
        assert torch.allclose(z_fused, expected)
        assert torch.all(info["alpha"] == 0.5)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway1ResidualFusion:
    def test_residual_fusion_enabled(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(
            use_residual_fusion=True,
            residual_weight=0.1,
        )
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert "residual_norm" in info
        assert "residual_weight" in info
        assert info["residual_weight"] == 0.1
        assert z_fused.shape == (4, 256)
    
    def test_residual_fusion_disabled(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(use_residual_fusion=False)
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert "residual_norm" not in info
        assert z_fused.shape == (4, 256)
    
    def test_residual_fusion_gradient_flow(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(use_residual_fusion=True, residual_weight=0.1)
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256, requires_grad=True)
        z_n = torch.randn(4, 256, requires_grad=True)
        
        z_fused, _ = gateway(z_a, z_n)
        loss = z_fused.sum()
        loss.backward()
        
        assert z_a.grad is not None
        assert z_n.grad is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway1ReliabilityHeads:
    def test_reliability_aware_true(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(reliability_aware=True)
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert "reliability_audio" in info
        assert "reliability_network" in info
        assert info["reliability_audio"].shape == (4, 1)
        assert info["reliability_network"].shape == (4, 1)
        assert torch.all(info["reliability_audio"] >= 0) and torch.all(info["reliability_audio"] <= 1)
        assert torch.all(info["reliability_network"] >= 0) and torch.all(info["reliability_network"] <= 1)
    
    def test_reliability_aware_false(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(reliability_aware=False)
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert "reliability_audio" not in info
        assert "reliability_network" not in info
    
    def test_external_context(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(context_dim=32)
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        c_a = torch.randn(4, 32)
        c_n = torch.randn(4, 32)
        
        z_fused, info = gateway(z_a, z_n, c_a, c_n)
        
        assert info["c_a"].shape == (4, 32)
        assert info["c_n"].shape == (4, 32)
        assert torch.allclose(info["c_a"], c_a)
        assert torch.allclose(info["c_n"], c_n)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway1ReturnOptions:
    def test_return_gate(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config()
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info, alpha = gateway(z_a, z_n, return_gate=True)
        
        assert alpha.shape == (4, 1)
        assert torch.allclose(info["alpha"], alpha)
    
    def test_return_components(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config()
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n, return_components=True)
        
        assert "z_a_weighted" in info
        assert "z_n_weighted" in info
        assert "z_a" in info
        assert "z_n" in info
        assert info["z_a_weighted"].shape == (4, 256)
        assert info["z_n_weighted"].shape == (4, 256)
    
    def test_gate_statistics_methods(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(log_gate_stats=True)
        gateway = Gateway1(config)
        gateway.train()
        
        for _ in range(3):
            z_a = torch.randn(4, 256)
            z_n = torch.randn(4, 256)
            gateway(z_a, z_n)
        
        stats = gateway.get_gate_statistics()
        assert stats["num_samples"] > 0
        
        gateway.reset_gate_statistics()
        stats = gateway.get_gate_statistics()
        assert stats["num_samples"] == 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway1Forward:
    def test_basic_forward(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config()
        gateway = Gateway1(config)
        
        z_a = torch.randn(2, 256)
        z_n = torch.randn(2, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert z_fused.shape == (2, 256)
        assert "alpha" in info
        assert info["alpha"].shape == (2, 1)
        assert torch.all(info["alpha"] >= 0) and torch.all(info["alpha"] <= 1)
    
    def test_forward_with_context(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(context_dim=16)
        gateway = Gateway1(config)
        
        z_a = torch.randn(2, 256)
        z_n = torch.randn(2, 256)
        c_a = torch.randn(2, 16)
        c_n = torch.randn(2, 16)
        
        z_fused, info = gateway(z_a, z_n, c_a, c_n)
        
        assert z_fused.shape == (2, 256)
    
    def test_gate_values_in_range(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config()
        gateway = Gateway1(config)
        
        for _ in range(10):
            z_a = torch.randn(8, 256)
            z_n = torch.randn(8, 256)
            
            z_fused, info = gateway(z_a, z_n)
            
            assert torch.all(info["alpha"] >= 0)
            assert torch.all(info["alpha"] <= 1)
    
    def test_different_batch_sizes(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config()
        gateway = Gateway1(config)
        
        for batch_size in [1, 2, 8, 16, 32]:
            z_a = torch.randn(batch_size, 256)
            z_n = torch.randn(batch_size, 256)
            
            z_fused, info = gateway(z_a, z_n)
            
            assert z_fused.shape == (batch_size, 256)
            assert info["alpha"].shape == (batch_size, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestFixedFusionGateway1:
    def test_audio_only_mode(self):
        from rafnet.fusion.gateway1 import FixedFusionGateway1
        
        gateway = FixedFusionGateway1(mode="audio_only")
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert torch.allclose(z_fused, z_a)
        assert torch.all(info["alpha"] == 1.0)
        assert info["ablation_mode"] == "audio_only"
    
    def test_network_only_mode(self):
        from rafnet.fusion.gateway1 import FixedFusionGateway1
        
        gateway = FixedFusionGateway1(mode="network_only")
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert torch.allclose(z_fused, z_n)
        assert torch.all(info["alpha"] == 0.0)
        assert info["ablation_mode"] == "network_only"
    
    def test_equal_mode(self):
        from rafnet.fusion.gateway1 import FixedFusionGateway1
        
        gateway = FixedFusionGateway1(mode="equal")
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        expected = 0.5 * z_a + 0.5 * z_n
        assert torch.allclose(z_fused, expected)
        assert torch.all(info["alpha"] == 0.5)
        assert info["ablation_mode"] == "equal"
    
    def test_concat_mlp_mode(self):
        from rafnet.fusion.gateway1 import FixedFusionGateway1
        
        gateway = FixedFusionGateway1(mode="concat_mlp")
        
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert z_fused.shape == (4, 256)
        assert info["ablation_mode"] == "concat_mlp"
    
    def test_invalid_mode_raises(self):
        from rafnet.fusion.gateway1 import FixedFusionGateway1
        
        with pytest.raises(ValueError):
            FixedFusionGateway1(mode="invalid_mode")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway1GradientFlow:
    def test_gradient_to_audio_encoder(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config()
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256, requires_grad=True)
        z_n = torch.randn(4, 256, requires_grad=True)
        
        z_fused, _ = gateway(z_a, z_n)
        loss = z_fused.sum()
        loss.backward()
        
        assert z_a.grad is not None
        assert z_n.grad is not None
    
    def test_gradient_to_gate_network(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config()
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256, requires_grad=True)
        z_n = torch.randn(4, 256, requires_grad=True)
        
        z_fused, _ = gateway(z_a, z_n)
        loss = z_fused.sum()
        loss.backward()
        
        # Check gate network has gradients
        for param in gateway.reliability_gate.gate_network.parameters():
            assert param.grad is not None
    
    def test_gradient_to_reliability_extractor(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(context_dim=16)
        gateway = Gateway1(config)
        
        z_a = torch.randn(4, 256, requires_grad=True)
        z_n = torch.randn(4, 256, requires_grad=True)
        
        z_fused, _ = gateway(z_a, z_n)
        loss = z_fused.sum()
        loss.backward()
        
        # Check reliability extractor has gradients
        for param in gateway.reliability_extractor.parameters():
            assert param.grad is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway1GateDistribution:
    def test_gate_histogram_collection(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        
        config = Gateway1Config(log_gate_stats=True)
        gate = ReliabilityGate(config)
        gate.train()
        
        # Feed many samples
        for _ in range(20):
            z_a = torch.randn(10, 256)
            z_n = torch.randn(10, 256)
            _, _, info = gate(z_a, z_n)
        
        # Check histogram was collected
        if "gate_histogram" in info:
            hist = info["gate_histogram"]
            assert isinstance(hist, list)
            assert len(hist) == 10
            assert sum(hist) > 0
    
    def test_gate_distribution_properties(self):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config(log_gate_stats=True)
        gateway = Gateway1(config)
        gateway.train()
        
        # Feed samples with different characteristics
        for i in range(10):
            z_a = torch.randn(8, 256) * (i + 1) * 0.5
            z_n = torch.randn(8, 256) * (10 - i) * 0.5
            gateway(z_a, z_n)
        
        stats = gateway.get_gate_statistics()
        
        assert 0 <= stats["mean"] <= 1
        assert 0 <= stats["std"] <= 1
        assert 0 <= stats["min"] <= 1
        assert 0 <= stats["max"] <= 1


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway1GateBehavior:
    def test_gate_responds_to_input_changes(self):
        """Test that gate output changes with different inputs (sanity check)."""
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        
        config = Gateway1Config()
        gateway = Gateway1(config)
        gateway.eval()
        
        # Same inputs should give same gate values
        z_a1 = torch.randn(4, 256)
        z_n1 = torch.randn(4, 256)
        
        _, info1 = gateway(z_a1, z_n1)
        _, info2 = gateway(z_a1, z_n1)
        
        assert torch.allclose(info1["alpha"], info2["alpha"])
        
        # Different inputs should give different gate values (with high probability)
        z_a2 = torch.randn(4, 256)
        z_n2 = torch.randn(4, 256)
        
        _, info3 = gateway(z_a2, z_n2)
        
        # Not guaranteed but very likely different
        diff = (info1["alpha"] - info3["alpha"]).abs().mean().item()
        assert diff > 1e-5  # Should be different


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway1IntegrationWithModel:
    def test_gateway1_in_model(self):
        from rafnet.config.schema import RAFNetConfig, AudioConfig, NetworkConfig, Gateway1Config, ClassicalConfig
        from rafnet.model import RAFNet
        
        config = RAFNetConfig(
            device="cpu",
            audio=AudioConfig(wavlm_freeze=True, use_handcrafted=False, projection_dim=64),
            network=NetworkConfig(input_dim=32, output_dim=64, hidden_dims=[64]),
            gateway1=Gateway1Config(
                audio_dim=64,
                network_dim=64,
                hidden_dim=32,
                num_layers=2,
                use_residual_fusion=True,
            ),
            classical=ClassicalConfig(model_type="logistic"),
        )
        
        model = RAFNet(config)
        
        audio = torch.randn(2, 16000)
        network = torch.randn(2, 32)
        
        output = model(audio, network)
        
        assert output.z_fused is not None
        assert output.alpha is not None
        assert output.gate1_info is not None
        assert "reliability_audio" in output.gate1_info
        assert "reliability_network" in output.gate1_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])