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
class TestGateway1:
    def test_gateway1_init(self, config):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        gateway_config = Gateway1Config(
            audio_dim=256,
            network_dim=256,
            hidden_dim=config.gateway1.hidden_dim,
        )
        gateway = Gateway1(gateway_config)
        assert gateway is not None
    
    def test_gateway1_forward(self, config):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        gateway_config = Gateway1Config(
            audio_dim=256,
            network_dim=256,
            hidden_dim=config.gateway1.hidden_dim,
        )
        gateway = Gateway1(gateway_config)
        
        z_a = torch.randn(2, 256)
        z_n = torch.randn(2, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert z_fused.shape == (2, 256)
        assert "alpha" in info
        assert info["alpha"].shape == (2, 1)
        assert torch.all(info["alpha"] >= 0) and torch.all(info["alpha"] <= 1)
    
    def test_gateway1_with_context(self, config):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        gateway_config = Gateway1Config(
            audio_dim=256,
            network_dim=256,
            context_dim=32,
            hidden_dim=config.gateway1.hidden_dim,
        )
        gateway = Gateway1(gateway_config)
        
        z_a = torch.randn(2, 256)
        z_n = torch.randn(2, 256)
        c_a = torch.randn(2, 32)
        c_n = torch.randn(2, 32)
        
        z_fused, info = gateway(z_a, z_n, c_a, c_n)
        
        assert z_fused.shape == (2, 256)
        assert "alpha" in info
    
    def test_gateway1_reliability_aware(self, config):
        from rafnet.fusion.gateway1 import Gateway1, Gateway1Config
        gateway_config = Gateway1Config(
            audio_dim=256,
            network_dim=256,
            hidden_dim=config.gateway1.hidden_dim,
            reliability_aware=True,
        )
        gateway = Gateway1(gateway_config)
        
        z_a = torch.randn(2, 256)
        z_n = torch.randn(2, 256)
        
        z_fused, info = gateway(z_a, z_n)
        
        assert "reliability_audio" in info
        assert "reliability_network" in info
        assert info["reliability_audio"].shape == (2, 1)
        assert info["reliability_network"].shape == (2, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestReliabilityGate:
    def test_reliability_gate_sigmoid(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        gate = ReliabilityGate(Gateway1Config(gate_type="sigmoid"))
        z_a = torch.randn(2, 256)
        z_n = torch.randn(2, 256)
        
        alpha, alpha_raw, info = gate(z_a, z_n)
        
        assert alpha.shape == (2, 1)
        assert torch.all(alpha >= 0) and torch.all(alpha <= 1)
        assert "alpha_raw_mean" in info
    
    def test_reliability_gate_softmax(self):
        from rafnet.fusion.gateway1 import ReliabilityGate, Gateway1Config
        gate = ReliabilityGate(Gateway1Config(
            gate_type="softmax",
            audio_dim=256,
            network_dim=256,
            context_dim=1,
        ))
        z_a = torch.randn(2, 256)
        z_n = torch.randn(2, 256)
        c_a = torch.randn(2, 1)
        c_n = torch.randn(2, 1)
        
        alpha, alpha_raw, info = gate(z_a, z_n, c_a, c_n)
        
        assert alpha.shape == (2, 1)
        assert torch.all(alpha >= 0) and torch.all(alpha <= 1)
        assert "alpha_raw_mean" in info


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestGateway2:
    def test_gateway2_init(self, config):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        gateway_config = Gateway2Config(
            input_dim=5,
            hidden_dim=config.gateway2.hidden_dim,
        )
        gateway = Gateway2(gateway_config)
        assert gateway is not None
    
    def test_gateway2_forward(self, config):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        gateway_config = Gateway2Config(
            input_dim=3,
            hidden_dim=config.gateway2.hidden_dim,
        )
        gateway = Gateway2(gateway_config)
        
        p_c = torch.tensor([0.2, 0.8])
        p_q = torch.tensor([0.3, 0.7])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert p_fused.shape == (2,)
        assert "beta" in info
        assert info["beta"].shape == (2,)
        assert torch.all(info["beta"] >= 0) and torch.all(info["beta"] <= 1)
        assert "disagreement" in info
        assert info["disagreement"].shape == (2,)
    
    def test_gateway2_disagreement(self, config):
        from rafnet.fusion.gateway2 import Gateway2, Gateway2Config
        gateway_config = Gateway2Config(input_dim=3, hidden_dim=32)
        gateway = Gateway2(gateway_config)
        
        p_c = torch.tensor([0.1, 0.9])
        p_q = torch.tensor([0.9, 0.1])
        
        p_fused, info = gateway(p_c, p_q)
        
        assert info["disagreement"][0] > 0.7
        assert info["disagreement"][1] > 0.7