import pytest
import sys
sys.path.insert(0, '/home/sk/rafnet/src')

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestRAFNetModel:
    def test_model_init(self, config):
        from rafnet.model import RAFNet
        model = RAFNet(config)
        assert model is not None
    
    def test_model_forward(self, config, sample_audio, sample_network):
        from rafnet.model import RAFNet
        import torch
        
        model = RAFNet(config)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_audio, sample_network)
        
        from rafnet.model import RAFNetOutput
        assert isinstance(output, RAFNetOutput)
        assert output.probs.shape == (2,)
        assert output.decisions.shape == (2,)
        assert output.z_fused.shape == (2, config.audio.projection_dim)
        assert output.gate1_info is not None
        assert output.gate2_info is not None
        assert output.decision_info is not None
    
    def test_model_compute_loss(self, config, sample_audio, sample_network):
        from rafnet.model import RAFNet
        import torch
        
        model = RAFNet(config)
        
        output = model(sample_audio, sample_network)
        labels = torch.tensor([0.0, 1.0])
        
        losses = model.compute_loss(output, labels)
        
        assert "total" in losses
        assert "ce" in losses
        assert losses["total"].item() >= 0
    
    def test_model_fit_classical(self, config, sample_audio, sample_network):
        from rafnet.model import RAFNet
        import torch
        
        model = RAFNet(config)
        
        z_fused = torch.randn(10, config.audio.projection_dim)
        labels = torch.randint(0, 2, (10,)).float()
        
        model.fit_classical(z_fused, labels)
        
        assert model.classical_fitted
    
    def test_model_fit_quantum(self, config):
        from rafnet.model import RAFNet
        import torch
        
        model = RAFNet(config)
        
        z_fused = torch.randn(20, config.audio.projection_dim)
        
        model.fit_quantum(z_fused)
        
        assert model.quantum_fitted
    
    def test_model_fit_calibration(self, config):
        from rafnet.model import RAFNet
        import torch
        
        model = RAFNet(config)
        
        probs = torch.rand(40)
        labels = (probs > 0.5).float()
        
        # Split into train/val
        val_probs = probs[:20]
        val_labels = labels[:20]
        train_probs = probs[20:]
        train_labels = labels[20:]
        
        model.fit_calibration(train_probs, train_labels, val_probs, val_labels)
        
        assert model.calibration_fitted
    
    def test_model_optimize_thresholds(self, config):
        from rafnet.config.schema import RAFNetConfig, DecisionGateConfig
        from rafnet.model import RAFNet
        import torch
        
        # Create config with optimize_thresholds=True
        test_config = RAFNetConfig(
            device="cpu",
            audio=config.audio,
            network=config.network,
            gateway1=config.gateway1,
            classical=config.classical,
            quantum=config.quantum,
            gateway2=config.gateway2,
            calibration=config.calibration,
            decision_gate=DecisionGateConfig(optimize_thresholds=True, grid_resolution=10),
            loss=config.loss,
            training=config.training,
        )
        
        model = RAFNet(test_config)
        
        probs = torch.rand(100)
        labels = (probs > 0.5).long()
        
        # Split into train/val
        val_probs = probs[:50]
        val_labels = labels[:50]
        train_probs = probs[50:]
        train_labels = labels[50:]
        
        model.optimize_decision_thresholds(train_probs, train_labels, val_probs, val_labels)
        
        assert model.decision_gate.optimized
    
    def test_model_with_audio_mask(self, config):
        from rafnet.model import RAFNet
        import torch
        
        model = RAFNet(config)
        model.eval()
        
        audio = torch.randn(2, 16000)
        network = torch.randn(2, 20)
        audio_mask = torch.ones(2, 16000)
        audio_mask[1, 8000:] = 0
        
        with torch.no_grad():
            output = model(audio, network, audio_mask)
        
        assert output.probs.shape == (2,)