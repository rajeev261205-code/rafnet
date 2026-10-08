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

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestConfig:
    def test_default_config(self):
        from rafnet.config.schema import RAFNetConfig
        config = RAFNetConfig()
        
        assert config.audio.sample_rate == 16000
        assert config.network.input_dim == 32
        assert config.loss.ce_weight == 1.0
        assert config.training.batch_size == 16
    
    def test_config_from_yaml(self, tmp_path):
        from rafnet.config.schema import RAFNetConfig
        config = RAFNetConfig()
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(str(yaml_path))
        
        loaded = RAFNetConfig.from_yaml(str(yaml_path))
        
        assert loaded.audio.sample_rate == config.audio.sample_rate
        assert loaded.network.input_dim == config.network.input_dim
    
    def test_load_config_none(self):
        from rafnet.config.schema import load_config, RAFNetConfig
        config = load_config(None)
        assert isinstance(config, RAFNetConfig)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestUtils:
    def test_set_seed(self):
        from rafnet.utils import set_seed
        import torch
        import numpy as np
        
        set_seed(42)
        a = torch.randn(10)
        b = np.random.randn(10)
        
        set_seed(42)
        a2 = torch.randn(10)
        b2 = np.random.randn(10)
        
        assert torch.allclose(a, a2)
        assert np.allclose(b, b2)
    
    def test_get_device_auto(self):
        from rafnet.utils import get_device
        device = get_device("auto")
        assert isinstance(device, torch.device)
    
    def test_get_device_explicit(self):
        from rafnet.utils import get_device
        device = get_device("cpu")
        assert device == torch.device("cpu")
    
    def test_count_parameters(self):
        from rafnet.utils import count_parameters
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        params = count_parameters(model)
        assert params == 10 * 5 + 5
    
    def test_format_number(self):
        from rafnet.utils import format_number
        
        assert format_number(500) == "500"
        assert format_number(1500) == "1.50K"
        assert format_number(1_500_000) == "1.50M"
        assert format_number(1_500_000_000) == "1.50B"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestSmokeTest:
    def test_full_pipeline_smoke(self):
        from rafnet.config.schema import (
            RAFNetConfig, AudioConfig, NetworkConfig, Gateway1Config,
            ClassicalConfig, QuantumConfig, Gateway2Config, CalibrationConfig,
            DecisionGateConfig, LossConfig
        )
        from rafnet.model import RAFNet
        from rafnet.data.dataset import create_synthetic_fixture, create_datasets_from_manifest, collate_fn
        
        config = RAFNetConfig(
            device="cpu",
            audio=AudioConfig(wavlm_freeze=True, use_handcrafted=False, projection_dim=64),
            network=NetworkConfig(input_dim=20, output_dim=64, hidden_dims=[64]),
            gateway1=Gateway1Config(audio_dim=64, network_dim=64, hidden_dim=32),
            classical=ClassicalConfig(model_type="logistic"),
            quantum=QuantumConfig(n_qubits=4, n_layers=1, pca_dim=4, shots=None),
            gateway2=Gateway2Config(hidden_dim=16),
            calibration=CalibrationConfig(method="temperature"),
            decision_gate=DecisionGateConfig(optimize_thresholds=False),
            loss=LossConfig(ce_weight=1.0, rel_weight=0.0, selcal_weight=0.0, agree_weight=0.0),
        )
        
        model = RAFNet(config)
        model.eval()
        
        # Use the new dataset API
        manifest = create_synthetic_fixture(n_samples=20, seed=42)
        train_ds, val_ds, test_ds = create_datasets_from_manifest(manifest, max_samples_per_split=5)
        
        # Test forward pass on train dataset
        # Just test with a single batch
        batch = next(iter(torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)))
        
        with torch.no_grad():
            output = model(batch["audio"], batch["network"])
        
        assert output.probs.shape == (4,)
        assert output.decisions.shape == (4,)
        assert torch.all(output.probs >= 0) and torch.all(output.probs <= 1)
        assert torch.all(output.decisions >= 0) and torch.all(output.decisions <= 2)
        
        losses = model.compute_loss(output, batch["labels"].float())
        assert "total" in losses
        assert losses["total"].item() >= 0
        
        print(f"Smoke test passed! Loss: {losses['total'].item():.4f}")
        print(f"Decisions: REAL={(output.decisions==0).sum()}, UNCERTAIN={(output.decisions==1).sum()}, SYNTHETIC={(output.decisions==2).sum()}")