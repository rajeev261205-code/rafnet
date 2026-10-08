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

import numpy as np


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

import numpy as np


class TestDataModule:
    def test_generate_synthetic_data(self):
        from rafnet.data.dataset import generate_synthetic_data
        from rafnet.data.manifest import SampleManifest
        samples = generate_synthetic_data(n_samples=50, seed=42)
        
        assert len(samples) == 50
        assert all(isinstance(s, SampleManifest) for s in samples)
        assert all(hasattr(s, 'audio_path') for s in samples)
        assert all(hasattr(s, 'speaker_id') for s in samples)
        assert all(hasattr(s, 'class_label') for s in samples)
        assert all(s.class_label in [0, 1] for s in samples)
        assert all(len(s.speaker_id) > 0 for s in samples)
    
    def test_split_dataset_speaker_disjoint(self):
        from rafnet.data.dataset import generate_synthetic_data, split_dataset
        from rafnet.data.manifest import assign_splits, create_synthetic_manifest
        
        # Use manifest-based approach
        manifest = create_synthetic_manifest(n_samples=200, n_speakers=20, seed=42)
        manifest = assign_splits(manifest, strategy="speaker", seed=42)
        
        train = manifest.get_split("train").samples
        val = manifest.get_split("val").samples
        test = manifest.get_split("test").samples
        
        train_speakers = set(s.speaker_id for s in train)
        val_speakers = set(s.speaker_id for s in val)
        test_speakers = set(s.speaker_id for s in test)
        
        assert len(train_speakers & val_speakers) == 0
        assert len(train_speakers & test_speakers) == 0
        assert len(val_speakers & test_speakers) == 0
        assert len(train) + len(val) + len(test) == 200
    
    def test_split_dataset_random(self):
        from rafnet.data.dataset import generate_synthetic_data, split_dataset
        from rafnet.data.manifest import assign_splits, create_synthetic_manifest
        
        manifest = create_synthetic_manifest(n_samples=100, seed=42)
        manifest = assign_splits(manifest, strategy="random", seed=42)
        
        train = manifest.get_split("train").samples
        val = manifest.get_split("val").samples
        test = manifest.get_split("test").samples
        
        assert len(train) + len(val) + len(test) == 100
    
    def test_dataset_and_collate(self):
        from rafnet.data.dataset import SyntheticVoiceDataset, collate_fn, create_synthetic_fixture
        from rafnet.data.manifest import Manifest
        
        manifest = create_synthetic_fixture(n_samples=10, seed=42)
        dataset = SyntheticVoiceDataset(manifest, network_dim=20)
        
        assert len(dataset) == 10
        
        item = dataset[0]
        assert "audio" in item
        assert "network" in item
        assert "label" in item
        
        batch = collate_fn([dataset[i] for i in range(4)])
        assert batch["audio"].shape == (4, 16000)
        assert batch["network"].shape == (4, 20)
        assert batch["labels"].shape == (4,)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestAblation:
    def test_ablation_model_m1(self, config):
        from rafnet.evaluation.ablation import AblationModel
        import torch
        
        model = AblationModel("M1_audio_only", config)
        
        audio = torch.randn(2, 16000)
        network = torch.randn(2, 20)
        
        output = model(audio, network)
        
        assert output.shape == (2,)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_ablation_model_m2(self, config):
        from rafnet.evaluation.ablation import AblationModel
        import torch
        
        model = AblationModel("M2_network_only", config)
        
        audio = torch.randn(2, 16000)
        network = torch.randn(2, 20)
        
        output = model(audio, network)
        
        assert output.shape == (2,)
    
    def test_ablation_model_m3(self, config):
        from rafnet.evaluation.ablation import AblationModel
        import torch
        
        model = AblationModel("M3_fixed_fusion", config)
        
        audio = torch.randn(2, 16000)
        network = torch.randn(2, 20)
        
        output = model(audio, network)
        
        assert output.shape == (2,)
    
    def test_ablation_model_m4(self, config):
        from rafnet.evaluation.ablation import AblationModel
        import torch
        
        model = AblationModel("M4_gateway1", config)
        
        audio = torch.randn(2, 16000)
        network = torch.randn(2, 20)
        
        output = model(audio, network)
        
        assert output.shape == (2,)
    
    def test_ablation_model_invalid(self, config):
        from rafnet.evaluation.ablation import AblationModel
        
        with pytest.raises(ValueError):
            AblationModel("INVALID_MODEL", config)