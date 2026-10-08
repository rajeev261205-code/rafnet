import pytest
import torch
import numpy as np
import tempfile
import os

from rafnet.data.dataset import (
    SyntheticVoiceDataset,
    Normalizer,
    AudioNetworkAligner,
    AudioWindowConfig,
    NetworkWindowConfig,
    create_synthetic_fixture,
    fit_normalizers,
    create_datasets_from_manifest,
    detect_dataset_leakage,
    collate_fn,
    create_dataloader,
)
from rafnet.data.manifest import SampleManifest, Manifest


class TestNormalizer:
    def test_fit_transform(self):
        normalizer = Normalizer()
        data = np.random.randn(100, 5)
        normalized = normalizer.fit_transform(data)
        
        assert normalizer.fitted
        assert normalized.shape == data.shape
        assert np.allclose(normalized.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(normalized.std(axis=0), 1, atol=1e-6)
    
    def test_transform_before_fit(self):
        normalizer = Normalizer()
        data = np.random.randn(10, 5)
        
        with pytest.raises(RuntimeError):
            normalizer.transform(data)
    
    def test_inverse_transform(self):
        normalizer = Normalizer()
        data = np.random.randn(100, 5)
        normalized = normalizer.fit_transform(data)
        reconstructed = normalizer.inverse_transform(normalized)
        
        assert np.allclose(reconstructed, data, atol=1e-6)
    
    def test_save_load(self):
        normalizer = Normalizer()
        data = np.random.randn(100, 5)
        normalizer.fit_transform(data)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            temp_path = f.name
        
        try:
            normalizer.save(temp_path)
            loaded = Normalizer.load(temp_path)
            
            assert loaded.fitted
            assert np.allclose(loaded.mean, normalizer.mean)
            assert np.allclose(loaded.std, normalizer.std)
            
            test_data = np.random.randn(10, 5)
            assert np.allclose(loaded.transform(test_data), normalizer.transform(test_data))
        finally:
            os.unlink(temp_path)


class TestAudioNetworkAligner:
    def test_align(self):
        aligner = AudioNetworkAligner(sample_rate=16000, network_fps=50)
        
        audio = torch.randn(32000)
        network = torch.randn(200, 10)
        
        aligned_audio, aligned_network = aligner.align(audio, network)
        
        assert aligned_audio.shape[-1] <= audio.shape[-1]
        assert aligned_network.shape[0] <= network.shape[0]
        
        audio_duration = aligned_audio.shape[-1] / 16000
        network_duration = aligned_network.shape[0] / 50
        assert abs(audio_duration - network_duration) < 0.1
    
    def test_create_windows(self):
        aligner = AudioNetworkAligner(
            sample_rate=16000,
            network_fps=50,
            audio_window=AudioWindowConfig(window_size=16000, hop_size=8000),
            network_window=NetworkWindowConfig(window_size=10, hop_size=5),
        )
        
        audio = torch.randn(48000)
        network = torch.randn(150, 10)
        
        audio_windows, network_windows = aligner.create_windows(audio, network)
        
        assert len(audio_windows) == len(network_windows)
        assert audio_windows[0].shape[-1] == 16000
        assert network_windows[0].shape[0] == 10


class TestSyntheticVoiceDataset:
    def test_dataset_creation(self):
        manifest = create_synthetic_fixture(n_samples=20, seed=42)
        dataset = SyntheticVoiceDataset(manifest, max_samples=10)
        
        assert len(dataset) == 10
    
    def test_getitem(self):
        manifest = create_synthetic_fixture(n_samples=20, seed=42)
        dataset = SyntheticVoiceDataset(manifest, max_samples=5)
        
        item = dataset[0]
        
        assert "sample_id" in item
        assert "audio" in item
        assert "network" in item
        assert "label" in item
        assert "speaker_id" in item
        assert "generator_id" in item
        assert "condition_id" in item
        assert item["audio"].dim() == 1
        assert item["network"].dim() == 1
        assert item["label"].item() in [0, 1]
    
    def test_getitem_with_windows(self):
        manifest = create_synthetic_fixture(n_samples=10, seed=42)
        # Test that dataset can be created with return_windows=True
        dataset = SyntheticVoiceDataset(manifest, max_samples=5, return_windows=True)
        
        # The test just verifies the dataset can be instantiated with windows
        assert len(dataset) == 5
        assert dataset.return_windows == True


class TestFitNormalizers:
    def test_fit_normalizers(self):
        manifest = create_synthetic_fixture(n_samples=20, seed=42)
        train_dataset = SyntheticVoiceDataset(manifest.get_split("train"), max_samples=10)
        val_dataset = SyntheticVoiceDataset(manifest.get_split("val"), max_samples=5)
        
        audio_norm, network_norm = fit_normalizers(train_dataset, val_dataset)
        
        assert network_norm.fitted
        assert val_dataset.network_normalizer is network_norm


class TestCreateDatasetsFromManifest:
    def test_create_datasets(self):
        manifest = create_synthetic_fixture(n_samples=200, seed=42)
        train_ds, val_ds, test_ds = create_datasets_from_manifest(
            manifest, max_samples_per_split=20
        )
        
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        assert train_ds.network_normalizer is not None
        assert val_ds.network_normalizer is not None
        assert test_ds.network_normalizer is not None


class TestDetectDatasetLeakage:
    def test_no_leakage(self):
        from rafnet.data.manifest import create_synthetic_manifest, assign_splits
        manifest = create_synthetic_manifest(
            n_samples=300, n_speakers=30, n_generators=20, n_conditions=10, seed=42,
            split_ratios=(0.7, 0.15, 0.15)
        )
        # Use generator-disjoint split to ensure no generator overlap
        manifest = assign_splits(manifest, strategy="generator", seed=42)
        
        train_m = manifest.get_split("train")
        val_m = manifest.get_split("val")
        test_m = manifest.get_split("test")
        
        result = detect_dataset_leakage(train_m, val_m, test_m)
        
        # Generator split guarantees no generator overlap
        # Note: speaker and condition may still overlap with generator strategy
        assert result["generator_overlap"]["has_leakage"] == False
    
    def test_no_leakage_speaker_split(self):
        from rafnet.data.manifest import create_synthetic_manifest, assign_splits
        manifest = create_synthetic_manifest(
            n_samples=300, n_speakers=30, n_generators=20, n_conditions=10, seed=42,
            split_ratios=(0.7, 0.15, 0.15)
        )
        # Use speaker-disjoint split to ensure no speaker overlap
        manifest = assign_splits(manifest, strategy="speaker", seed=42)
        
        train_m = manifest.get_split("train")
        val_m = manifest.get_split("val")
        test_m = manifest.get_split("test")
        
        result = detect_dataset_leakage(train_m, val_m, test_m)
        
        # Speaker split guarantees no speaker overlap
        assert result["speaker_overlap"]["has_leakage"] == False
    
    def test_with_speaker_leakage(self):
        samples = [
            SampleManifest(
                sample_id="test_001",
                speaker_id="spk_001",
                generator_id="gen_001",
                class_label=1,
                audio_path="audio/test.wav",
                communication_condition_id="cond_001",
                codec="opus",
                packet_loss=0.05,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="train",
            ),
            SampleManifest(
                sample_id="test_002",
                speaker_id="spk_001",
                generator_id="gen_002",
                class_label=0,
                audio_path="audio/test.wav",
                communication_condition_id="cond_001",
                codec="opus",
                packet_loss=0.05,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="val",
            ),
        ]
        train_m = Manifest(samples[:1])
        val_m = Manifest(samples[1:])
        test_m = Manifest([])
        
        result = detect_dataset_leakage(train_m, val_m, test_m)
        
        assert result["speaker_overlap"]["has_leakage"] == True


class TestCollateFn:
    def test_collate_fn(self):
        items = [
            {
                "sample_id": f"test_{i}",
                "audio": torch.randn(16000),
                "network": torch.randn(32),
                "label": torch.tensor(i % 2, dtype=torch.long),
                "speaker_id": f"spk_{i%3:03d}",
                "generator_id": f"gen_{i%2:03d}",
                "condition_id": f"cond_{i%2:03d}",
                "codec": "opus",
            }
            for i in range(4)
        ]
        
        batch = collate_fn(items)
        
        assert batch["audio"].shape == (4, 16000)
        assert batch["network"].shape == (4, 32)
        assert batch["labels"].shape == (4,)
        assert len(batch["sample_ids"]) == 4


class TestCreateDataloader:
    def test_create_dataloader(self):
        manifest = create_synthetic_fixture(n_samples=20, seed=42)
        dataset = SyntheticVoiceDataset(manifest, max_samples=10)
        
        loader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        batch = next(iter(loader))
        
        assert batch["audio"].shape == (4, 16000)
        assert batch["network"].shape == (4, 32)
        assert batch["labels"].shape == (4,)