import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from rafnet.data.manifest import (
    SampleManifest,
    Manifest,
    create_synthetic_manifest,
    assign_splits,
    check_leakage,
    enforce_condition_class_balance,
    check_entity_overlap,
    check_condition_class_balance,
    check_network_feature_correlation,
)


class TestSampleManifest:
    def test_create_sample(self):
        sample = SampleManifest(
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
        )
        assert sample.sample_id == "test_001"
        assert sample.class_label == 1
    
    def test_to_dict(self):
        sample = SampleManifest(
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
        )
        d = sample.to_dict()
        assert d["sample_id"] == "test_001"
        assert d["class_label"] == 1
    
    def test_from_dict(self):
        d = {
            "sample_id": "test_001",
            "speaker_id": "spk_001",
            "generator_id": "gen_001",
            "class_label": 1,
            "audio_path": "audio/test.wav",
            "communication_condition_id": "cond_001",
            "codec": "opus",
            "packet_loss": 0.05,
            "jitter": 0.01,
            "rtt": 0.1,
            "bitrate": 64000,
            "split": "train",
        }
        sample = SampleManifest.from_dict(d)
        assert sample.sample_id == "test_001"


class TestManifest:
    def test_create_manifest(self):
        samples = [
            SampleManifest(
                sample_id=f"test_{i:03d}",
                speaker_id="spk_001",
                generator_id="gen_001",
                class_label=i % 2,
                audio_path=f"audio/test_{i}.wav",
                communication_condition_id="cond_001",
                codec="opus",
                packet_loss=0.05,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="train",
            )
            for i in range(10)
        ]
        manifest = Manifest(samples)
        assert len(manifest) == 10
    
    def test_validate_valid(self):
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
            )
        ]
        manifest = Manifest(samples)
        errors = manifest.validate()
        assert len(errors) == 0
    
    def test_validate_missing_fields(self):
        manifest = Manifest([])
        errors = manifest.validate()
        assert "general" in errors
    
    def test_validate_invalid_label(self):
        samples = [
            SampleManifest(
                sample_id="test_001",
                speaker_id="spk_001",
                generator_id="gen_001",
                class_label=2,
                audio_path="audio/test.wav",
                communication_condition_id="cond_001",
                codec="opus",
                packet_loss=0.05,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="train",
            )
        ]
        manifest = Manifest(samples)
        errors = manifest.validate()
        assert "invalid_labels" in errors
    
    def test_validate_invalid_split(self):
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
                split="invalid",
            )
        ]
        manifest = Manifest(samples)
        errors = manifest.validate()
        assert "invalid_splits" in errors
    
    def test_validate_packet_loss_bounds(self):
        samples = [
            SampleManifest(
                sample_id="test_001",
                speaker_id="spk_001",
                generator_id="gen_001",
                class_label=1,
                audio_path="audio/test.wav",
                communication_condition_id="cond_001",
                codec="opus",
                packet_loss=1.5,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="train",
            )
        ]
        manifest = Manifest(samples)
        errors = manifest.validate()
        assert "invalid_packet_loss" in errors
    
    def test_detect_duplicates(self):
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
                generator_id="gen_001",
                class_label=0,
                audio_path="audio/test.wav",
                communication_condition_id="cond_001",
                codec="opus",
                packet_loss=0.05,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="train",
            ),
        ]
        manifest = Manifest(samples)
        duplicates = manifest.detect_duplicates(["audio_path"])
        assert len(duplicates) == 1
        assert len(duplicates[0]) == 2
    
    def test_get_split(self):
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
                generator_id="gen_001",
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
        manifest = Manifest(samples)
        train = manifest.get_split("train")
        val = manifest.get_split("val")
        assert len(train) == 1
        assert len(val) == 1
    
    def test_speakers_generators_conditions(self):
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
                speaker_id="spk_002",
                generator_id="gen_002",
                class_label=0,
                audio_path="audio/test.wav",
                communication_condition_id="cond_002",
                codec="opus",
                packet_loss=0.05,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="train",
            ),
        ]
        manifest = Manifest(samples)
        assert len(manifest.speakers()) == 2
        assert len(manifest.generators()) == 2
        assert len(manifest.conditions()) == 2
    
    def test_class_distribution(self):
        samples = [
            SampleManifest(
                sample_id=f"test_{i:03d}",
                speaker_id="spk_001",
                generator_id="gen_001",
                class_label=i % 2,
                audio_path=f"audio/test_{i}.wav",
                communication_condition_id="cond_001",
                codec="opus",
                packet_loss=0.05,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="train",
            )
            for i in range(10)
        ]
        manifest = Manifest(samples)
        dist = manifest.class_distribution()
        assert dist[0] == 5
        assert dist[1] == 5
    
    def test_save_load(self):
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
            )
        ]
        manifest = Manifest(samples)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manifest.save(temp_path)
            loaded = Manifest.load(temp_path)
            assert len(loaded) == 1
            assert loaded[0].sample_id == "test_001"
        finally:
            os.unlink(temp_path)
    
    def test_compute_hash(self):
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
            )
        ]
        manifest = Manifest(samples)
        hash1 = manifest.compute_hash()
        hash2 = manifest.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 16


class TestCreateSyntheticManifest:
    def test_create_synthetic(self):
        manifest = create_synthetic_manifest(
            n_samples=100,
            n_speakers=5,
            n_generators=3,
            n_conditions=2,
            seed=42,
        )
        assert len(manifest) == 100
        assert all(s.split in {"train", "val", "test"} for s in manifest)
    
    def test_synthetic_class_balance(self):
        manifest = create_synthetic_manifest(n_samples=1000, seed=42)
        dist = manifest.class_distribution()
        assert 0 in dist
        assert 1 in dist


class TestAssignSplits:
    def test_speaker_disjoint(self):
        manifest = create_synthetic_manifest(n_samples=200, n_speakers=10, seed=42)
        split_manifest = assign_splits(manifest, strategy="speaker", seed=42)
        
        train_speakers = set(s.speaker_id for s in split_manifest if s.split == "train")
        val_speakers = set(s.speaker_id for s in split_manifest if s.split == "val")
        test_speakers = set(s.speaker_id for s in split_manifest if s.split == "test")
        
        assert len(train_speakers & val_speakers) == 0
        assert len(train_speakers & test_speakers) == 0
        assert len(val_speakers & test_speakers) == 0
    
    def test_generator_disjoint(self):
        manifest = create_synthetic_manifest(n_samples=200, n_generators=10, seed=42)
        split_manifest = assign_splits(manifest, strategy="generator", seed=42)
        
        train_gens = set(s.generator_id for s in split_manifest if s.split == "train")
        val_gens = set(s.generator_id for s in split_manifest if s.split == "val")
        test_gens = set(s.generator_id for s in split_manifest if s.split == "test")
        
        assert len(train_gens & val_gens) == 0
        assert len(train_gens & test_gens) == 0
        assert len(val_gens & test_gens) == 0
    
    def test_condition_disjoint(self):
        manifest = create_synthetic_manifest(n_samples=200, n_conditions=10, seed=42)
        split_manifest = assign_splits(manifest, strategy="condition", seed=42)
        
        train_conds = set(s.communication_condition_id for s in split_manifest if s.split == "train")
        val_conds = set(s.communication_condition_id for s in split_manifest if s.split == "val")
        test_conds = set(s.communication_condition_id for s in split_manifest if s.split == "test")
        
        assert len(train_conds & val_conds) == 0
        assert len(train_conds & test_conds) == 0
        assert len(val_conds & test_conds) == 0
    
    def test_random_split(self):
        manifest = create_synthetic_manifest(n_samples=200, seed=42)
        split_manifest = assign_splits(manifest, strategy="random", seed=42)
        
        assert all(s.split in {"train", "val", "test"} for s in split_manifest)
        train_count = sum(1 for s in split_manifest if s.split == "train")
        val_count = sum(1 for s in split_manifest if s.split == "val")
        test_count = sum(1 for s in split_manifest if s.split == "test")
        
        assert train_count > 0
        assert val_count > 0
        assert test_count > 0


class TestLeakageDetection:
    def test_check_entity_overlap_no_leakage(self):
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
                speaker_id="spk_002",
                generator_id="gen_002",
                class_label=0,
                audio_path="audio/test.wav",
                communication_condition_id="cond_002",
                codec="opus",
                packet_loss=0.05,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="val",
            ),
        ]
        manifest = Manifest(samples)
        result = check_leakage(manifest)
        
        assert result["speaker_overlap"]["has_leakage"] == False
        assert result["generator_overlap"]["has_leakage"] == False
        assert result["condition_overlap"]["has_leakage"] == False
    
    def test_check_entity_overlap_with_leakage(self):
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
        manifest = Manifest(samples)
        result = check_leakage(manifest)
        
        assert result["speaker_overlap"]["has_leakage"] == True
        assert "train_vs_val" in result["speaker_overlap"]["overlaps"]
    
    def test_check_condition_class_balance(self):
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
                speaker_id="spk_002",
                generator_id="gen_002",
                class_label=0,
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
                sample_id="test_003",
                speaker_id="spk_003",
                generator_id="gen_003",
                class_label=0,
                audio_path="audio/test.wav",
                communication_condition_id="cond_002",
                codec="opus",
                packet_loss=0.05,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="train",
            ),
        ]
        manifest = Manifest(samples)
        result = check_leakage(manifest)
        
        assert "cond_001" in result["condition_class_balance"]
        assert "cond_002" in result["condition_class_balance"]
        assert result["condition_class_balance"]["cond_001"]["is_imbalanced"] == False
        assert result["condition_class_balance"]["cond_002"]["is_imbalanced"] == True
    
    def test_check_network_feature_correlation(self):
        np.random.seed(42)
        samples = []
        for i in range(100):
            is_synthetic = np.random.rand() > 0.5
            label = 1 if is_synthetic else 0
            samples.append(SampleManifest(
                sample_id=f"test_{i:03d}",
                speaker_id=f"spk_{i%5:03d}",
                generator_id=f"gen_{i%3:03d}",
                class_label=label,
                audio_path=f"audio/test_{i}.wav",
                communication_condition_id=f"cond_{i%2:03d}",
                codec="opus",
                packet_loss=0.1 if is_synthetic else 0.01,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="train",
            ))
        
        manifest = Manifest(samples)
        result = check_leakage(manifest)
        
        assert "packet_loss" in result["network_feature_correlation"]
        assert abs(result["network_feature_correlation"]["packet_loss"]) > 0.1


class TestEnforceConditionClassBalance:
    def test_enforce_balance(self):
        np.random.seed(42)
        samples = []
        for i in range(100):
            label = 1 if i < 80 else 0
            samples.append(SampleManifest(
                sample_id=f"test_{i:03d}",
                speaker_id=f"spk_{i%5:03d}",
                generator_id=f"gen_{i%3:03d}",
                class_label=label,
                audio_path=f"audio/test_{i}.wav",
                communication_condition_id="cond_001",
                codec="opus",
                packet_loss=0.1 if label == 1 else 0.01,
                jitter=0.01,
                rtt=0.1,
                bitrate=64000,
                split="train",
            ))
        
        manifest = Manifest(samples)
        balanced = enforce_condition_class_balance(manifest, target_balance=0.5)
        
        dist = balanced.class_distribution()
        assert abs(dist[0] - dist[1]) <= 1


class TestCheckEntityOverlap:
    def test_speaker_overlap(self):
        df = pd.DataFrame({
            "speaker_id": ["spk_001", "spk_001", "spk_002", "spk_003"],
            "split": ["train", "val", "train", "test"],
        })
        
        result = check_entity_overlap(df, "speaker_id")
        assert result["has_leakage"] == True
        assert "train_vs_val" in result["overlaps"]
        assert "spk_001" in result["overlaps"]["train_vs_val"]


class TestCheckConditionClassBalance:
    def test_balanced_condition(self):
        df = pd.DataFrame({
            "communication_condition_id": ["cond_001", "cond_001"],
            "class_label": [0, 1],
        })
        
        result = check_condition_class_balance(df)
        assert result["cond_001"]["is_imbalanced"] == False
    
    def test_imbalanced_condition(self):
        df = pd.DataFrame({
            "communication_condition_id": ["cond_001"] * 10,
            "class_label": [0] * 9 + [1],
        })
        
        result = check_condition_class_balance(df)
        assert result["cond_001"]["is_imbalanced"] == True


class TestCheckNetworkFeatureCorrelation:
    def test_correlation(self):
        df = pd.DataFrame({
            "packet_loss": [0.1, 0.1, 0.01, 0.01],
            "jitter": [0.01, 0.01, 0.01, 0.01],
            "rtt": [0.1, 0.1, 0.1, 0.1],
            "bitrate": [64000, 64000, 64000, 64000],
            "class_label": [1, 1, 0, 0],
        })
        
        result = check_network_feature_correlation(df)
        assert abs(result["packet_loss"]) > 0.5
        assert abs(result["jitter"]) < 0.1