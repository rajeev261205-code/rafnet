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


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestNetworkTelemetryEncoder:
    def test_telemetry_encoder_basic(self):
        from rafnet.data.dataset import NetworkTelemetryEncoder, SampleManifest
        
        encoder = NetworkTelemetryEncoder(network_dim=32)
        
        sample = SampleManifest(
            sample_id="test_001",
            speaker_id="spk_001",
            generator_id="gen_001",
            class_label=0,
            audio_path="audio/test.wav",
            communication_condition_id="cond_001",
            codec="opus",
            packet_loss=0.05,
            jitter=0.02,
            rtt=0.1,
            bitrate=64000,
            packet_timing=[0.0, 0.02, 0.04, 0.06, 0.08],
            packet_size=[1200, 1180, 1220, 1190, 1210],
            burst_loss_count=1,
            burst_loss_duration=0.05,
            loss_burstiness=0.3,
            split="train"
        )
        
        features, mask = encoder.extract_features(sample)
        
        assert features.shape == (32,)
        assert mask.shape == (32,)
        assert not np.isnan(features).any()
        assert mask.sum() > 0
    
    def test_telemetry_encoder_missing_packet_timing(self):
        from rafnet.data.dataset import NetworkTelemetryEncoder, SampleManifest
        
        encoder = NetworkTelemetryEncoder(network_dim=32)
        
        sample = SampleManifest(
            sample_id="test_002",
            speaker_id="spk_001",
            generator_id="gen_001",
            class_label=0,
            audio_path="audio/test.wav",
            communication_condition_id="cond_001",
            codec="opus",
            packet_loss=0.05,
            jitter=0.02,
            rtt=0.1,
            bitrate=64000,
            packet_timing=None,
            packet_size=[1200, 1180, 1220],
            burst_loss_count=0,
            burst_loss_duration=0.0,
            loss_burstiness=0.0,
            split="train"
        )
        
        features, mask = encoder.extract_features(sample)
        
        assert features.shape == (32,)
        assert mask.shape == (32,)
        timing_mask_indices = [4, 5, 6]
        for idx in timing_mask_indices:
            assert mask[idx] == 0.0
    
    def test_telemetry_encoder_missing_packet_size(self):
        from rafnet.data.dataset import NetworkTelemetryEncoder, SampleManifest
        
        encoder = NetworkTelemetryEncoder(network_dim=32)
        
        sample = SampleManifest(
            sample_id="test_003",
            speaker_id="spk_001",
            generator_id="gen_001",
            class_label=0,
            audio_path="audio/test.wav",
            communication_condition_id="cond_001",
            codec="opus",
            packet_loss=0.05,
            jitter=0.02,
            rtt=0.1,
            bitrate=64000,
            packet_timing=[0.0, 0.02, 0.04],
            packet_size=None,
            burst_loss_count=0,
            burst_loss_duration=0.0,
            loss_burstiness=0.0,
            split="train"
        )
        
        features, mask = encoder.extract_features(sample)
        
        assert features.shape == (32,)
        size_mask_indices = [7, 8]
        for idx in size_mask_indices:
            assert mask[idx] == 0.0
    
    def test_telemetry_encoder_missing_burst_stats(self):
        from rafnet.data.dataset import NetworkTelemetryEncoder, SampleManifest
        
        encoder = NetworkTelemetryEncoder(network_dim=32)
        
        sample = SampleManifest(
            sample_id="test_004",
            speaker_id="spk_001",
            generator_id="gen_001",
            class_label=0,
            audio_path="audio/test.wav",
            communication_condition_id="cond_001",
            codec="opus",
            packet_loss=0.05,
            jitter=0.02,
            rtt=0.1,
            bitrate=64000,
            packet_timing=[0.0, 0.02, 0.04],
            packet_size=[1200, 1180, 1220],
            burst_loss_count=None,
            burst_loss_duration=None,
            loss_burstiness=None,
            split="train"
        )
        
        features, mask = encoder.extract_features(sample)
        
        assert features.shape == (32,)
        burst_mask_indices = [9, 10, 11]
        for idx in burst_mask_indices:
            assert mask[idx] == 0.0
    
    def test_telemetry_encoder_unknown_codec(self):
        from rafnet.data.dataset import NetworkTelemetryEncoder, SampleManifest
        
        encoder = NetworkTelemetryEncoder(network_dim=32)
        
        sample = SampleManifest(
            sample_id="test_005",
            speaker_id="spk_001",
            generator_id="gen_001",
            class_label=0,
            audio_path="audio/test.wav",
            communication_condition_id="cond_001",
            codec="unknown_codec",
            packet_loss=0.05,
            jitter=0.02,
            rtt=0.1,
            bitrate=64000,
            packet_timing=[0.0, 0.02, 0.04],
            packet_size=[1200, 1180, 1220],
            burst_loss_count=0,
            burst_loss_duration=0.0,
            loss_burstiness=0.0,
            split="train"
        )
        
        features, mask = encoder.extract_features(sample)
        
        assert features.shape == (32,)
        codec_start = 12
        assert mask[codec_start + 7] == 0.0
    
    def test_telemetry_encoder_known_codecs(self):
        from rafnet.data.dataset import NetworkTelemetryEncoder, SampleManifest
        
        encoder = NetworkTelemetryEncoder(network_dim=32)
        
        for codec in ["opus", "g722", "pcmu", "pcma", "g729", "amr", "silk"]:
            sample = SampleManifest(
                sample_id=f"test_{codec}",
                speaker_id="spk_001",
                generator_id="gen_001",
                class_label=0,
                audio_path="audio/test.wav",
                communication_condition_id="cond_001",
                codec=codec,
                packet_loss=0.05,
                jitter=0.02,
                rtt=0.1,
                bitrate=64000,
                packet_timing=[0.0, 0.02, 0.04],
                packet_size=[1200, 1180, 1220],
                burst_loss_count=0,
                burst_loss_duration=0.0,
                loss_burstiness=0.0,
                split="train"
            )
            
            features, mask = encoder.extract_features(sample)
            assert features.shape == (32,)
            codec_start = 12
            codec_idx = encoder.codec_to_idx[codec]
            assert mask[codec_start + codec_idx] == 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestNetworkEncoderEdgeCases:
    def test_missing_telemetry(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.network.input_dim)
        mask = torch.zeros_like(x)
        mask[:, :4] = 1.0
        
        encoder.fit_normalizer(x, mask)
        output = encoder(x, mask=mask)
        
        assert output.shape == (batch_size, config.network.output_dim)
        assert not torch.isnan(output).any()
    
    def test_malformed_telemetry(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.network.input_dim)
        x[0, 0] = float('inf')
        x[1, 1] = float('-inf')
        x[2, 2] = float('nan')
        
        mask = torch.ones_like(x)
        mask[0, 0] = 0.0
        mask[1, 1] = 0.0
        mask[2, 2] = 0.0
        
        clean_x = torch.randn(batch_size, config.network.input_dim)
        encoder.fit_normalizer(clean_x)
        
        output = encoder(x, mask=mask)
        
        assert output.shape == (batch_size, config.network.output_dim)
    
    def test_extreme_jitter(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.network.input_dim)
        x[:, 1] = 1000.0
        
        mask = torch.ones_like(x)
        clean_x = torch.randn(batch_size, config.network.input_dim)
        encoder.fit_normalizer(clean_x)
        
        output = encoder(x, mask=mask)
        
        assert output.shape == (batch_size, config.network.output_dim)
        assert not torch.isnan(output).any()
    
    def test_zero_packet_loss(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.network.input_dim)
        x[:, 0] = 0.0
        
        mask = torch.ones_like(x)
        clean_x = torch.randn(batch_size, config.network.input_dim)
        encoder.fit_normalizer(clean_x)
        
        output = encoder(x, mask=mask)
        
        assert output.shape == (batch_size, config.network.output_dim)
        assert not torch.isnan(output).any()
    
    def test_high_packet_loss(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.network.input_dim)
        x[:, 0] = 0.95
        
        mask = torch.ones_like(x)
        clean_x = torch.randn(batch_size, config.network.input_dim)
        encoder.fit_normalizer(clean_x)
        
        output = encoder(x, mask=mask)
        
        assert output.shape == (batch_size, config.network.output_dim)
        assert not torch.isnan(output).any()
    
    def test_unseen_codec(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
            num_codecs=config.network.num_codecs,
            codec_embed_dim=config.network.codec_embed_dim,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.network.input_dim)
        mask = torch.ones_like(x)
        codec_indices = torch.full((batch_size,), config.network.num_codecs, dtype=torch.long)
        
        clean_x = torch.randn(batch_size, config.network.input_dim)
        encoder.fit_normalizer(clean_x)
        
        output = encoder(x, mask=mask, codec_indices=codec_indices)
        
        assert output.shape == (batch_size, config.network.output_dim)
        assert not torch.isnan(output).any()
    
    def test_missing_codec(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
            num_codecs=config.network.num_codecs,
            codec_embed_dim=config.network.codec_embed_dim,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.network.input_dim)
        mask = torch.ones_like(x)
        codec_indices = None
        
        clean_x = torch.randn(batch_size, config.network.input_dim)
        encoder.fit_normalizer(clean_x)
        
        output = encoder(x, mask=mask, codec_indices=codec_indices)
        
        assert output.shape == (batch_size, config.network.output_dim)
        assert not torch.isnan(output).any()
    
    def test_with_transformer(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
            use_transformer=True,
            transformer_layers=2,
            transformer_heads=4,
            num_codecs=config.network.num_codecs,
            codec_embed_dim=config.network.codec_embed_dim,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, config.network.input_dim)
        mask = torch.ones(batch_size, seq_len, config.network.input_dim)
        codec_indices = torch.randint(0, config.network.num_codecs, (batch_size,))
        
        clean_x = torch.randn(100, config.network.input_dim)
        encoder.fit_normalizer(clean_x)
        
        output = encoder(x, mask=mask, codec_indices=codec_indices)
        
        assert output.shape == (batch_size, config.network.output_dim)
        assert not torch.isnan(output).any()
    
    def test_temporal_window_processing(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
            temporal_window=5,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 2
        seq_len = 20
        x = torch.randn(batch_size, seq_len, config.network.input_dim)
        mask = torch.ones(batch_size, seq_len, config.network.input_dim)
        
        clean_x = torch.randn(100, config.network.input_dim)
        encoder.fit_normalizer(clean_x)
        
        output = encoder(x, mask=mask)
        
        assert output.shape == (batch_size, config.network.output_dim)
        assert not torch.isnan(output).any()
    
    def test_short_sequence_padding(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
            temporal_window=10,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 2
        seq_len = 3
        x = torch.randn(batch_size, seq_len, config.network.input_dim)
        mask = torch.ones(batch_size, seq_len, config.network.input_dim)
        
        clean_x = torch.randn(100, config.network.input_dim)
        encoder.fit_normalizer(clean_x)
        
        output = encoder(x, mask=mask)
        
        assert output.shape == (batch_size, config.network.output_dim)
        assert not torch.isnan(output).any()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestMaskedNormalizer:
    def test_masked_normalizer_fit_transform(self):
        from rafnet.network.encoder import MaskedNormalizer
        
        normalizer = MaskedNormalizer(10)
        data = torch.randn(100, 10) * 5 + 2
        mask = torch.ones_like(data)
        mask[:20, 0] = 0.0
        mask[30:50, 1:3] = 0.0
        
        normalizer.fit(data, mask)
        normalized = normalizer(data, mask)
        
        assert normalized.shape == data.shape
        valid_mask = mask.bool()
        assert torch.allclose(normalized[valid_mask].mean(), torch.tensor(0.0), atol=1e-1)
        assert torch.allclose(normalized[valid_mask].std(), torch.tensor(1.0), atol=1e-1)
    
    def test_masked_normalizer_unfitted(self):
        from rafnet.network.encoder import MaskedNormalizer
        
        normalizer = MaskedNormalizer(10)
        data = torch.randn(5, 10)
        mask = torch.ones_like(data)
        output = normalizer(data, mask)
        assert torch.allclose(output, data)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestCodecEmbedding:
    def test_codec_embedding_known(self):
        from rafnet.network.encoder import CodecEmbedding
        
        embedding = CodecEmbedding(num_codecs=8, embed_dim=8)
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        
        output = embedding(indices)
        
        assert output.shape == (8, 8)
        assert not torch.isnan(output).any()
    
    def test_codec_embedding_unknown(self):
        from rafnet.network.encoder import CodecEmbedding
        
        embedding = CodecEmbedding(num_codecs=8, embed_dim=8)
        indices = torch.tensor([8, 8, 8])
        
        output = embedding(indices)
        
        assert output.shape == (3, 8)
        assert torch.allclose(output, torch.zeros_like(output))
    
    def test_codec_embedding_mixed(self):
        from rafnet.network.encoder import CodecEmbedding
        
        embedding = CodecEmbedding(num_codecs=8, embed_dim=8)
        indices = torch.tensor([0, 8, 3, 8, 7])
        
        output = embedding(indices)
        
        assert output.shape == (5, 8)
        assert not torch.isnan(output).any()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDatasetIntegration:
    def test_dataset_with_telemetry_encoder(self):
        from rafnet.data.dataset import SyntheticVoiceDataset, NetworkTelemetryEncoder
        from rafnet.data.manifest import create_synthetic_manifest
        
        manifest = create_synthetic_manifest(n_samples=20, n_speakers=5, n_generators=3, n_conditions=2, seed=42)
        
        encoder = NetworkTelemetryEncoder(network_dim=32)
        dataset = SyntheticVoiceDataset(
            manifest=manifest,
            network_dim=32,
            telemetry_encoder=encoder,
            max_samples=10
        )
        
        assert len(dataset) == 10
        
        item = dataset[0]
        assert "network" in item
        assert "network_mask" in item
        assert item["network"].shape == (32,)
        assert item["network_mask"].shape == (32,)
    
    def test_collate_fn_with_masks(self):
        from rafnet.data.dataset import SyntheticVoiceDataset, NetworkTelemetryEncoder, collate_fn
        from rafnet.data.manifest import create_synthetic_manifest
        
        manifest = create_synthetic_manifest(n_samples=20, n_speakers=5, n_generators=3, n_conditions=2, seed=42)
        
        encoder = NetworkTelemetryEncoder(network_dim=32)
        dataset = SyntheticVoiceDataset(
            manifest=manifest,
            network_dim=32,
            telemetry_encoder=encoder,
            max_samples=4
        )
        
        batch = [dataset[i] for i in range(4)]
        collated = collate_fn(batch)
        
        assert "network" in collated
        assert "network_mask" in collated
        assert collated["network"].shape == (4, 32)
        assert collated["network_mask"].shape == (4, 32)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestOutputDimension:
    def test_output_dim_256(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=256,
            num_codecs=config.network.num_codecs,
            codec_embed_dim=config.network.codec_embed_dim,
        )
        encoder = NetworkEncoder(net_config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.network.input_dim)
        mask = torch.ones_like(x)
        codec_indices = torch.randint(0, config.network.num_codecs, (batch_size,))
        
        clean_x = torch.randn(100, config.network.input_dim)
        encoder.fit_normalizer(clean_x)
        
        output = encoder(x, mask=mask, codec_indices=codec_indices)
        
        assert output.shape == (batch_size, 256)
        assert encoder.get_output_dim() == 256