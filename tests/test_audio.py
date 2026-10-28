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
class TestAudioEncoder:
    def test_audio_encoder_init(self, config):
        from rafnet.audio.encoder import AudioEncoder, AudioEncoderConfig
        audio_config = AudioEncoderConfig(
            sample_rate=config.audio.sample_rate,
            wavlm_model=config.audio.wavlm_model,
            wavlm_freeze=config.audio.wavlm_freeze,
            projection_dim=config.audio.projection_dim,
            use_handcrafted=config.audio.use_handcrafted,
            device=config.device,
        )
        encoder = AudioEncoder(audio_config)
        assert encoder is not None
        assert encoder.get_output_dim() == config.audio.projection_dim
    
    def test_audio_encoder_forward(self, config, sample_audio):
        from rafnet.audio.encoder import AudioEncoder, AudioEncoderConfig
        audio_config = AudioEncoderConfig(
            sample_rate=config.audio.sample_rate,
            wavlm_model=config.audio.wavlm_model,
            wavlm_freeze=config.audio.wavlm_freeze,
            projection_dim=config.audio.projection_dim,
            use_handcrafted=config.audio.use_handcrafted,
            device=config.device,
        )
        encoder = AudioEncoder(audio_config)
        
        output = encoder(sample_audio)
        assert output.shape == (2, config.audio.projection_dim)
        assert not torch.isnan(output).any()
    
    def test_audio_encoder_with_handcrafted(self, config, sample_audio):
        from rafnet.audio.encoder import AudioEncoder, AudioEncoderConfig
        audio_config = AudioEncoderConfig(
            sample_rate=config.audio.sample_rate,
            wavlm_model=config.audio.wavlm_model,
            wavlm_freeze=config.audio.wavlm_freeze,
            projection_dim=config.audio.projection_dim,
            use_handcrafted=True,
            handcrafted_features=["mfcc", "spectral"],
            device=config.device,
        )
        encoder = AudioEncoder(audio_config)
        
        output = encoder(sample_audio)
        assert output.shape == (2, config.audio.projection_dim)
        assert not torch.isnan(output).any()
    
    def test_audio_encoder_return_components(self, config, sample_audio):
        from rafnet.audio.encoder import AudioEncoder, AudioEncoderConfig
        audio_config = AudioEncoderConfig(
            sample_rate=config.audio.sample_rate,
            wavlm_model=config.audio.wavlm_model,
            wavlm_freeze=config.audio.wavlm_freeze,
            projection_dim=config.audio.projection_dim,
            use_handcrafted=True,
            handcrafted_features=["mfcc"],
            device=config.device,
        )
        encoder = AudioEncoder(audio_config)
        
        output = encoder(sample_audio, return_components=True)
        assert "fused" in output
        assert "wavlm" in output
        assert "handcrafted" in output
        assert output["fused"].shape == (2, config.audio.projection_dim)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestHandcraftedFeatures:
    def test_handcrafted_extractor_init(self):
        from rafnet.audio.handcrafted import HandcraftedFeatureExtractor, HandcraftedConfig
        config = HandcraftedConfig()
        extractor = HandcraftedFeatureExtractor(config, ["mfcc", "lfcc", "spectral"])
        assert extractor is not None
    
    def test_handcrafted_forward(self, sample_audio):
        from rafnet.audio.handcrafted import HandcraftedFeatureExtractor, HandcraftedConfig
        config = HandcraftedConfig()
        extractor = HandcraftedFeatureExtractor(config, ["mfcc", "spectral"])
        
        features = extractor(sample_audio)
        assert "mfcc" in features
        assert "spectral" in features
        assert features["mfcc"].shape[0] == 2
        assert features["spectral"].shape[0] == 2
    
    def test_individual_extractors(self, sample_audio):
        from rafnet.audio.handcrafted import (
            LFCCExtractor, MFCCExtractor, CQTExtractor,
            SpectralExtractor, ProsodicExtractor, VoiceQualityExtractor,
            HandcraftedConfig
        )
        config = HandcraftedConfig()
        
        lfcc_ext = LFCCExtractor(config)
        mfcc_ext = MFCCExtractor(config)
        spectral_ext = SpectralExtractor(config)
        prosodic_ext = ProsodicExtractor(config)
        vq_ext = VoiceQualityExtractor(config)
        
        lfcc = lfcc_ext(sample_audio)
        mfcc = mfcc_ext(sample_audio)
        spectral = spectral_ext(sample_audio)
        prosodic = prosodic_ext(sample_audio)
        vq = vq_ext(sample_audio)
        
        assert lfcc.shape[0] == 2
        assert mfcc.shape[0] == 2
        assert spectral.shape[0] == 2
        assert prosodic.shape[0] == 2
        assert vq.shape[0] == 2