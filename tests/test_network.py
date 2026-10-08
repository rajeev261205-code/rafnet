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
class TestNetworkEncoder:
    def test_network_encoder_init(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
            use_transformer=config.network.use_transformer,
        )
        encoder = NetworkEncoder(net_config)
        assert encoder is not None
        assert encoder.get_output_dim() == config.network.output_dim
    
    def test_network_encoder_forward(self, config, sample_network):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
        )
        encoder = NetworkEncoder(net_config)
        
        output = encoder(sample_network)
        assert output.shape == (2, config.network.output_dim)
        assert not torch.isnan(output).any()
    
    def test_network_encoder_fit_normalizer(self, config):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        net_config = NetworkEncoderConfig(input_dim=config.network.input_dim)
        encoder = NetworkEncoder(net_config)
        
        train_data = torch.randn(100, config.network.input_dim)
        encoder.fit_normalizer(train_data)
        
        test_data = torch.randn(10, config.network.input_dim)
        output = encoder(test_data)
        assert output.shape == (10, config.network.output_dim)
    
    def test_network_encoder_with_transformer(self, config, sample_network):
        from rafnet.network.encoder import NetworkEncoder, NetworkEncoderConfig
        net_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
            use_transformer=True,
            transformer_layers=2,
            transformer_heads=4,
        )
        encoder = NetworkEncoder(net_config)
        
        output = encoder(sample_network)
        assert output.shape == (2, config.network.output_dim)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestNetworkFeatureNormalizer:
    def test_normalizer_fit_transform(self):
        from rafnet.network.encoder import NetworkFeatureNormalizer
        normalizer = NetworkFeatureNormalizer(10)
        data = torch.randn(100, 10) * 5 + 2
        
        normalizer.fit(data)
        normalized = normalizer(data)
        
        assert normalized.shape == data.shape
        assert torch.allclose(normalized.mean(dim=0), torch.zeros(10), atol=1e-1)
        assert torch.allclose(normalized.std(dim=0), torch.ones(10), atol=1e-1)
    
    def test_normalizer_unfitted(self):
        from rafnet.network.encoder import NetworkFeatureNormalizer
        normalizer = NetworkFeatureNormalizer(10)
        data = torch.randn(5, 10)
        output = normalizer(data)
        assert torch.allclose(output, data)