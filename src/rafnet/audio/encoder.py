import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass

from .wavlm_encoder import WavLMEncoder, WavLMEncoderConfig, create_wavlm_encoder
from .handcrafted import HandcraftedFeatureExtractor, HandcraftedConfig


@dataclass
class AudioEncoderConfig:
    sample_rate: int = 16000
    wavlm_model: str = "microsoft/wavlm-base-plus"
    wavlm_freeze: bool = True
    wavlm_output_dim: int = 768
    projection_dim: int = 256
    wavlm_pooling: str = "mean"
    wavlm_checkpointing: bool = False
    wavlm_mixed_precision: bool = False
    use_handcrafted: bool = True
    handcrafted_features: List[str] = None
    handcrafted_projection_dim: int = 128
    device: str = "cpu"
    wavlm_stride: int = 320
    
    def __post_init__(self):
        if self.handcrafted_features is None:
            self.handcrafted_features = ["lfcc", "mfcc", "cqt", "spectral", "prosodic", "voice_quality"]


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        
        self.wavlm_encoder = create_wavlm_encoder(
            model_name=config.wavlm_model,
            freeze=config.wavlm_freeze,
            output_dim=config.projection_dim,
            pooling=config.wavlm_pooling,
            use_checkpointing=config.wavlm_checkpointing,
            mixed_precision=config.wavlm_mixed_precision,
            device=config.device,
        )
        
        self.use_handcrafted = config.use_handcrafted
        self.handcrafted_dim = 0
        
        if self.use_handcrafted:
            hc_config = HandcraftedConfig(
                sample_rate=config.sample_rate,
                n_mfcc=40,
                n_lfcc=60,
                n_cqt=84
            )
            self.handcrafted_extractor = HandcraftedFeatureExtractor(
                hc_config, config.handcrafted_features
            )
            self.handcrafted_dim = self.handcrafted_extractor.get_feature_dim()
            
            self.handcrafted_projection = nn.Sequential(
                nn.Linear(self.handcrafted_dim, config.handcrafted_projection_dim),
                nn.LayerNorm(config.handcrafted_projection_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        
        self.output_dim = config.projection_dim + (config.handcrafted_projection_dim if self.use_handcrafted else 0)
        
        self.fusion = nn.Sequential(
            nn.Linear(self.output_dim, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        self.to(config.device)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        # Convert attention mask from raw audio length to WavLM feature extractor output length
        wavlm_attention_mask = None
        if attention_mask is not None:
            # Downsample mask: WavLM feature extractor has stride of 320 for 16kHz audio
            wavlm_attention_mask = attention_mask[:, ::self.config.wavlm_stride]
            # Ensure mask is boolean or float
            wavlm_attention_mask = wavlm_attention_mask.float()
        
        wavlm_features = self.wavlm_encoder(x, wavlm_attention_mask)
        
        # If mask was provided but length doesn't match, trim or pad mask to match WavLM output
        if wavlm_attention_mask is not None:
            # We'll handle this in the wavlm_encoder by checking the actual output length
            pass
        
        if self.use_handcrafted:
            hc_features_dict = self.handcrafted_extractor(x)
            hc_features = torch.cat(list(hc_features_dict.values()), dim=-1)
            hc_projected = self.handcrafted_projection(hc_features)
            combined = torch.cat([wavlm_features, hc_projected], dim=-1)
        else:
            combined = wavlm_features
            hc_features_dict = {}
        
        output = self.fusion(combined)
        
        if return_components:
            return {
                "fused": output,
                "wavlm": wavlm_features,
                "handcrafted": hc_features_dict
            }
        return output

    def get_output_dim(self) -> int:
        return self.config.projection_dim