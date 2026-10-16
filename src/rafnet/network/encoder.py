import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class NetworkEncoderConfig:
    input_dim: int = 32
    hidden_dims: List[int] = None
    output_dim: int = 256
    use_transformer: bool = False
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1
    normalize_input: bool = True
    temporal_window: int = 10
    num_codecs: int = 8
    codec_embed_dim: int = 8
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]


class NetworkFeatureNormalizer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))
        self.register_buffer("count", torch.tensor(0))
        self.fitted = False

    def fit(self, x: torch.Tensor) -> None:
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0).clamp(min=1e-6)
        self.count = torch.tensor(x.shape[0])
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            return x
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class MaskedNormalizer(nn.Module):
    """Normalizer that respects missing-data masks."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))
        self.register_buffer("count", torch.zeros(input_dim))
        self.fitted = False

    def fit(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        if mask is None:
            mask = torch.ones_like(x)
        
        masked_x = x * mask
        self.count = mask.sum(dim=0).clamp(min=1)
        self.mean = masked_x.sum(dim=0) / self.count
        
        var = ((x - self.mean) ** 2 * mask).sum(dim=0) / self.count
        self.std = var.sqrt().clamp(min=1e-6)
        self.fitted = True

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.fitted:
            return x
        if mask is None:
            mask = torch.ones_like(x)
        normalized = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return normalized * mask.to(x.device)


class TemporalProcessor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, window: int = 10):
        super().__init__()
        self.window = window
        self.gru = nn.GRU(
            input_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, feat_dim = x.shape
        
        if seq_len < self.window:
            padding = torch.zeros(batch_size, self.window - seq_len, feat_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)
            if mask is not None:
                mask_padding = torch.zeros(batch_size, self.window - seq_len, feat_dim, device=mask.device)
                mask = torch.cat([mask, mask_padding], dim=1)
            seq_len = self.window
        
        num_windows = seq_len - self.window + 1
        windows = x.unfold(1, self.window, 1).permute(0, 1, 3, 2)
        windows = windows.reshape(batch_size * num_windows, self.window, feat_dim)
        
        if mask is not None:
            mask_windows = mask.unfold(1, self.window, 1).permute(0, 1, 3, 2)
            mask_windows = mask_windows.reshape(batch_size * num_windows, self.window, feat_dim)
        else:
            mask_windows = None
        
        if mask_windows is not None:
            windows = windows * mask_windows
        
        gru_out, _ = self.gru(windows)
        last_hidden = gru_out[:, -1, :]
        projected = self.projection(last_hidden)
        
        return projected.reshape(batch_size, num_windows, -1).mean(dim=1)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        x = torch.cat([cls_tokens, x], dim=1)
        
        if mask is not None:
            # mask has shape [batch, 1, feat_dim], we need [batch, seq_len] for padding mask
            # The mask indicates valid features per position, so we check if any feature is valid
            seq_mask = (mask.sum(dim=-1) > 0).float()  # [batch, seq_len]
            # Add cls token position (always valid)
            cls_mask = torch.ones(batch_size, 1, device=x.device, dtype=seq_mask.dtype)
            src_key_padding_mask = torch.cat([cls_mask, seq_mask], dim=1) == 0
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer(x)
        
        return x[:, 0]


class CodecEmbedding(nn.Module):
    """Embedding layer for codec types that prevents generator identity leakage."""
    
    def __init__(self, num_codecs: int, embed_dim: int):
        super().__init__()
        self.num_codecs = num_codecs
        self.embedding = nn.Embedding(num_codecs + 1, embed_dim, padding_idx=num_codecs)
        self.embed_dim = embed_dim

    def forward(self, codec_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(codec_indices)


class NetworkEncoder(nn.Module):
    def __init__(self, config: NetworkEncoderConfig):
        super().__init__()
        self.config = config
        
        self.normalizer = MaskedNormalizer(config.input_dim)
        
        self.temporal_processor = TemporalProcessor(
            config.input_dim, config.hidden_dims[0], config.temporal_window
        )
        
        self.codec_embedding = CodecEmbedding(config.num_codecs, config.codec_embed_dim)
        
        mlp_input_dim = config.hidden_dims[0] + config.codec_embed_dim
        mlp_layers = []
        in_dim = mlp_input_dim
        for hidden_dim in config.hidden_dims[1:]:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = hidden_dim
        
        if len(config.hidden_dims) == 1:
            self.mlp = nn.Identity()
            mlp_output_dim = mlp_input_dim
        elif len(mlp_layers) == 0:
            self.mlp = nn.Identity()
            mlp_output_dim = mlp_input_dim
        else:
            self.mlp = nn.Sequential(*mlp_layers)
            mlp_output_dim = config.hidden_dims[-1]
        
        if config.use_transformer:
            self.transformer = TransformerEncoder(
                mlp_output_dim,
                config.transformer_layers,
                config.transformer_heads,
                config.dropout
            )
            transformer_out_dim = mlp_output_dim
        else:
            self.transformer = None
            transformer_out_dim = mlp_output_dim
        
        self.projection = nn.Sequential(
            nn.Linear(transformer_out_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

    def fit_normalizer(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        self.normalizer.fit(x, mask)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        codec_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if mask is not None and mask.dim() == 2:
                mask = mask.unsqueeze(1)
        
        x = self.normalizer(x, mask)
        
        temporal_features = self.temporal_processor(x, mask)
        
        if codec_indices is not None:
            codec_embed = self.codec_embedding(codec_indices)
        else:
            codec_embed = torch.zeros(
                temporal_features.shape[0], 
                self.config.codec_embed_dim, 
                device=temporal_features.device,
                dtype=temporal_features.dtype
            )
        temporal_features = torch.cat([temporal_features, codec_embed], dim=-1)
        
        mlp_features = self.mlp(temporal_features)
        
        if self.transformer is not None:
            mlp_features = mlp_features.unsqueeze(1)
            if mask is not None:
                pooled_mask = mask.mean(dim=1, keepdim=True)
                pooled_mask = (pooled_mask > 0.5).float()
            else:
                pooled_mask = None
            transformer_features = self.transformer(mlp_features, pooled_mask)
        else:
            transformer_features = mlp_features
        
        output = self.projection(transformer_features)
        return output

    def get_output_dim(self) -> int:
        return self.config.output_dim