import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig
from typing import Optional, Literal
import time
from dataclasses import dataclass


@dataclass
class WavLMEncoderConfig:
    model_name: str = "microsoft/wavlm-base-plus"
    freeze: bool = True
    output_dim: int = 256
    pooling: Literal["mean", "attention", "cls"] = "mean"
    use_checkpointing: bool = False
    mixed_precision: bool = False
    device: str = "cpu"
    projection_hidden_dim: Optional[int] = None
    projection_dropout: float = 0.1


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        weights = self.attention(x).squeeze(-1)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(weights, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled


class WavLMEncoder(nn.Module):
    def __init__(self, config: WavLMEncoderConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.freeze = config.freeze
        self.output_dim = config.output_dim
        self.pooling = config.pooling
        self.use_checkpointing = config.use_checkpointing
        self.mixed_precision = config.mixed_precision
        
        self.wavlm = WavLMModel.from_pretrained(config.model_name)
        self.hidden_size = self.wavlm.config.hidden_size
        
        if config.freeze:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.wavlm.eval()
        elif config.use_checkpointing:
            self.wavlm.gradient_checkpointing_enable()
        
        proj_hidden = config.projection_hidden_dim or (self.hidden_size // 2)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.GELU(),
            nn.Dropout(config.projection_dropout),
            nn.Linear(proj_hidden, config.output_dim),
        )
        
        if config.pooling == "attention":
            self.pooling_module = AttentionPooling(self.hidden_size)
        elif config.pooling == "cls":
            self.register_buffer("cls_token", torch.zeros(1, 1, self.hidden_size))
        
        self.to(config.device)
        
        self._register_timing_hooks()
    
    def _register_timing_hooks(self):
        self._forward_times = []
        
        def pre_hook(module, input):
            self._start_time = time.perf_counter()
        
        def post_hook(module, input, output):
            elapsed = time.perf_counter() - self._start_time
            self._forward_times.append(elapsed)
        
        self.wavlm.register_forward_pre_hook(pre_hook)
        self.wavlm.register_forward_hook(post_hook)
    
    def get_timing_stats(self) -> dict:
        if not self._forward_times:
            return {}
        times = torch.tensor(self._forward_times)
        return {
            "mean_ms": times.mean().item() * 1000,
            "std_ms": times.std().item() * 1000,
            "min_ms": times.min().item() * 1000,
            "max_ms": times.max().item() * 1000,
            "num_runs": len(self._forward_times),
        }
    
    def reset_timing(self):
        self._forward_times = []
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> torch.Tensor:
        if self.mixed_precision and x.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self._forward_impl(x, attention_mask, return_hidden)
        return self._forward_impl(x, attention_mask, return_hidden)
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                outputs = self.wavlm(
                    x,
                    attention_mask=attention_mask,
                    output_hidden_states=return_hidden
                )
        else:
            outputs = self.wavlm(
                x,
                attention_mask=attention_mask,
                output_hidden_states=return_hidden
            )
        
        last_hidden = outputs.last_hidden_state
        
        if attention_mask is not None:
            seq_len = last_hidden.shape[1]
            if attention_mask.shape[1] != seq_len:
                if attention_mask.shape[1] > seq_len:
                    attention_mask = attention_mask[:, :seq_len]
                else:
                    padding = torch.zeros(
                        attention_mask.shape[0],
                        seq_len - attention_mask.shape[1],
                        device=attention_mask.device,
                        dtype=attention_mask.dtype
                    )
                    attention_mask = torch.cat([attention_mask, padding], dim=1)
        
        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = last_hidden.mean(dim=1)
                
        elif self.pooling == "attention":
            pooled = self.pooling_module(last_hidden, attention_mask)
            
        elif self.pooling == "cls":
            if attention_mask is not None:
                pooled = last_hidden[:, 0]
            else:
                pooled = last_hidden[:, 0]
        
        projected = self.projection(pooled)
        
        if return_hidden:
            return projected, outputs.hidden_states
        return projected
    
    def extract_features(self, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.wavlm(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
        return hidden_states
    
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        wavlm_params = sum(p.numel() for p in self.wavlm.parameters())
        proj_params = sum(p.numel() for p in self.projection.parameters())
        pooling_params = sum(p.numel() for p in self.pooling_module.parameters()) if hasattr(self, 'pooling_module') else 0
        
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "wavlm": wavlm_params,
            "projection": proj_params,
            "pooling": pooling_params,
        }
    
    def estimate_memory(self, batch_size: int, seq_len: int) -> dict:
        element_size = 2 if self.mixed_precision else 4
        
        input_mem = batch_size * seq_len * element_size
        hidden_mem = batch_size * seq_len * self.hidden_size * element_size * 2
        
        if self.pooling == "attention":
            pooling_mem = batch_size * seq_len * (self.hidden_size // 4) * element_size
        else:
            pooling_mem = 0
        
        projection_mem = batch_size * self.output_dim * element_size
        
        activation_mem = hidden_mem + pooling_mem + projection_mem
        param_mem = sum(p.numel() for p in self.parameters()) * element_size
        grad_mem = sum(p.numel() for p in self.parameters() if p.requires_grad) * element_size
        
        total_mem = input_mem + activation_mem + param_mem + grad_mem
        
        return {
            "input_mb": input_mem / 1e6,
            "activations_mb": activation_mem / 1e6,
            "parameters_mb": param_mem / 1e6,
            "gradients_mb": grad_mem / 1e6,
            "total_mb": total_mem / 1e6,
        }
    
    def get_output_dim(self) -> int:
        return self.output_dim


def create_wavlm_encoder(
    model_name: str = "microsoft/wavlm-base-plus",
    freeze: bool = True,
    output_dim: int = 256,
    pooling: str = "mean",
    use_checkpointing: bool = False,
    mixed_precision: bool = False,
    device: str = "cpu",
) -> WavLMEncoder:
    config = WavLMEncoderConfig(
        model_name=model_name,
        freeze=freeze,
        output_dim=output_dim,
        pooling=pooling,
        use_checkpointing=use_checkpointing,
        mixed_precision=mixed_precision,
        device=device,
    )
    return WavLMEncoder(config)