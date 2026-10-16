import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Gateway1Config:
    audio_dim: int = 256
    network_dim: int = 256
    context_dim: int = 0
    hidden_dim: int = 128
    gate_type: str = "sigmoid"
    reliability_aware: bool = True
    dropout: float = 0.1
    num_layers: int = 2
    use_residual_fusion: bool = False
    residual_weight: float = 0.1
    gate_bias_init: float = 0.0
    log_gate_stats: bool = True
    ablation_mode: Optional[str] = None

    @property
    def input_dim(self) -> int:
        return self.audio_dim + self.network_dim + 2 * self.context_dim


class ReliabilityFeatureExtractor(nn.Module):
    """
    Extracts reliability/context features from modality embeddings.
    
    Reliability features capture the quality/confidence of each modality:
    
    - Audio reliability (c_a): Derived from audio embedding norm, spectral features,
      or learned projection. High norm/entropy → more confident audio.
    - Network reliability (c_n): Derived from network embedding, packet loss,
      jitter, burst statistics. Low packet loss, stable timing → more confident network.
    
    These are NOT hardcoded heuristics but learned projections that the gate
    can use to make data-driven fusion decisions.
    """
    
    def __init__(self, audio_dim: int, network_dim: int, hidden_dim: int, context_dim: int = 0):
        super().__init__()
        self.context_dim = context_dim
        
        # Audio reliability head: projects audio embedding to reliability score
        self.audio_head = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim if context_dim > 0 else 1),
        )
        
        # Network reliability head: projects network embedding to reliability score
        self.network_head = nn.Sequential(
            nn.Linear(network_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim if context_dim > 0 else 1),
        )
        
        if context_dim > 0:
            self.audio_proj = nn.Linear(context_dim, context_dim)
            self.network_proj = nn.Linear(context_dim, context_dim)
    
    def forward(
        self, 
        z_a: torch.Tensor, 
        z_n: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute reliability/context features from embeddings.
        
        Args:
            z_a: Audio embedding [B, D_a]
            z_n: Network embedding [B, D_n]
            
        Returns:
            c_a: Audio context/reliability [B, context_dim] or [B, 1]
            c_n: Network context/reliability [B, context_dim] or [B, 1]
        """
        c_a = self.audio_head(z_a)
        c_n = self.network_head(z_n)
        
        if self.context_dim > 0:
            c_a = self.audio_proj(c_a)
            c_n = self.network_proj(c_n)
        else:
            c_a = torch.sigmoid(c_a)
            c_n = torch.sigmoid(c_n)
            
        return c_a, c_n


class ReliabilityGate(nn.Module):
    """
    Learnable gating network for reliability-aware fusion.
    
    The gate computes α_a = sigmoid(W_g [z_a; z_n; c_a; c_n] + b_g)
    
    This is a TRAINABLE mechanism - NOT hardcoded. The model learns
    when to trust audio vs network based on the joint representation.
    
    Key design principles:
    - No manual rules like "high packet loss → trust audio more"
    - The gate sees FULL context: both embeddings + reliability features
    - Gradient flows through gate to both encoders
    - Ablation mode allows testing fixed fusion baselines
    """
    
    def __init__(self, config: Gateway1Config):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = config.input_dim
        
        for i in range(config.num_layers):
            layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = config.hidden_dim
        
        # Final gate projection
        layers.append(nn.Linear(config.hidden_dim, 1))
        
        self.gate_network = nn.Sequential(*layers)
        
        if config.gate_type == "sigmoid":
            self.activation = nn.Sigmoid()
        elif config.gate_type == "softmax":
            self.activation = nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unknown gate_type: {config.gate_type}")
        
        # Initialize gate bias for balanced start
        if config.gate_bias_init != 0.0:
            with torch.no_grad():
                self.gate_network[-1].bias.fill_(config.gate_bias_init)
        
        # Statistics tracking
        self.register_buffer("gate_mean", torch.tensor(0.0))
        self.register_buffer("gate_std", torch.tensor(0.0))
        self.register_buffer("gate_min", torch.tensor(1.0))
        self.register_buffer("gate_max", torch.tensor(0.0))
        self.register_buffer("num_samples", torch.tensor(0))
        self._gate_history: List[torch.Tensor] = []
    
    def forward(
        self,
        z_a: torch.Tensor,
        z_n: torch.Tensor,
        c_a: Optional[torch.Tensor] = None,
        c_n: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute gate value α.
        
        Args:
            z_a: Audio embedding [B, D_a]
            z_n: Network embedding [B, D_n]
            c_a: Audio reliability context [B, C_a] or None
            c_n: Network reliability context [B, C_n] or None
            
        Returns:
            alpha: Gate value [B, 1] (weight for audio)
            alpha_raw: Pre-activation gate value [B, 1]
            info: Dictionary with gate statistics
        """
        batch_size = z_a.shape[0]
        
        # Handle ablation modes - fixed fusion baselines
        if self.config.ablation_mode == "fixed_audio":
            alpha = torch.ones(batch_size, 1, device=z_a.device, dtype=z_a.dtype)
            alpha_raw = torch.full_like(alpha, 10.0)
            info = {"ablation": "fixed_audio"}
            return alpha, alpha_raw, info
        elif self.config.ablation_mode == "fixed_network":
            alpha = torch.zeros(batch_size, 1, device=z_a.device, dtype=z_a.dtype)
            alpha_raw = torch.full_like(alpha, -10.0)
            info = {"ablation": "fixed_network"}
            return alpha, alpha_raw, info
        elif self.config.ablation_mode == "fixed_equal":
            alpha = torch.full((batch_size, 1), 0.5, device=z_a.device, dtype=z_a.dtype)
            alpha_raw = torch.zeros(batch_size, 1, device=z_a.device, dtype=z_a.dtype)
            info = {"ablation": "fixed_equal"}
            return alpha, alpha_raw, info
        
        gate_input = [z_a, z_n]
        
        if c_a is not None:
            gate_input.append(c_a)
        if c_n is not None:
            gate_input.append(c_n)
        
        gate_input = torch.cat(gate_input, dim=-1)
        
        alpha_raw = self.gate_network(gate_input)
        alpha = self.activation(alpha_raw)
        
        # Track statistics for logging/analysis
        info = {}
        if self.config.log_gate_stats and self.training:
            with torch.no_grad():
                self._gate_history.append(alpha.detach())
                if len(self._gate_history) > 1000:
                    self._gate_history.pop(0)
                
                all_gates = torch.cat(self._gate_history, dim=0)
                self.gate_mean = all_gates.mean()
                self.gate_std = all_gates.std()
                self.gate_min = all_gates.min()
                self.gate_max = all_gates.max()
                self.num_samples = torch.tensor(all_gates.shape[0])
                
                info["gate_mean"] = self.gate_mean.item()
                info["gate_std"] = self.gate_std.item()
                info["gate_min"] = self.gate_min.item()
                info["gate_max"] = self.gate_max.item()
                info["gate_histogram"] = torch.histc(all_gates, bins=10, min=0, max=1).cpu().tolist()
        
        info["alpha_raw_mean"] = alpha_raw.mean().item()
        info["alpha_raw_std"] = alpha_raw.std().item()
        
        return alpha, alpha_raw, info
    
    def get_gate_statistics(self) -> Dict[str, float]:
        """Return accumulated gate statistics."""
        return {
            "mean": self.gate_mean.item(),
            "std": self.gate_std.item(),
            "min": self.gate_min.item(),
            "max": self.gate_max.item(),
            "num_samples": self.num_samples.item(),
        }
    
    def reset_statistics(self) -> None:
        """Reset accumulated statistics."""
        self._gate_history.clear()
        self.gate_mean.zero_()
        self.gate_std.zero_()
        self.gate_min.fill_(1.0)
        self.gate_max.zero_()
        self.num_samples.zero_()


class Gateway1(nn.Module):
    """
    Gateway 1: Reliability-aware fusion of audio and network embeddings.
    
    Fusion: z_f = α_a z_a + (1 - α_a) z_n (+ optional residual)
    
    The gate learns to weight modalities based on their joint reliability
    without any hardcoded heuristics.
    
    Features:
    - Reliability-aware gating with learnable context extraction
    - Configurable gate architecture (depth, activation, dropout)
    - Optional residual fusion for gradient flow
    - Comprehensive logging and statistics
    - Ablation modes for fixed fusion baselines
    """
    
    def __init__(self, config: Gateway1Config):
        super().__init__()
        self.config = config
        
        # Reliability/context feature extractors
        self.reliability_extractor = ReliabilityFeatureExtractor(
            config.audio_dim,
            config.network_dim,
            config.hidden_dim,
            config.context_dim
        )
        
        # Main reliability gate
        self.reliability_gate = ReliabilityGate(config)
        
        # Optional residual fusion pathway
        if config.use_residual_fusion:
            self.residual_proj = nn.Sequential(
                nn.Linear(config.audio_dim + config.network_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.audio_dim),  # Project to audio dim for residual
            )
            self.residual_weight = config.residual_weight
        else:
            self.residual_proj = None
            self.residual_weight = 0.0
        
        # Audio/network reliability heads for auxiliary supervision
        if config.reliability_aware:
            self.audio_reliability_head = nn.Sequential(
                nn.Linear(config.audio_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
                nn.Sigmoid()
            )
            
            self.network_reliability_head = nn.Sequential(
                nn.Linear(config.network_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.audio_reliability_head = None
            self.network_reliability_head = None
    
    def forward(
        self,
        z_a: torch.Tensor,
        z_n: torch.Tensor,
        c_a: Optional[torch.Tensor] = None,
        c_n: Optional[torch.Tensor] = None,
        return_gate: bool = False,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with reliability-aware fusion.
        
        Args:
            z_a: Audio embedding [B, D_a]
            z_n: Network embedding [B, D_n]
            c_a: External audio context [B, C_a] or None
            c_n: External network context [B, C_n] or None
            return_gate: Whether to return gate value separately
            return_components: Whether to return individual components
            
        Returns:
            z_fused: Fused embedding [B, D_a]
            info: Dictionary with gate values, reliability scores, statistics
        """
        # Extract reliability features from embeddings (if not provided externally)
        if c_a is None or c_n is None:
            c_a_extracted, c_n_extracted = self.reliability_extractor(z_a, z_n)
            if c_a is None:
                c_a = c_a_extracted
            if c_n is None:
                c_n = c_n_extracted
        
        # Only pass context to gate if context_dim > 0
        gate_c_a = c_a if self.config.context_dim > 0 else None
        gate_c_n = c_n if self.config.context_dim > 0 else None
        
        # Compute gate
        alpha, alpha_raw, gate_info = self.reliability_gate(z_a, z_n, gate_c_a, gate_c_n)
        
        # Main fusion: z_f = α z_a + (1-α) z_n
        z_fused = alpha * z_a + (1 - alpha) * z_n
        
        # Optional residual fusion
        residual_info = {}
        if self.config.use_residual_fusion and self.residual_proj is not None:
            residual_input = torch.cat([z_a, z_n], dim=-1)
            residual = self.residual_proj(residual_input)
            z_fused = z_fused + self.residual_weight * residual
            residual_info = {
                "residual_norm": residual.norm(dim=-1).mean().item(),
                "residual_weight": self.residual_weight,
            }
        
        # Reliability predictions for auxiliary loss
        info = {
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "c_a": c_a,
            "c_n": c_n,
            **gate_info,
            **residual_info,
        }
        
        if self.config.reliability_aware:
            r_a = self.audio_reliability_head(z_a)
            r_n = self.network_reliability_head(z_n)
            info["reliability_audio"] = r_a
            info["reliability_network"] = r_n
        
        if return_components:
            info["z_a_weighted"] = alpha * z_a
            info["z_n_weighted"] = (1 - alpha) * z_n
            info["z_a"] = z_a
            info["z_n"] = z_n
        
        if return_gate:
            return z_fused, info, alpha
        
        return z_fused, info
    
    def get_gate_statistics(self) -> Dict[str, float]:
        """Get accumulated gate statistics for monitoring."""
        return self.reliability_gate.get_gate_statistics()
    
    def reset_gate_statistics(self) -> None:
        """Reset gate statistics."""
        self.reliability_gate.reset_statistics()
    
    def set_ablation_mode(self, mode: Optional[str]) -> None:
        """Set ablation mode for testing fixed fusion baselines."""
        self.config.ablation_mode = mode
        self.reliability_gate.config.ablation_mode = mode


class FixedFusionGateway1(nn.Module):
    """
    Ablation: Fixed fusion baselines for comparison.
    
    This module provides non-trainable fixed fusion strategies to compare
    against the learned gate. Use via config.ablation_mode or directly.
    """
    
    def __init__(self, mode: str = "equal"):
        super().__init__()
        self.mode = mode
        if mode not in ["audio_only", "network_only", "equal", "concat_mlp"]:
            raise ValueError(f"Unknown fixed fusion mode: {mode}")
        
        if mode == "concat_mlp":
            self.fusion = nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, 256),
            )
    
    def forward(
        self,
        z_a: torch.Tensor,
        z_n: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size = z_a.shape[0]
        device = z_a.device
        
        if self.mode == "audio_only":
            z_fused = z_a
            alpha = torch.ones(batch_size, 1, device=device)
        elif self.mode == "network_only":
            z_fused = z_n
            alpha = torch.zeros(batch_size, 1, device=device)
        elif self.mode == "equal":
            z_fused = 0.5 * z_a + 0.5 * z_n
            alpha = torch.full((batch_size, 1), 0.5, device=device)
        elif self.mode == "concat_mlp":
            z_fused = self.fusion(torch.cat([z_a, z_n], dim=-1))
            alpha = torch.full((batch_size, 1), 0.5, device=device)  # Not meaningful
        
        info = {
            "alpha": alpha,
            "ablation_mode": self.mode,
        }
        
        return z_fused, info