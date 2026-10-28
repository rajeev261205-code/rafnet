import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import logging

from rafnet.config.schema import Gateway2Config

logger = logging.getLogger(__name__)


class Gateway2(nn.Module):
    """
    Gateway 2: Reliability-aware fusion of classical and quantum branch probabilities.
    
    Fusion: p_fused = β p_c + (1 - β) p_q
    
    Gate: β = sigmoid(W_2 [p_c; p_q; |p_c - p_q|] + b_2)
    
    This is a TRAINABLE mechanism - NOT hardcoded. The model learns
    when to trust classical vs quantum based on their outputs and disagreement.
    
    Features:
    - Configurable gate architecture (depth, activation, dropout)
    - Branch disable support (classical-only, quantum-only)
    - Ablation modes for fixed fusion baselines
    - Comprehensive logging and statistics
    - Numerical boundary handling
    """
    
    def __init__(self, config: Gateway2Config):
        super().__init__()
        self.config = config
        
        # Build gate network
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
        self._beta_c_history: List[torch.Tensor] = []  # beta when classical favored
        self._beta_q_history: List[torch.Tensor] = []  # beta when quantum favored
    
    def forward(
        self,
        p_c: torch.Tensor,
        p_q: torch.Tensor,
        return_gate: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute fused probability.
        
        Args:
            p_c: Classical branch probability [B] or [B, 1]
            p_q: Quantum branch probability [B] or [B, 1]
            return_gate: Whether to return gate value separately
            
        Returns:
            p_fused: Fused probability [B]
            info: Dictionary with beta, disagreement, p_c, p_q, statistics
        """
        # Ensure 2D
        if p_c.dim() == 1:
            p_c = p_c.unsqueeze(-1)
        if p_q.dim() == 1:
            p_q = p_q.unsqueeze(-1)
        
        batch_size = p_c.shape[0]
        device = p_c.device
        dtype = p_c.dtype
        
        # Clamp probabilities for numerical stability
        p_c_clamped = p_c.clamp(0.0, 1.0)
        p_q_clamped = p_q.clamp(0.0, 1.0)
        
        # Handle branch disabling
        if not self.config.enable_classical and not self.config.enable_quantum:
            raise ValueError("At least one branch must be enabled")
        
        if not self.config.enable_classical:
            # Classical disabled - trust quantum
            p_fused = p_q_clamped.squeeze(-1)
            beta = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            beta_raw = torch.full((batch_size, 1), -10.0, device=device, dtype=dtype)
            info = self._build_info(beta, beta_raw, p_c_clamped, p_q_clamped, {"classical_disabled": True})
            return (p_fused, info, beta) if return_gate else (p_fused, info)
        
        if not self.config.enable_quantum:
            # Quantum disabled - trust classical
            p_fused = p_c_clamped.squeeze(-1)
            beta = torch.ones(batch_size, 1, device=device, dtype=dtype)
            beta_raw = torch.full((batch_size, 1), 10.0, device=device, dtype=dtype)
            info = self._build_info(beta, beta_raw, p_c_clamped, p_q_clamped, {"quantum_disabled": True})
            return (p_fused, info, beta) if return_gate else (p_fused, info)
        
        # Handle ablation modes
        if self.config.ablation_mode == "fixed_classical":
            p_fused = p_c_clamped.squeeze(-1)
            beta = torch.ones(batch_size, 1, device=device, dtype=dtype)
            info = self._build_info(beta, torch.full_like(beta, 10.0), p_c_clamped, p_q_clamped, {"ablation": "fixed_classical"})
            return (p_fused, info, beta) if return_gate else (p_fused, info)
        
        if self.config.ablation_mode == "fixed_quantum":
            p_fused = p_q_clamped.squeeze(-1)
            beta = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            info = self._build_info(beta, torch.full_like(beta, -10.0), p_c_clamped, p_q_clamped, {"ablation": "fixed_quantum"})
            return (p_fused, info, beta) if return_gate else (p_fused, info)
        
        if self.config.ablation_mode == "fixed_equal":
            p_fused = 0.5 * (p_c_clamped + p_q_clamped).squeeze(-1)
            beta = torch.full((batch_size, 1), 0.5, device=device, dtype=dtype)
            info = self._build_info(beta, torch.zeros_like(beta), p_c_clamped, p_q_clamped, {"ablation": "fixed_equal"})
            return (p_fused, info, beta) if return_gate else (p_fused, info)
        
        # Compute features: [p_c, p_q, |p_c - p_q|]
        diff = torch.abs(p_c_clamped - p_q_clamped)
        gate_input = torch.cat([p_c_clamped, p_q_clamped, diff], dim=-1)
        
        # Compute gate
        beta_raw = self.gate_network(gate_input)
        beta = self.activation(beta_raw)
        
        # Fusion: p_fused = β p_c + (1 - β) p_q
        p_fused = (beta * p_c_clamped + (1 - beta) * p_q_clamped).squeeze(-1)
        
        # Build info dict using clamped values
        info = self._build_info(beta, beta_raw, p_c_clamped, p_q_clamped, {})
        
        # Track statistics
        if self.config.log_gate_stats and self.training:
            self._update_statistics(beta, beta_raw, p_c_clamped, p_q_clamped)
            info.update(self._get_statistics())
        
        if return_gate:
            return p_fused, info, beta
        
        return p_fused, info
    
    def _build_info(
        self, 
        beta: torch.Tensor, 
        beta_raw: torch.Tensor,
        p_c: torch.Tensor,
        p_q: torch.Tensor,
        extra: Dict
    ) -> Dict:
        """Build info dictionary with all outputs."""
        diff = torch.abs(p_c - p_q)
        
        info = {
            "beta": beta.squeeze(-1),
            "beta_raw": beta_raw.squeeze(-1),
            "disagreement": diff.squeeze(-1),
            "p_c": p_c.squeeze(-1),
            "p_q": p_q.squeeze(-1),
            "beta_raw_mean": beta_raw.mean().item(),
            # Single-sample batches have no meaningful std (and torch warns
            # about degrees of freedom <= 0) — report 0.0 instead of NaN.
            "beta_raw_std": beta_raw.std().item() if beta_raw.numel() > 1 else 0.0,
            **extra,
        }
        return info
    
    def _update_statistics(
        self, 
        beta: torch.Tensor, 
        beta_raw: torch.Tensor,
        p_c: torch.Tensor,
        p_q: torch.Tensor
    ) -> None:
        """Update running statistics for monitoring."""
        with torch.no_grad():
            self._gate_history.append(beta.detach())
            if len(self._gate_history) > 1000:
                self._gate_history.pop(0)
            
            all_gates = torch.cat(self._gate_history, dim=0)
            self.gate_mean = all_gates.mean()
            self.gate_std = all_gates.std()
            self.gate_min = all_gates.min()
            self.gate_max = all_gates.max()
            self.num_samples = torch.tensor(all_gates.shape[0])
            
            # Track beta when classical vs quantum is more confident
            mask_c = (p_c > p_q).squeeze(-1)
            mask_q = (p_q > p_c).squeeze(-1)
            if mask_c.any():
                self._beta_c_history.append(beta[mask_c].detach())
            if mask_q.any():
                self._beta_q_history.append(beta[mask_q].detach())
    
    def _get_statistics(self) -> Dict:
        """Get current statistics for logging."""
        stats = {
            "gate_mean": self.gate_mean.item(),
            "gate_std": self.gate_std.item(),
            "gate_min": self.gate_min.item(),
            "gate_max": self.gate_max.item(),
            "num_samples": self.num_samples.item(),
        }
        
        # Histogram
        if len(self._gate_history) > 0:
            all_gates = torch.cat(self._gate_history, dim=0)
            stats["gate_histogram"] = torch.histc(all_gates, bins=10, min=0, max=1).cpu().tolist()
        
        # Conditional statistics
        if len(self._beta_c_history) > 0:
            beta_c = torch.cat(self._beta_c_history, dim=0)
            stats["beta_when_c_confident_mean"] = beta_c.mean().item()
            stats["beta_when_c_confident_std"] = beta_c.std().item()
        if len(self._beta_q_history) > 0:
            beta_q = torch.cat(self._beta_q_history, dim=0)
            stats["beta_when_q_confident_mean"] = beta_q.mean().item()
            stats["beta_when_q_confident_std"] = beta_q.std().item()
        
        return stats
    
    def get_gate_statistics(self) -> Dict[str, float]:
        """Return accumulated gate statistics for monitoring."""
        return {
            "mean": self.gate_mean.item(),
            "std": self.gate_std.item(),
            "min": self.gate_min.item(),
            "max": self.gate_max.item(),
            "num_samples": self.num_samples.item(),
            "beta_when_c_confident_mean": (
                torch.cat(self._beta_c_history).mean().item() 
                if self._beta_c_history else 0.0
            ),
            "beta_when_q_confident_mean": (
                torch.cat(self._beta_q_history).mean().item() 
                if self._beta_q_history else 0.0
            ),
        }
    
    def reset_statistics(self) -> None:
        """Reset accumulated statistics."""
        self._gate_history.clear()
        self._beta_c_history.clear()
        self._beta_q_history.clear()
        self.gate_mean.zero_()
        self.gate_std.zero_()
        self.gate_min.fill_(1.0)
        self.gate_max.zero_()
        self.num_samples.zero_()
    
    def set_ablation_mode(self, mode: Optional[str]) -> None:
        """Set ablation mode for testing fixed fusion baselines."""
        self.config.ablation_mode = mode
    
    def enable_classical_branch(self, enable: bool = True) -> None:
        """Enable/disable classical branch."""
        self.config.enable_classical = enable
    
    def enable_quantum_branch(self, enable: bool = True) -> None:
        """Enable/disable quantum branch."""
        self.config.enable_quantum = enable


class FixedFusionGateway2(nn.Module):
    """
    Ablation: Fixed fusion baselines for Gateway 2 comparison.
    """
    
    def __init__(self, mode: str = "equal"):
        super().__init__()
        self.mode = mode
        if mode not in ["classical_only", "quantum_only", "equal", "max", "min", "concat_mlp"]:
            raise ValueError(f"Unknown fixed fusion mode: {mode}")
        
        if mode == "concat_mlp":
            self.fusion = nn.Sequential(
                nn.Linear(2, 16),
                nn.LayerNorm(16),
                nn.GELU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        p_c: torch.Tensor,
        p_q: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        if p_c.dim() == 1:
            p_c = p_c.unsqueeze(-1)
        if p_q.dim() == 1:
            p_q = p_q.unsqueeze(-1)
        
        batch_size = p_c.shape[0]
        device = p_c.device
        
        if self.mode == "classical_only":
            p_fused = p_c.squeeze(-1)
            beta = torch.ones(batch_size, 1, device=device)
        elif self.mode == "quantum_only":
            p_fused = p_q.squeeze(-1)
            beta = torch.zeros(batch_size, 1, device=device)
        elif self.mode == "equal":
            p_fused = 0.5 * (p_c + p_q).squeeze(-1)
            beta = torch.full((batch_size, 1), 0.5, device=device)
        elif self.mode == "max":
            p_fused = torch.maximum(p_c, p_q).squeeze(-1)
            beta = (p_c >= p_q).float()
        elif self.mode == "min":
            p_fused = torch.minimum(p_c, p_q).squeeze(-1)
            beta = (p_c <= p_q).float()
        elif self.mode == "concat_mlp":
            p_fused = self.fusion(torch.cat([p_c, p_q], dim=-1)).squeeze(-1)
            beta = torch.full((batch_size, 1), 0.5, device=device)
        
        diff = torch.abs(p_c - p_q)
        
        info = {
            "beta": beta.squeeze(-1),
            "disagreement": diff.squeeze(-1),
            "p_c": p_c.squeeze(-1),
            "p_q": p_q.squeeze(-1),
            "ablation_mode": self.mode,
        }
        
        return p_fused, info