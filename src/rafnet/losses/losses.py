import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class LossConfig:
    ce_weight: float = 1.0
    rel_weight: float = 0.5
    selcal_weight: float = 0.1
    agree_weight: float = 0.1


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        probs = probs.clamp(eps, 1 - eps)
        loss = - (labels * torch.log(probs) + (1 - labels) * torch.log(1 - probs))
        return self.weight * loss.mean()


class ReliabilityLoss(nn.Module):
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        r_a: torch.Tensor,
        r_n: torch.Tensor,
        z_a: torch.Tensor,
        z_n: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            audio_correct = (torch.sigmoid(z_a.norm(dim=-1, keepdim=True)) > 0.5).float() == labels
            network_correct = (torch.sigmoid(z_n.norm(dim=-1, keepdim=True)) > 0.5).float() == labels
        
        target_a = audio_correct.float()
        target_n = network_correct.float()
        
        loss_a = F.binary_cross_entropy(r_a, target_a)
        loss_n = F.binary_cross_entropy(r_n, target_n)
        
        gate_target = target_a / (target_a + target_n + 1e-7)
        gate_loss = F.binary_cross_entropy(alpha, gate_target)
        
        return self.weight * (loss_a + loss_n + gate_loss)


class SelectiveCalibrationLoss(nn.Module):
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        tau_low: float,
        tau_high: float
    ) -> torch.Tensor:
        certain_mask = (probs < tau_low) | (probs > tau_high)
        
        if certain_mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device)
        
        certain_probs = probs[certain_mask]
        certain_labels = labels[certain_mask]
        
        eps = 1e-7
        certain_probs = certain_probs.clamp(eps, 1 - eps)
        
        loss = - (certain_labels * torch.log(certain_probs) + 
                  (1 - certain_labels) * torch.log(1 - certain_probs))
        
        return self.weight * loss.mean()


class AgreementLoss(nn.Module):
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        p_c: torch.Tensor,
        p_q: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        disagreement = torch.abs(p_c - p_q)
        
        correct_c = (p_c > 0.5).float() == labels
        correct_q = (p_q > 0.5).float() == labels
        
        both_correct = (correct_c & correct_q).float()
        both_wrong = (~correct_c & ~correct_q).float()
        one_correct = (correct_c ^ correct_q).float()
        
        target_agreement = both_correct + 0.5 * one_correct
        
        loss = F.mse_loss(1 - disagreement, target_agreement)
        
        return self.weight * loss


class RAFNetLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
        self.ce_loss = CrossEntropyLoss(config.ce_weight)
        self.rel_loss = ReliabilityLoss(config.rel_weight)
        self.selcal_loss = SelectiveCalibrationLoss(config.selcal_weight)
        self.agree_loss = AgreementLoss(config.agree_weight)
    
    def forward(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        p_c: Optional[torch.Tensor] = None,
        p_q: Optional[torch.Tensor] = None,
        r_a: Optional[torch.Tensor] = None,
        r_n: Optional[torch.Tensor] = None,
        z_a: Optional[torch.Tensor] = None,
        z_n: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        tau_low: float = 0.3,
        tau_high: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        
        losses["ce"] = self.ce_loss(probs, labels)
        
        if self.config.rel_weight > 0 and all(v is not None for v in [r_a, r_n, z_a, z_n, alpha]):
            losses["reliability"] = self.rel_loss(r_a, r_n, z_a, z_n, labels, alpha)
        else:
            losses["reliability"] = torch.tensor(0.0, device=probs.device)
        
        if self.config.selcal_weight > 0:
            losses["selcal"] = self.selcal_loss(probs, labels, tau_low, tau_high)
        else:
            losses["selcal"] = torch.tensor(0.0, device=probs.device)
        
        if self.config.agree_weight > 0 and p_c is not None and p_q is not None:
            losses["agree"] = self.agree_loss(p_c, p_q, labels)
        else:
            losses["agree"] = torch.tensor(0.0, device=probs.device)
        
        losses["total"] = sum(losses.values())
        
        return losses