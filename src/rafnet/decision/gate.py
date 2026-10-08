import torch
import torch.nn as nn
import numpy as np
from typing import Literal, Tuple, Dict, Optional, List
from dataclasses import dataclass
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix
)


@dataclass
class DecisionGateConfig:
    tau_low: float = 0.3
    tau_high: float = 0.7
    optimize_thresholds: bool = True
    metric_for_optimization: Literal["f1", "precision", "recall", "balanced_accuracy", "mcc"] = "f1"
    grid_resolution: int = 100
    # Validation-based threshold selection (prevents leakage)
    select_on_validation: bool = True
    # Uncertainty region handling
    uncertainty_action: Literal["reject", "random", "classical", "quantum"] = "reject"
    # Constraints
    min_uncertainty_width: float = 0.1
    max_uncertainty_width: float = 0.5


class DecisionGate(nn.Module):
    """
    Decision gate with three-way classification: REAL, UNCERTAIN, SYNTHETIC.
    
    Decision:
        p < τ_low  → REAL (0)
        p > τ_high → SYNTHETIC (2)
        otherwise  → UNCERTAIN (1)
    
    Features:
    - Validation-based threshold selection (prevents leakage)
    - Multiple optimization metrics (F1, precision, recall, balanced_accuracy, MCC)
    - Configurable uncertainty region handling
    - Threshold constraints
    - Detailed decision statistics
    """
    
    def __init__(self, config: DecisionGateConfig):
        super().__init__()
        self.config = config
        self.register_buffer("tau_low", torch.tensor(config.tau_low))
        self.register_buffer("tau_high", torch.tensor(config.tau_high))
        self.optimized = False
        self._optimization_history: List[Dict] = []
    
    def forward(
        self, 
        probs: torch.Tensor, 
        return_uncertainty_mask: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply decision gate to probabilities.
        
        Args:
            probs: Calibrated probabilities [N] or [N, 1]
            return_uncertainty_mask: Whether to return uncertainty mask
            
        Returns:
            decisions: 0=REAL, 1=UNCERTAIN, 2=SYNTHETIC
            info: Dictionary with decision statistics
        """
        if probs.dim() > 1:
            probs = probs.squeeze(-1)
        
        decisions = torch.zeros_like(probs, dtype=torch.long)
        decisions[probs < self.tau_low] = 0  # REAL
        decisions[probs > self.tau_high] = 2  # SYNTHETIC
        decisions[(probs >= self.tau_low) & (probs <= self.tau_high)] = 1  # UNCERTAIN
        
        # Statistics
        n_total = len(decisions)
        n_real = (decisions == 0).sum().item()
        n_uncertain = (decisions == 1).sum().item()
        n_synthetic = (decisions == 2).sum().item()
        
        uncertainty_width = self.tau_high.item() - self.tau_low.item()
        
        info = {
            "tau_low": self.tau_low.item(),
            "tau_high": self.tau_high.item(),
            "uncertainty_width": uncertainty_width,
            "n_real": n_real,
            "n_uncertain": n_uncertain,
            "n_synthetic": n_synthetic,
            "pct_real": n_real / n_total if n_total > 0 else 0,
            "pct_uncertain": n_uncertain / n_total if n_total > 0 else 0,
            "pct_synthetic": n_synthetic / n_total if n_total > 0 else 0,
            "optimized": self.optimized,
        }
        
        if return_uncertainty_mask:
            info["uncertainty_mask"] = (decisions == 1)
            info["certain_mask"] = (decisions != 1)
        
        return decisions, info
    
    def optimize_thresholds(
        self, 
        probs: torch.Tensor, 
        labels: torch.Tensor,
        validation_probs: torch.Tensor,
        validation_labels: torch.Tensor
    ) -> "DecisionGate":
        """
        Optimize tau_low and tau_high on validation data ONLY (prevents leakage).
        
        Validation data is REQUIRED when optimize_thresholds=True.
        This prevents threshold overfitting to test data.
        
        Args:
            probs: Training probabilities [N] (not used for optimization)
            labels: Training labels [N] (not used for optimization)
            validation_probs: Validation probabilities [M] - REQUIRED
            validation_labels: Validation labels [M] - REQUIRED
        """
        if not self.config.optimize_thresholds:
            return self
        
        # Validation data is REQUIRED - no fallback to training data
        if validation_probs is None or validation_labels is None:
            raise ValueError(
                "DecisionGate.optimize_thresholds() requires validation_probs and validation_labels "
                "when optimize_thresholds=True. Optimizing on test data causes leakage. "
                "Provide a separate validation set or set optimize_thresholds=False."
            )
        
        opt_probs = validation_probs.detach().cpu().numpy()
        opt_labels = validation_labels.detach().cpu().numpy()
        
        # Enforce constraint: tau_high - tau_low >= min_uncertainty_width
        min_width = self.config.min_uncertainty_width
        max_width = self.config.max_uncertainty_width
        
        best_score = -1
        best_tau_low = self.config.tau_low
        best_tau_high = self.config.tau_high
        
        tau_low_candidates = np.linspace(0.05, 0.45, self.config.grid_resolution)
        tau_high_candidates = np.linspace(0.55, 0.95, self.config.grid_resolution)
        
        for tau_low in tau_low_candidates:
            # tau_high must be > tau_low + min_width
            valid_high = tau_high_candidates[
                (tau_high_candidates > tau_low + min_width) & 
                (tau_high_candidates <= tau_low + max_width)
            ]
            if len(valid_high) == 0:
                continue
                
            for tau_high in valid_high:
                decisions = np.zeros_like(opt_labels)
                decisions[opt_probs < tau_low] = 0
                decisions[opt_probs > tau_high] = 2
                decisions[(opt_probs >= tau_low) & (opt_probs <= tau_high)] = 1
                
                score = self._compute_score(opt_labels, decisions)
                
                if score > best_score:
                    best_score = score
                    best_tau_low = tau_low
                    best_tau_high = tau_high
        
        self.tau_low = torch.tensor(best_tau_low)
        self.tau_high = torch.tensor(best_tau_high)
        self.optimized = True
        
        self._optimization_history.append({
            "best_score": best_score,
            "tau_low": best_tau_low,
            "tau_high": best_tau_high,
            "metric": self.config.metric_for_optimization,
        })
        
        return self
    
    def _compute_score(self, labels: np.ndarray, decisions: np.ndarray) -> float:
        """Compute optimization metric."""
        metric = self.config.metric_for_optimization
        
        if metric == "f1":
            return float(f1_score(labels, decisions, average="macro", zero_division=0))
        elif metric == "precision":
            return float(precision_score(labels, decisions, average="macro", zero_division=0))
        elif metric == "recall":
            return float(recall_score(labels, decisions, average="macro", zero_division=0))
        elif metric == "balanced_accuracy":
            return float(balanced_accuracy_score(labels, decisions))
        elif metric == "mcc":
            return float(matthews_corrcoef(labels, decisions))
        else:
            return float(f1_score(labels, decisions, average="macro", zero_division=0))
    
    def get_thresholds(self) -> Tuple[float, float]:
        return self.tau_low.item(), self.tau_high.item()
    
    def set_thresholds(self, tau_low: float, tau_high: float) -> "DecisionGate":
        if tau_low >= tau_high:
            raise ValueError("tau_low must be < tau_high")
        self.tau_low = torch.tensor(tau_low)
        self.tau_high = torch.tensor(tau_high)
        return self
    
    def get_optimization_history(self) -> List[Dict]:
        return self._optimization_history.copy()
    
    def decide_with_fallback(
        self,
        probs: torch.Tensor,
        classical_probs: Optional[torch.Tensor] = None,
        quantum_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Make decisions with fallback for uncertain samples.
        
        Args:
            probs: Calibrated probabilities
            classical_probs: Classical branch probabilities (for fallback)
            quantum_probs: Quantum branch probabilities (for fallback)
        """
        decisions, info = self.forward(probs, return_uncertainty_mask=True)
        
        uncertain_mask = info["uncertainty_mask"]
        n_uncertain = uncertain_mask.sum().item()
        
        if n_uncertain > 0:
            if self.config.uncertainty_action == "reject":
                # Keep as UNCERTAIN (1)
                pass
            elif self.config.uncertainty_action == "random":
                decisions[uncertain_mask] = torch.randint(0, 3, (n_uncertain,))
            elif self.config.uncertainty_action == "classical" and classical_probs is not None:
                # Use classical branch decision
                class_decisions = torch.zeros_like(classical_probs, dtype=torch.long)
                class_decisions[classical_probs < self.tau_low] = 0
                class_decisions[classical_probs > self.tau_high] = 2
                class_decisions[(classical_probs >= self.tau_low) & (classical_probs <= self.tau_high)] = 1
                decisions[uncertain_mask] = class_decisions[uncertain_mask]
            elif self.config.uncertainty_action == "quantum" and quantum_probs is not None:
                # Use quantum branch decision
                quantum_decisions = torch.zeros_like(quantum_probs, dtype=torch.long)
                quantum_decisions[quantum_probs < self.tau_low] = 0
                quantum_decisions[quantum_probs > self.tau_high] = 2
                quantum_decisions[(quantum_probs >= self.tau_low) & (quantum_probs <= self.tau_high)] = 1
                decisions[uncertain_mask] = quantum_decisions[uncertain_mask]
        
        info["fallback_used"] = n_uncertain > 0 and self.config.uncertainty_action != "reject"
        info["fallback_action"] = self.config.uncertainty_action
        
        return decisions, info