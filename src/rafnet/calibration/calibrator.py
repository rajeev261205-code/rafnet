import torch
import torch.nn as nn
import numpy as np
from typing import Literal, Optional, Dict, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass
class CalibrationConfig:
    method: Literal["temperature", "platt", "isotonic"] = "temperature"
    temperature_init: float = 1.0
    max_iter: int = 100
    lr: float = 0.01
    # Validation-based fitting
    fit_on_validation_only: bool = True
    # Metrics
    compute_ece: bool = True
    n_bins_ece: int = 10
    compute_brier: bool = True


class CalibrationMethod(ABC):
    """Abstract base class for calibration methods."""
    
    @abstractmethod
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "CalibrationMethod":
        pass
    
    @abstractmethod
    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        pass


class TemperatureScaling(nn.Module, CalibrationMethod):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))
        self._fitted = False
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)
    
    def fit(self, probs: np.ndarray, labels: np.ndarray, max_iter: int = 100, lr: float = 0.01) -> "TemperatureScaling":
        """Fit temperature on probabilities or logits.
        
        If input values are outside [0, 1], they are treated as logits.
        Otherwise, they are treated as probabilities and converted to logits.
        """
        # Convert to numpy if torch tensors
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Detect if input is logits (outside [0, 1] range) or probabilities
        if probs.min() < 0 or probs.max() > 1:
            # Input is already logits
            logits = torch.from_numpy(probs).float()
        else:
            # Input is probabilities, convert to logits
            logits = torch.log(torch.from_numpy(probs / (1 - probs + 1e-10) + 1e-10)).float()
        
        labels_t = torch.from_numpy(labels).float()
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nn.functional.binary_cross_entropy_with_logits(scaled_logits, labels_t)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        self._fitted = True
        return self
    
    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TemperatureScaling must be fitted before prediction")
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        logits = torch.log(torch.from_numpy(probs / (1 - probs + 1e-10) + 1e-10)).float()
        calibrated_logits = self.forward(logits)
        return torch.sigmoid(calibrated_logits).detach().numpy()
    
    def get_params(self) -> Dict:
        return {"temperature": self.temperature.item(), "method": "temperature"}
    
    @property
    def fitted(self) -> bool:
        return self._fitted


class PlattScaling(CalibrationMethod):
    def __init__(self):
        self.calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        self._fitted = False
    
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "PlattScaling":
        logits = np.log(probs / (1 - probs + 1e-10) + 1e-10)
        logits = logits.reshape(-1, 1)
        self.calibrator.fit(logits, labels)
        self._fitted = True
        return self
    
    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PlattScaling must be fitted before prediction")
        logits = np.log(probs / (1 - probs + 1e-10) + 1e-10)
        logits = logits.reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]
    
    def get_params(self) -> Dict:
        return {
            "coef": self.calibrator.coef_.tolist(),
            "intercept": self.calibrator.intercept_.tolist(),
            "method": "platt"
        }
    
    @property
    def fitted(self) -> bool:
        return self._fitted


class IsotonicCalibration(CalibrationMethod):
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False
    
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibration":
        self.calibrator.fit(probs, labels)
        self._fitted = True
        return self
    
    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibration must be fitted before prediction")
        return self.calibrator.predict(probs)
    
    def get_params(self) -> Dict:
        return {
            "x_thresholds": self.calibrator.X_thresholds_.tolist() if hasattr(self.calibrator, 'X_thresholds_') else [],
            "y_thresholds": self.calibrator.y_thresholds_.tolist() if hasattr(self.calibrator, 'y_thresholds_') else [],
            "method": "isotonic"
        }
    
    @property
    def fitted(self) -> bool:
        return self._fitted


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier Score."""
    return float(brier_score_loss(labels, probs))


def compute_reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict:
    """Compute reliability diagram data."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    results = {
        "bin_edges": bin_boundaries.tolist(),
        "bin_centers": ((bin_lowers + bin_uppers) / 2).tolist(),
        "bin_accuracies": [],
        "bin_confidences": [],
        "bin_counts": [],
    }
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        count = in_bin.sum()
        
        if count > 0:
            accuracy_in_bin = float(labels[in_bin].mean())
            avg_confidence_in_bin = float(probs[in_bin].mean())
        else:
            accuracy_in_bin = 0.0
            avg_confidence_in_bin = (bin_lower + bin_upper) / 2
        
        results["bin_accuracies"].append(accuracy_in_bin)
        results["bin_confidences"].append(avg_confidence_in_bin)
        results["bin_counts"].append(int(count))
    
    return results


class Calibrator(nn.Module):
    """
    Unified calibrator supporting multiple calibration methods.
    
    Key design: Fit ONLY on validation data to prevent leakage.
    """
    
    def __init__(self, config: CalibrationConfig):
        super().__init__()
        self.config = config
        self.method = config.method
        self._fitted = False
        self._calibration_method: Optional[CalibrationMethod] = None
        self._fit_stats: Dict = {}
        
        if config.method == "temperature":
            self._calibration_method = TemperatureScaling(config.temperature_init)
        elif config.method == "platt":
            self._calibration_method = PlattScaling()
        elif config.method == "isotonic":
            self._calibration_method = IsotonicCalibration()
        else:
            raise ValueError(f"Unknown calibration method: {config.method}")
    
    def fit(
        self, 
        probs: torch.Tensor, 
        labels: torch.Tensor,
        validation_probs: torch.Tensor,
        validation_labels: torch.Tensor
    ) -> "Calibrator":
        """
        Fit calibrator on validation data ONLY (prevents leakage).
        
        Validation data is REQUIRED. This prevents silent leakage from training data.
        
        Args:
            probs: Training probabilities [N] (not used for fitting, kept for API compatibility)
            labels: Training labels [N] (not used for fitting)
            validation_probs: Validation probabilities [M] - REQUIRED
            validation_labels: Validation labels [M] - REQUIRED
        """
        # Validation data is REQUIRED - no fallback to training data
        if validation_probs is None or validation_labels is None:
            raise ValueError(
                "Calibrator.fit() requires validation_probs and validation_labels. "
                "Fitting on training data causes leakage. Provide a separate validation set."
            )
        
        fit_probs = validation_probs.detach().cpu().numpy()
        fit_labels = validation_labels.detach().cpu().numpy()
        
        # Fit the calibration method
        self._calibration_method.fit(fit_probs, fit_labels)
        self._fitted = True
        
        # Compute calibration metrics on the fit data
        if self.config.compute_ece:
            calibrated_probs = self._calibration_method.predict_proba(fit_probs)
            self._fit_stats["ece"] = compute_ece(calibrated_probs, fit_labels, self.config.n_bins_ece)
        if self.config.compute_brier:
            calibrated_probs = self._calibration_method.predict_proba(fit_probs)
            self._fit_stats["brier"] = compute_brier(calibrated_probs, fit_labels)
        
        self._fit_stats["reliability_diagram"] = compute_reliability_diagram(
            self._calibration_method.predict_proba(fit_probs), fit_labels, self.config.n_bins_ece
        )
        self._fit_stats["method"] = self.method
        self._fit_stats["params"] = self._calibration_method.get_params()
        
        return self
    
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("Calibrator must be fitted before forward pass")
        
        probs_np = probs.detach().cpu().numpy()
        calibrated = self._calibration_method.predict_proba(probs_np)
        return torch.from_numpy(calibrated).float().to(probs.device)
    
    def calibrate(self, probs: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(probs)
    
    def get_fit_stats(self) -> Dict:
        """Return calibration fit statistics."""
        return self._fit_stats.copy()
    
    def get_temperature(self) -> Optional[float]:
        if self.method == "temperature" and self._fitted:
            return self._calibration_method.temperature.item()
        return None
    
    @property
    def fitted(self) -> bool:
        return self._fitted


def get_calibrator(config: CalibrationConfig) -> Calibrator:
    """Factory function to create calibrator."""
    return Calibrator(config)