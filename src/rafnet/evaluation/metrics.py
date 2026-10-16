import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d


@dataclass
class EvaluationConfig:
    metrics: List[str] = None
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "accuracy", "precision", "recall", "f1", "roc_auc", "eer",
                "fpr", "fnr", "ece", "brier", "risk_coverage", "latency", "throughput"
            ]


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_score)
    
    def f(x):
        return interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(0, 1))(x) - (1 - x)
    
    try:
        eer = brentq(f, 0, 1)
    except ValueError:
        eer = 0.5
    
    return float(eer)


def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sort by score descending to compute proper ROC curve
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_score_sorted = y_score[sorted_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    # Add threshold beyond max score to get FPR=0, TPR=0
    thresholds = np.concatenate([[np.inf], y_score_sorted])
    
    fpr = []
    tpr = []
    
    tp = 0
    fp = 0
    
    for thresh in thresholds:
        if thresh != np.inf:
            # Update TP and FP for this threshold
            mask = y_score_sorted >= thresh
            tp = np.sum(y_true_sorted[mask])
            fp = np.sum(1 - y_true_sorted[mask])
        
        fpr.append(fp / n_neg if n_neg > 0 else 0)
        tpr.append(tp / n_pos if n_pos > 0 else 0)
    
    return np.array(fpr), np.array(tpr), thresholds


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score."""
    return float(brier_score_loss(y_true, y_prob))


def compute_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
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
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        count = in_bin.sum()
        
        if count > 0:
            accuracy_in_bin = float(y_true[in_bin].mean())
            avg_confidence_in_bin = float(y_prob[in_bin].mean())
        else:
            accuracy_in_bin = 0.0
            avg_confidence_in_bin = (bin_lower + bin_upper) / 2
        
        results["bin_accuracies"].append(accuracy_in_bin)
        results["bin_confidences"].append(avg_confidence_in_bin)
        results["bin_counts"].append(int(count))
    
    return results


def compute_risk_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decisions: np.ndarray,
    coverages: List[float] = None,
    use_confidence_ordering: bool = True
) -> Dict[float, float]:
    """
    Compute risk-coverage curve.
    
    Risk-coverage measures the error rate (risk) as a function of the fraction
    of samples we make predictions on (coverage). We sort samples by confidence
    and evaluate risk at different coverage levels.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        decisions: Decision gate outputs (0=REAL, 1=UNCERTAIN, 2=SYNTHETIC)
        coverages: Target coverage levels
        use_confidence_ordering: If True, order by |p-0.5| (confidence)
    
    Returns:
        Dict mapping coverage -> risk (error rate)
    """
    if coverages is None:
        coverages = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    n_samples = len(y_true)
    
    # For risk-coverage, we only consider certain decisions (not UNCERTAIN)
    certain_mask = (decisions != 1)
    if not certain_mask.any():
        return {c: float('nan') for c in coverages}
    
    # Order by confidence (distance from 0.5)
    certain_probs = y_prob[certain_mask]
    certain_labels = y_true[certain_mask]
    
    if use_confidence_ordering:
        confidence = np.abs(certain_probs - 0.5)
        order = np.argsort(confidence)[::-1]  # Most confident first
    else:
        order = np.arange(len(certain_labels))
    
    ordered_labels = certain_labels[order]
    ordered_probs = certain_probs[order]
    n_certain = len(ordered_labels)
    
    results = {}
    for c in coverages:
        if c <= 1.0:
            n_select = int(c * n_samples)
            # But we can only select up to n_certain
            n_select = min(n_select, n_certain)
            if n_select > 0:
                selected_labels = ordered_labels[:n_select]
                selected_probs = ordered_probs[:n_select]
                preds = (selected_probs > 0.5).astype(int)
                risk = (selected_labels != preds).mean()
                results[c] = float(risk)
            else:
                results[c] = float('nan')
        else:
            results[c] = float('nan')
    
    return results


def compute_coverage_error_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decisions: np.ndarray,
    n_points: int = 20
) -> Dict[str, List[float]]:
    """
    Compute full coverage-error curve.
    
    Returns coverage and error arrays for plotting.
    """
    certain_mask = (decisions != 1)
    if not certain_mask.any():
        return {"coverage": [], "error": []}
    
    certain_probs = y_prob[certain_mask]
    certain_labels = y_true[certain_mask]
    
    # Order by confidence
    confidence = np.abs(certain_probs - 0.5)
    order = np.argsort(confidence)[::-1]
    
    ordered_labels = certain_labels[order]
    ordered_probs = certain_probs[order]
    n_certain = len(ordered_labels)
    
    coverages = []
    errors = []
    
    for i in range(1, n_certain + 1):
        coverage = i / len(y_true)
        preds = (ordered_probs[:i] > 0.5).astype(int)
        error = (ordered_labels[:i] != preds).mean()
        coverages.append(coverage)
        errors.append(float(error))
    
    return {"coverage": coverages, "error": errors}


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Compute comprehensive calibration metrics."""
    ece = compute_ece(y_true, y_prob, n_bins)
    brier = compute_brier(y_true, y_prob)
    
    # Maximum Calibration Error (MCE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        if in_bin.any():
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier),
    }


def compute_latency_throughput(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    n_runs: int = 100,
    warmup: int = 10,
    device: str = "cpu"
) -> Dict[str, float]:
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(input_shape).to(device)
    
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    import time
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_time = end - start
    latency_ms = (total_time / n_runs) * 1000
    throughput = n_runs / total_time
    
    return {
        "latency_ms": latency_ms,
        "throughput_samples_per_sec": throughput,
    }


def compute_model_size(model: torch.nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 ** 2),
    }


def evaluate_all(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray = None,
    decisions: np.ndarray = None,
    config: EvaluationConfig = None,
    model: torch.nn.Module = None,
    input_shape: Tuple[int, ...] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    if config is None:
        config = EvaluationConfig()
    
    if y_pred is None:
        y_pred = (y_prob > 0.5).astype(int)
    
    results = {}
    
    if "accuracy" in config.metrics:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    if "precision" in config.metrics:
        results["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    
    if "recall" in config.metrics:
        results["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    
    if "f1" in config.metrics:
        results["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    
    if "roc_auc" in config.metrics:
        try:
            results["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            results["roc_auc"] = 0.5
    
    if "eer" in config.metrics:
        results["eer"] = compute_eer(y_true, y_prob)
    
    if "fpr" in config.metrics:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    if "fnr" in config.metrics:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    if "ece" in config.metrics:
        results["ece"] = compute_ece(y_true, y_prob)
    
    if "mce" in config.metrics:
        results["mce"] = compute_calibration_metrics(y_true, y_prob)["mce"]
    
    if "brier" in config.metrics:
        results["brier"] = float(brier_score_loss(y_true, y_prob))
    
    if "risk_coverage" in config.metrics and decisions is not None:
        rc = compute_risk_coverage(y_true, y_prob, decisions)
        for c, r in rc.items():
            results[f"risk_coverage_{int(c*100)}"] = r
    
    if "coverage_error_curve" in config.metrics and decisions is not None:
        cec = compute_coverage_error_curve(y_true, y_prob, decisions)
        results["coverage_error_curve"] = cec
    
    if "latency" in config.metrics and model is not None and input_shape is not None:
        lt = compute_latency_throughput(model, input_shape, device=device)
        results["latency_ms"] = lt["latency_ms"]
        results["throughput"] = lt["throughput_samples_per_sec"]
    
    if "throughput" in config.metrics and model is not None and input_shape is not None:
        if "throughput" not in results:
            lt = compute_latency_throughput(model, input_shape, device=device)
            results["throughput"] = lt["throughput_samples_per_sec"]
    
    return results


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray = None,
    decisions: np.ndarray = None,
    config: EvaluationConfig = None,
    n_bootstrap: int = None,
    confidence_level: float = None,
) -> Dict[str, Tuple[float, float]]:
    if config is None:
        config = EvaluationConfig()
    
    if n_bootstrap is None:
        n_bootstrap = config.n_bootstrap
    if confidence_level is None:
        confidence_level = config.confidence_level
    
    n_samples = len(y_true)
    alpha = (1 - confidence_level) / 2
    
    metric_values = {metric: [] for metric in config.metrics}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_t = y_true[indices]
        y_p = y_prob[indices]
        y_d = decisions[indices] if decisions is not None else None
        y_pr = (y_p > 0.5).astype(int) if y_pred is None else y_pred[indices]
        
        metrics = evaluate_all(y_t, y_p, y_pr, y_d, config)
        
        for metric, value in metrics.items():
            if not np.isnan(value):
                metric_values[metric].append(value)
    
    results = {}
    for metric, values in metric_values.items():
        if len(values) > 0:
            lower = np.percentile(values, 100 * alpha)
            upper = np.percentile(values, 100 * (1 - alpha))
            results[metric] = (float(lower), float(upper))
        else:
            results[metric] = (np.nan, np.nan)
    
    return results