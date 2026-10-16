import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Literal, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import pickle
import os
import time
import json
from pathlib import Path
from contextlib import contextmanager

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)


@dataclass
class ClassicalConfig:
    model_type: Literal["xgboost", "logistic", "svm", "random_forest", "mlp"] = "xgboost"
    xgboost_params: Dict[str, Any] = None
    use_calibration: bool = True
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    random_state: int = 42
    n_jobs: int = -1
    measure_latency: bool = True
    calibration_cv: int = 3
    eval_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "roc_auc", "logloss"
    ])
    serialization_format: Literal["pickle", "joblib"] = "pickle"
    
    def __post_init__(self):
        if self.xgboost_params is None:
            self.xgboost_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "eval_metric": "logloss",
            }


class LatencyTracker:
    """Track latency statistics for inference."""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.batch_sizes: List[int] = []
    
    def record(self, latency_ms: float, batch_size: int) -> None:
        self.latencies.append(latency_ms)
        self.batch_sizes.append(batch_size)
    
    def get_stats(self) -> Dict[str, float]:
        if not self.latencies:
            return {}
        latencies = np.array(self.latencies)
        return {
            "mean_ms": float(latencies.mean()),
            "std_ms": float(latencies.std()),
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "total_samples": len(latencies),
            "total_time_ms": float(latencies.sum()),
        }
    
    def reset(self) -> None:
        self.latencies.clear()
        self.batch_sizes.clear()


class BaseClassicalModel(ABC):
    """Abstract base class for classical ML models."""
    
    def __init__(self):
        self.calibrated_model = None
        self.latency_tracker = LatencyTracker()
        self._is_calibrated = False
        self._feature_names: Optional[List[str]] = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassicalModel":
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def calibrate(self, X: np.ndarray, y: np.ndarray, method: str = "sigmoid", cv: int = 3) -> None:
        """Calibrate probabilities using CalibratedClassifierCV."""
        self.calibrated_model = CalibratedClassifierCV(
            clone(self.model), method=method, cv=cv
        )
        self.calibrated_model.fit(X, y)
        self._is_calibrated = True
    
    @contextmanager
    def measure_latency(self, batch_size: int = 1):
        """Context manager to measure inference latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.latency_tracker.record(elapsed_ms, batch_size)
    
    def get_latency_stats(self) -> Dict[str, float]:
        return self.latency_tracker.get_stats()
    
    def reset_latency_stats(self) -> None:
        self.latency_tracker.reset()
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate model on given data."""
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "logloss"]
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = float(accuracy_score(y, y_pred))
            elif metric == "precision":
                results[metric] = float(precision_score(y, y_pred, zero_division=0))
            elif metric == "recall":
                results[metric] = float(recall_score(y, y_pred, zero_division=0))
            elif metric == "f1":
                results[metric] = float(f1_score(y, y_pred, zero_division=0))
            elif metric == "roc_auc":
                try:
                    results[metric] = float(roc_auc_score(y, y_proba))
                except ValueError:
                    results[metric] = float("nan")
            elif metric == "logloss":
                try:
                    results[metric] = float(log_loss(y, y_proba))
                except ValueError:
                    results[metric] = float("nan")
            else:
                results[metric] = float("nan")
        
        return results
    
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_dict = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "_is_calibrated": self._is_calibrated,
            "_feature_names": self._feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, path: str) -> "BaseClassicalModel":
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        instance = cls.__new__(cls)
        instance.model = save_dict["model"]
        instance.calibrated_model = save_dict.get("calibrated_model")
        instance._is_calibrated = save_dict.get("_is_calibrated", False)
        instance._feature_names = save_dict.get("_feature_names")
        instance.latency_tracker = LatencyTracker()
        return instance
    
    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()
    
    def set_feature_names(self, names: List[str]) -> None:
        self._feature_names = names
    
    def get_feature_names(self) -> Optional[List[str]]:
        return self._feature_names
    
    def is_calibrated(self) -> bool:
        return self._is_calibrated


class XGBoostModel(BaseClassicalModel):
    def __init__(self, params: Dict[str, Any]):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        super().__init__()
        self.params = params
        self.model = xgb.XGBClassifier(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with self.measure_latency(batch_size=X.shape[0]):
            if self.calibrated_model is not None:
                return self.calibrated_model.predict_proba(X)
            return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class LogisticRegressionModel(BaseClassicalModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with self.measure_latency(batch_size=X.shape[0]):
            if self.calibrated_model is not None:
                return self.calibrated_model.predict_proba(X)
            return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class SVMModel(BaseClassicalModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = SVC(probability=True, random_state=42, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with self.measure_latency(batch_size=X.shape[0]):
            if self.calibrated_model is not None:
                return self.calibrated_model.predict_proba(X)
            return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestModel(BaseClassicalModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with self.measure_latency(batch_size=X.shape[0]):
            if self.calibrated_model is not None:
                return self.calibrated_model.predict_proba(X)
            return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MLPModel(BaseClassicalModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = MLPClassifier(
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPModel":
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with self.measure_latency(batch_size=X.shape[0]):
            if self.calibrated_model is not None:
                return self.calibrated_model.predict_proba(X)
            return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
    "logistic": LogisticRegressionModel,
    "svm": SVMModel,
    "random_forest": RandomForestModel,
    "mlp": MLPModel,
}


def create_classical_model(config: ClassicalConfig) -> BaseClassicalModel:
    model_class = MODEL_REGISTRY.get(config.model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    if config.model_type == "xgboost":
        return model_class(config.xgboost_params)
    else:
        return model_class()


class ClassicalBranch(nn.Module):
    """
    Unified classical ML branch for RAFNet.
    
    Features:
    - Multiple model types (XGBoost, Logistic, SVM, RF, MLP)
    - Reproducible training with fixed random seeds
    - Probability calibration (sigmoid/isotonic)
    - Latency measurement
    - Evaluation metrics
    - Serialization (pickle)
    - Probability prediction interface
    """
    
    def __init__(self, config: ClassicalConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.model = create_classical_model(config)
        self.is_fitted = False
        self._training_history: List[Dict] = []
        self._eval_history: List[Dict] = []
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> "ClassicalBranch":
        """Fit the classical model on features from Gateway 1."""
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # Set feature names if available
        if hasattr(self, "_feature_names"):
            self.model.set_feature_names(self._feature_names)
        
        self.model.fit(X_np, y_np)
        self.is_fitted = True
        
        # Record training info
        self._training_history.append({
            "n_samples": X_np.shape[0],
            "n_features": X_np.shape[1],
            "class_distribution": np.bincount(y_np.astype(np.int64)).tolist(),
        })
        
        return self
    
    def calibrate(self, X: torch.Tensor, y: torch.Tensor) -> "ClassicalBranch":
        """Calibrate predicted probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration")
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        self.model.calibrate(
            X_np, y_np, 
            method=self.config.calibration_method,
            cv=self.config.calibration_cv
        )
        return self
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Predict probabilities p_c for synthetic class."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inference")
        X_np = X.detach().cpu().numpy()
        proba = self.model.predict_proba(X_np)
        return torch.from_numpy(proba[:, 1]).float().to(X.device)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inference")
        X_np = X.detach().cpu().numpy()
        preds = self.model.predict(X_np)
        return torch.from_numpy(preds).long().to(X.device)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities [p(real), p(synthetic)]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inference")
        X_np = X.detach().cpu().numpy()
        proba = self.model.predict_proba(X_np)
        return torch.from_numpy(proba).float().to(X.device)
    
    def evaluate(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        results = self.model.evaluate(X_np, y_np, metrics or self.config.eval_metrics)
        results["latency"] = self.model.get_latency_stats()
        
        self._eval_history.append(results)
        return results
    
    def get_latency_stats(self) -> Dict[str, float]:
        return self.model.get_latency_stats()
    
    def reset_latency_stats(self) -> None:
        self.model.reset_latency_stats()
    
    def save(self, path: str) -> None:
        """Save the entire branch (model + config)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_dict = {
            "model": self.model,
            "config": self.config,
            "input_dim": self.input_dim,
            "is_fitted": self.is_fitted,
            "training_history": self._training_history,
            "eval_history": self._eval_history,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, path: str) -> "ClassicalBranch":
        """Load a saved branch."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        
        branch = cls(save_dict["config"], save_dict["input_dim"])
        branch.model = save_dict["model"]
        branch.is_fitted = save_dict["is_fitted"]
        branch._training_history = save_dict.get("training_history", [])
        branch._eval_history = save_dict.get("eval_history", [])
        return branch
    
    def get_training_history(self) -> List[Dict]:
        return self._training_history
    
    def get_eval_history(self) -> List[Dict]:
        return self._eval_history
    
    def set_feature_names(self, names: List[str]) -> None:
        self._feature_names = names
        self.model.set_feature_names(names)
    
    def get_feature_names(self) -> Optional[List[str]]:
        return self._feature_names
    
    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()
    
    def is_calibrated(self) -> bool:
        return self.model.is_calibrated()


def run_smoke_experiment(
    model_type: str = "logistic",
    n_samples: int = 200,
    n_features: int = 256,
    test_split: float = 0.2,
    calibrate: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a smoke experiment to verify the classical branch works correctly.
    
    Generates synthetic data with controllable separation between classes
    and evaluates the model.
    
    Args:
        model_type: Type of model to test
        n_samples: Number of synthetic samples
        n_features: Feature dimension (matches Gateway 1 output)
        test_split: Fraction for test set
        calibrate: Whether to calibrate probabilities
        seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        Dictionary with results and trained branch
    """
    np.random.seed(seed)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Smoke Experiment: {model_type.upper()}")
        print(f"{'='*60}")
        print(f"Samples: {n_samples}, Features: {n_features}, Test split: {test_split}")
        print(f"Calibration: {calibrate}")
    
    # Generate synthetic data with class separation
    # Class 0 (real): centered at -0.5, Class 1 (synthetic): centered at +0.5
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Add class separation
    separation = 1.5
    for i in range(n_samples):
        if y[i] == 1:
            X[i] += separation
        else:
            X[i] -= separation
    
    # Split
    n_test = int(n_samples * test_split)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[n_test:], indices[:n_test]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    if verbose:
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        print(f"Train class dist: {np.bincount(y_train)}")
        print(f"Test class dist: {np.bincount(y_test)}")
    
    # Create and train branch
    config = ClassicalConfig(
        model_type=model_type,
        use_calibration=calibrate,
        random_state=seed,
        measure_latency=True,
    )
    branch = ClassicalBranch(config, input_dim=n_features)
    
    # Train
    start = time.time()
    branch.fit(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_time = time.time() - start
    
    if verbose:
        print(f"Training time: {train_time:.3f}s")
    
    # Calibrate if requested
    if calibrate:
        branch.calibrate(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        if verbose:
            print(f"Calibrated: {branch.is_calibrated()}")
    
    # Evaluate
    results = branch.evaluate(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).long()
    )
    
    if verbose:
        print(f"\nTest Metrics:")
        for k, v in results.items():
            if k != "latency":
                print(f"  {k}: {v:.4f}")
        
        if "latency" in results and results["latency"]:
            lat = results["latency"]
            print(f"\nLatency Stats:")
            print(f"  Mean: {lat.get('mean_ms', 0):.2f} ms")
            print(f"  P95:  {lat.get('p95_ms', 0):.2f} ms")
            print(f"  P99:  {lat.get('p99_ms', 0):.2f} ms")
    
    # Test serialization
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name
    
    try:
        branch.save(temp_path)
        loaded_branch = ClassicalBranch.load(temp_path)
        
        # Verify loaded model works
        test_proba = loaded_branch(torch.from_numpy(X_test[:5]).float())
        original_proba = branch(torch.from_numpy(X_test[:5]).float())
        
        assert torch.allclose(test_proba, original_proba), "Serialization failed!"
        if verbose:
            print(f"\nSerialization: OK")
    finally:
        os.unlink(temp_path)
    
    # Test probability prediction
    test_X = torch.from_numpy(X_test[:10]).float()
    probas = branch.predict_proba(test_X)
    preds = branch.predict(test_X)
    
    if verbose:
        print(f"\nPredictions shape: {probas.shape}, Preds shape: {preds.shape}")
        print(f"Probabilities (first 5): {probas[:5].tolist()}")
        print(f"Predictions (first 5): {preds[:5].tolist()}")
    
    return {
        "branch": branch,
        "config": config,
        "results": results,
        "train_time": train_time,
        "X_test": X_test,
        "y_test": y_test,
    }


def run_all_models_smoke(
    n_samples: int = 200,
    n_features: int = 256,
    test_split: float = 0.2,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Run smoke experiment for all available model types."""
    models = ["logistic", "svm", "random_forest", "mlp"]
    if XGBOOST_AVAILABLE:
        models.insert(0, "xgboost")
    
    results = {}
    for model_type in models:
        try:
            if model_type == "svm":
                # SVM needs smaller dataset for speed
                result = run_smoke_experiment(
                    model_type=model_type,
                    n_samples=min(n_samples, 100),
                    n_features=n_features,
                    test_split=test_split,
                    seed=seed,
                    verbose=True,
                )
            else:
                result = run_smoke_experiment(
                    model_type=model_type,
                    n_samples=n_samples,
                    n_features=n_features,
                    test_split=test_split,
                    seed=seed,
                    verbose=True,
                )
            results[model_type] = {"success": True, **result}
        except Exception as e:
            results[model_type] = {"success": False, "error": str(e)}
            print(f"\n{model_type} FAILED: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    for model_type, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        if result["success"]:
            acc = result["results"].get("accuracy", 0)
            print(f"  {model_type:15s}: {status} (acc={acc:.4f})")
        else:
            print(f"  {model_type:15s}: {status} ({result['error']})")
    
    return results


if __name__ == "__main__":
    # Run smoke experiment for all models
    run_all_models_smoke()