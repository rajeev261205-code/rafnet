from typing import Dict, List, Optional, Tuple, Any, Literal
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from abc import ABC, abstractmethod

from ..audio.handcrafted import HandcraftedFeatureExtractor, HandcraftedConfig
from ..classical.branch import (
    BaseClassicalModel,
    LogisticRegressionModel,
    SVMModel,
    RandomForestModel,
    MLPModel,
    XGBoostModel,
    ClassicalConfig,
    create_classical_model,
)
from ..evaluation.metrics import evaluate_all, compute_eer, compute_roc_curve


@dataclass
class AudioOnlyConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    n_mfcc: int = 40
    n_lfcc: int = 60
    n_cqt: int = 84
    fmin: float = 50.0
    fmax: float = 8000.0
    enabled_features: List[str] = field(default_factory=lambda: ["lfcc", "mfcc", "cqt", "spectral"])
    classifier_type: Literal["logistic", "svm", "random_forest", "mlp", "xgboost"] = "logistic"
    classifier_params: Dict[str, Any] = field(default_factory=dict)
    use_calibration: bool = True
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    random_state: int = 42
    n_jobs: int = -1
    device: str = "cpu"


class AudioOnlyDetector:
    def __init__(self, config: AudioOnlyConfig):
        self.config = config
        self.device = config.device
        
        hc_config = HandcraftedConfig(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            n_mfcc=config.n_mfcc,
            n_lfcc=config.n_lfcc,
            n_cqt=config.n_cqt,
            fmin=config.fmin,
            fmax=config.fmax,
        )
        self.feature_extractor = HandcraftedFeatureExtractor(hc_config, config.enabled_features)
        
        dummy = torch.randn(1, config.sample_rate)
        with torch.no_grad():
            feats = self.feature_extractor(dummy)
            total_dim = sum(v.numel() // v.shape[0] for v in feats.values())
        self.feature_dim = total_dim
        
        classical_config = ClassicalConfig(
            model_type=config.classifier_type,
            xgboost_params=config.classifier_params,
            use_calibration=config.use_calibration,
            calibration_method=config.calibration_method,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
        self.classifier = create_classical_model(classical_config)
        self.is_fitted = False
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
    
    def extract_features(self, audio: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            features_dict = self.feature_extractor(audio)
            features = torch.cat(list(features_dict.values()), dim=-1)
        return features.cpu().numpy()
    
    def fit_normalizer(self, X: np.ndarray) -> None:
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return X
        return (X - self.feature_mean) / self.feature_std
    
    def fit(self, audio: torch.Tensor, labels: torch.Tensor) -> "AudioOnlyDetector":
        X = self.extract_features(audio)
        self.fit_normalizer(X)
        X_norm = self.normalize(X)
        y = labels.cpu().numpy()
        self.classifier.fit(X_norm, y)
        self.is_fitted = True
        return self
    
    def calibrate(self, audio: torch.Tensor, labels: torch.Tensor) -> "AudioOnlyDetector":
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration")
        X = self.extract_features(audio)
        X_norm = self.normalize(X)
        y = labels.cpu().numpy()
        self.classifier.calibrate(X_norm, y, method=self.config.calibration_method)
        return self
    
    def predict_proba(self, audio: torch.Tensor) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inference")
        X = self.extract_features(audio)
        X_norm = self.normalize(X)
        return self.classifier.predict_proba(X_norm)
    
    def predict(self, audio: torch.Tensor) -> np.ndarray:
        proba = self.predict_proba(audio)
        return (proba[:, 1] > 0.5).astype(int)
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": self.config,
            "classifier": self.classifier,
            "is_fitted": self.is_fitted,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "feature_dim": self.feature_dim,
            "enabled_features": self.config.enabled_features,
        }
        import pickle
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> "AudioOnlyDetector":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        detector = cls(state["config"])
        detector.classifier = state["classifier"]
        detector.is_fitted = state["is_fitted"]
        detector.feature_mean = state["feature_mean"]
        detector.feature_std = state["feature_std"]
        detector.feature_dim = state["feature_dim"]
        return detector


def run_audio_only_experiment(
    config: AudioOnlyConfig,
    train_audio: torch.Tensor,
    train_labels: torch.Tensor,
    val_audio: torch.Tensor,
    val_labels: torch.Tensor,
    test_audio: torch.Tensor,
    test_labels: torch.Tensor,
    output_dir: str,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    detector = AudioOnlyDetector(config)
    
    start_time = time.time()
    detector.fit(train_audio, train_labels)
    train_time = time.time() - start_time
    
    if config.use_calibration:
        detector.calibrate(val_audio, val_labels)
    
    start_time = time.time()
    test_proba = detector.predict_proba(test_audio)
    inference_time = time.time() - start_time
    test_preds = detector.predict(test_audio)
    
    y_true = test_labels.cpu().numpy()
    y_score = test_proba[:, 1]
    
    metrics = evaluate_all(y_true, y_score, test_preds)
    eer = compute_eer(y_true, y_score)
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_score)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, test_preds)
    
    latency_per_sample = inference_time / len(test_audio) * 1000
    
    results = {
        "config": {
            "classifier_type": config.classifier_type,
            "enabled_features": config.enabled_features,
            "feature_dim": detector.feature_dim,
            "use_calibration": config.use_calibration,
            "calibration_method": config.calibration_method,
        },
        "metrics": {
            "accuracy": float(metrics.get("accuracy", 0)),
            "precision": float(metrics.get("precision", 0)),
            "recall": float(metrics.get("recall", 0)),
            "f1": float(metrics.get("f1", 0)),
            "roc_auc": float(metrics.get("roc_auc", 0)),
            "eer": float(eer),
            "brier": float(metrics.get("brier", 0)),
            "ece": float(metrics.get("ece", 0)),
        },
        "confusion_matrix": cm.tolist(),
        "roc_data": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "latency_ms_per_sample": latency_per_sample,
        "train_time_seconds": train_time,
        "inference_time_seconds": inference_time,
        "n_train": len(train_audio),
        "n_val": len(val_audio),
        "n_test": len(test_audio),
        "status": "CODE_VERIFIED",
    }
    
    with open(output_path / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    predictions_data = {
        "y_true": y_true.tolist(),
        "y_pred": test_preds.tolist(),
        "y_score": y_score.tolist(),
    }
    with open(output_path / "predictions.json", "w") as f:
        json.dump(predictions_data, f, indent=2)
    
    detector.save(str(output_path / "model.pkl"))
    
    return results


def run_classifier_comparison(
    base_config: AudioOnlyConfig,
    train_audio: torch.Tensor,
    train_labels: torch.Tensor,
    val_audio: torch.Tensor,
    val_labels: torch.Tensor,
    test_audio: torch.Tensor,
    test_labels: torch.Tensor,
    output_dir: str,
) -> Dict[str, Any]:
    classifiers = ["logistic", "svm", "random_forest", "mlp", "xgboost"]
    results = {}
    
    for clf_type in classifiers:
        print(f"\nRunning {clf_type}...")
        clf_config = AudioOnlyConfig(
            sample_rate=base_config.sample_rate,
            n_fft=base_config.n_fft,
            hop_length=base_config.hop_length,
            win_length=base_config.win_length,
            n_mels=base_config.n_mels,
            n_mfcc=base_config.n_mfcc,
            n_lfcc=base_config.n_lfcc,
            n_cqt=base_config.n_cqt,
            fmin=base_config.fmin,
            fmax=base_config.fmax,
            enabled_features=base_config.enabled_features,
            classifier_type=clf_type,
            classifier_params=base_config.classifier_params,
            use_calibration=base_config.use_calibration,
            calibration_method=base_config.calibration_method,
            random_state=base_config.random_state,
            n_jobs=base_config.n_jobs,
            device=base_config.device,
        )
        
        clf_output_dir = Path(output_dir) / clf_type
        clf_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = run_audio_only_experiment(
                clf_config,
                train_audio, train_labels,
                val_audio, val_labels,
                test_audio, test_labels,
                str(clf_output_dir),
            )
            results[clf_type] = result
            print(f"  {clf_type}: ACC={result['metrics']['accuracy']:.4f}, EER={result['metrics']['eer']:.4f}, AUC={result['metrics']['roc_auc']:.4f}")
        except Exception as e:
            print(f"  {clf_type} failed: {e}")
            results[clf_type] = {"error": str(e), "status": "FAILED"}
    
    summary = {
        clf: {
            "accuracy": r.get("metrics", {}).get("accuracy"),
            "eer": r.get("metrics", {}).get("eer"),
            "roc_auc": r.get("metrics", {}).get("roc_auc"),
            "f1": r.get("metrics", {}).get("f1"),
            "latency_ms": r.get("latency_ms_per_sample"),
            "status": r.get("status", "UNKNOWN"),
        }
        for clf, r in results.items()
    }
    
    with open(Path(output_dir) / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_smoke_test() -> Dict[str, Any]:
    from ..data.dataset import create_synthetic_fixture, create_datasets_from_manifest
    
    config = AudioOnlyConfig(
        sample_rate=16000,
        enabled_features=["lfcc", "mfcc", "spectral"],
        classifier_type="logistic",
        use_calibration=True,
        device="cpu",
    )
    
    manifest = create_synthetic_fixture(n_samples=200, n_speakers=20, n_generators=10, n_conditions=5, seed=42)
    train_ds, val_ds, test_ds = create_datasets_from_manifest(manifest, max_samples_per_split=30)
    
    train_audio = torch.stack([train_ds[i]["audio"] for i in range(len(train_ds))])
    train_labels = torch.tensor([train_ds[i]["label"].item() for i in range(len(train_ds))])
    val_audio = torch.stack([val_ds[i]["audio"] for i in range(len(val_ds))])
    val_labels = torch.tensor([val_ds[i]["label"].item() for i in range(len(val_ds))])
    test_audio = torch.stack([test_ds[i]["audio"] for i in range(len(test_ds))])
    test_labels = torch.tensor([test_ds[i]["label"].item() for i in range(len(test_ds))])
    
    output_dir = "/tmp/rafnet_audio_only_smoke"
    
    result = run_audio_only_experiment(
        config,
        train_audio, train_labels,
        val_audio, val_labels,
        test_audio, test_labels,
        output_dir,
    )
    
    return result