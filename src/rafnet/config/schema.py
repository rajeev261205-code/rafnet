from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal
import json


def _dataclass_to_dict(obj):
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for f in obj.__dataclass_fields__:
            v = getattr(obj, f)
            result[f] = _dataclass_to_dict(v)
        return result
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    wavlm_model: str = "microsoft/wavlm-base-plus"
    wavlm_freeze: bool = True
    wavlm_output_dim: int = 768
    projection_dim: int = 256
    wavlm_pooling: str = "mean"
    wavlm_checkpointing: bool = False
    wavlm_mixed_precision: bool = False
    use_handcrafted: bool = True
    handcrafted_features: List[str] = field(default_factory=lambda: ["lfcc", "mfcc", "cqt", "spectral", "prosodic", "voice_quality"])
    lfcc_n_filters: int = 60
    mfcc_n_mfcc: int = 40
    cqt_n_bins: int = 84
    handcrafted_projection_dim: int = 128
    wavlm_stride: int = 320
    device: str = "cpu"


@dataclass
class NetworkConfig:
    input_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    output_dim: int = 256
    use_transformer: bool = False
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1
    normalize_input: bool = True
    temporal_window: int = 10
    num_codecs: int = 8
    codec_embed_dim: int = 8


@dataclass
class Gateway1Config:
    audio_dim: int = 256
    network_dim: int = 256
    context_dim: int = 0
    hidden_dim: int = 128
    gate_type: Literal["sigmoid", "softmax"] = "sigmoid"
    reliability_aware: bool = True
    dropout: float = 0.1
    num_layers: int = 2
    use_residual_fusion: bool = False
    residual_weight: float = 0.1
    gate_bias_init: float = 0.0
    log_gate_stats: bool = True
    ablation_mode: Optional[Literal["fixed_audio", "fixed_network", "fixed_equal", None]] = None
    
    @property
    def input_dim(self) -> int:
        return self.audio_dim + self.network_dim + 2 * self.context_dim


@dataclass
class ClassicalConfig:
    model_type: Literal["xgboost", "logistic", "svm", "random_forest", "mlp"] = "logistic"
    xgboost_params: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    })
    use_calibration: bool = True
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    random_state: int = 42
    n_jobs: int = -1
    # Additional options
    measure_latency: bool = True
    calibration_cv: int = 3
    eval_metrics: list = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "roc_auc", "logloss"])
    serialization_format: Literal["pickle", "joblib"] = "pickle"


@dataclass
class QuantumConfig:
    n_qubits: int = 8
    n_layers: int = 3
    encoding: Literal["angle", "amplitude"] = "angle"
    pca_dim: int = 8
    device: str = "default.qubit"
    shots: Optional[int] = None
    use_classical_head: bool = True
    classical_head_hidden: int = 32
    random_state: int = 42
    # Additional options
    entanglement: Literal["linear", "circular", "full"] = "circular"
    rotation_gates: Literal["RX_RY_RZ", "RY_RZ", "RX_RY"] = "RX_RY_RZ"
    measure_latency: bool = True
    checkpointing: bool = False
    fallback_mode: Literal["classical", "none"] = "classical"
    serialization_format: Literal["pickle", "joblib"] = "pickle"


@dataclass
class Gateway2Config:
    input_dim: int = 3
    hidden_dim: int = 32
    gate_type: Literal["sigmoid", "softmax"] = "sigmoid"
    dropout: float = 0.1
    num_layers: int = 2
    gate_bias_init: float = 0.0
    log_gate_stats: bool = True
    # Branch control
    enable_classical: bool = True
    enable_quantum: bool = True
    # Ablation modes
    ablation_mode: Optional[Literal["fixed_classical", "fixed_quantum", "fixed_equal", None]] = None


@dataclass
class CalibrationConfig:
    method: Literal["temperature", "platt", "isotonic"] = "temperature"
    temperature_init: float = 1.0
    max_iter: int = 100
    lr: float = 0.01
    # Validation-based fitting (prevents leakage)
    fit_on_validation_only: bool = True
    # Metrics
    compute_ece: bool = True
    n_bins_ece: int = 10
    compute_brier: bool = True


@dataclass
class DecisionGateConfig:
    tau_low: float = 0.3
    tau_high: float = 0.7
    optimize_thresholds: bool = True
    metric_for_optimization: Literal["f1", "precision", "recall", "balanced_accuracy", "mcc"] = "f1"
    grid_resolution: int = 100
    # Validation-based threshold selection
    select_on_validation: bool = True
    # Uncertainty region handling
    uncertainty_action: Literal["reject", "random", "classical", "quantum"] = "reject"
    # Constraints
    min_uncertainty_width: float = 0.1
    max_uncertainty_width: float = 0.5


@dataclass
class LossConfig:
    ce_weight: float = 1.0
    rel_weight: float = 0.5
    selcal_weight: float = 0.1
    agree_weight: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    gradient_clip: float = 1.0
    mixed_precision: bool = False
    num_workers: int = 4
    val_split: float = 0.2
    test_split: float = 0.1


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "roc_auc", "eer",
        "fpr", "fnr", "ece", "brier", "risk_coverage", "latency", "throughput"
    ])
    n_bootstrap: int = 1000
    confidence_level: float = 0.95


@dataclass
class AblationConfig:
    models: List[str] = field(default_factory=lambda: [
        "M1_audio_only",
        "M2_network_only",
        "M3_fixed_fusion",
        "M4_gateway1",
        "M5_gateway1_classical_quantum",
        "M6_gateway2",
        "M7_complete",
    ])


@dataclass
class DataConfig:
    train_path: str = "data/train"
    val_path: str = "data/val"
    test_path: str = "data/test"
    speaker_disjoint: bool = True
    generator_disjoint: bool = True
    condition_disjoint: bool = True


@dataclass
class RAFNetConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    gateway1: Gateway1Config = field(default_factory=Gateway1Config)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    gateway2: Gateway2Config = field(default_factory=Gateway2Config)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    decision_gate: DecisionGateConfig = field(default_factory=DecisionGateConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "outputs"
    experiment_name: str = "rafnet"

    @classmethod
    def from_yaml(cls, path: str) -> "RAFNetConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict):
        # Recursively convert dicts to dataclass instances
        def convert(obj, target_cls):
            if hasattr(target_cls, '__dataclass_fields__'):
                kwargs = {}
                for f in target_cls.__dataclass_fields__:
                    v = obj.get(f)
                    if v is not None:
                        field_type = target_cls.__dataclass_fields__[f].type
                        if hasattr(field_type, '__dataclass_fields__'):
                            kwargs[f] = convert(v, field_type)
                        elif hasattr(field_type, '__origin__') and hasattr(field_type, '__args__'):
                            # Handle List[SomeDataclass] or Optional[SomeDataclass]
                            args = field_type.__args__
                            if args and hasattr(args[0], '__dataclass_fields__'):
                                if isinstance(v, list):
                                    kwargs[f] = [convert(item, args[0]) for item in v]
                                else:
                                    kwargs[f] = convert(v, args[0])
                            else:
                                kwargs[f] = v
                        else:
                            kwargs[f] = v
                    else:
                        kwargs[f] = getattr(target_cls, f) if hasattr(target_cls, f) else None
                return target_cls(**kwargs)
            return obj
        
        return convert(data, cls)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(_dataclass_to_dict(self), f, indent=2)


def load_config(path: Optional[str] = None) -> RAFNetConfig:
    if path is None:
        return RAFNetConfig()
    return RAFNetConfig.from_yaml(path)