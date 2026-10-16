"""
Complete Ablation Experiment Runner for RAFNet - Optimized Version.

Key optimizations:
- Share audio/network encoders across ablation models
- Pre-compute embeddings once
- Run classical/quantum fitting on pre-computed embeddings
- Run calibration/decision gate on pre-computed probabilities
"""

import os
import json
import hashlib
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from .metrics import (
    evaluate_all, 
    bootstrap_confidence_intervals, 
    EvaluationConfig,
    compute_latency_throughput,
    compute_model_size
)
from ..config.schema import RAFNetConfig
from ..model import RAFNet
from ..audio import AudioEncoder, AudioEncoderConfig
from ..network import NetworkEncoder, NetworkEncoderConfig
from ..fusion import Gateway1, Gateway1Config, Gateway2, Gateway2Config
from ..classical import ClassicalBranch, ClassicalConfig
from ..quantum import QuantumBranch, QuantumConfig
from ..calibration import Calibrator, CalibrationConfig
from ..decision import DecisionGate, DecisionGateConfig
from ..data.dataset import (
    SyntheticVoiceDataset, 
    create_datasets_from_manifest, 
    create_synthetic_fixture,
    collate_fn,
    create_dataloader,
    fit_normalizers,
)
from ..data.manifest import Manifest, assign_splits, create_synthetic_manifest


@dataclass
class ExperimentConfig:
    """Configuration for ablation experiment runner."""
    experiment_name: str = "rafnet_ablation"
    output_dir: str = "experiments"
    
    rafnet_config: RAFNetConfig = field(default_factory=RAFNetConfig)
    
    n_samples: int = 2000
    n_speakers: int = 50
    n_generators: int = 20
    n_conditions: int = 10
    split_strategy: str = "generator"
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    audio_length: int = 16000
    network_dim: int = 32
    batch_size: int = 16
    num_workers: int = 4
    
    classical_model_type: str = "logistic"
    quantum_fallback_mode: str = "classical"
    calibrate_classical: bool = True
    calibrate_quantum: bool = True
    train_quantum_head: bool = True
    quantum_train_epochs: int = 30
    decision_optimize_thresholds: bool = True
    decision_grid_resolution: int = 50
    
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "roc_auc", "eer",
        "fpr", "fnr", "ece", "brier", "risk_coverage", "coverage_error_curve",
        "latency", "throughput"
    ])
    bootstrap_ci: bool = True
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    
    seed: int = 42
    deterministic: bool = True
    device: str = "cpu"
    
    # Quick mode for testing
    quick_mode: bool = False
    quick_n_samples: int = 100
    
    def to_dict(self) -> Dict:
        result = {}
        for k, v in asdict(self).items():
            if isinstance(v, tuple):
                result[k] = list(v)
            elif hasattr(v, '__dataclass_fields__'):
                result[k] = asdict(v)
            elif isinstance(v, Path):
                result[k] = str(v)
            else:
                result[k] = v
        return result
    
    def config_hash(self) -> str:
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class ExperimentResult:
    model_name: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    bootstrap_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    latency_stats: Dict[str, float] = field(default_factory=dict)
    model_size: Dict[str, int] = field(default_factory=dict)
    training_time_sec: float = 0.0
    fitting_details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result["bootstrap_ci"] = {k: list(v) for k, v in self.bootstrap_ci.items()}
        return result


@dataclass 
class ExperimentMetadata:
    experiment_id: str
    experiment_name: str
    timestamp: str
    git_commit: str
    git_branch: str
    config_hash: str
    model_version: str
    dataset_version: str
    seed: int
    device: str
    python_version: str
    torch_version: str
    numpy_version: str
    total_duration_sec: float
    models_run: List[str]
    models_completed: List[str]
    models_failed: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SharedComponents:
    """Shared encoders and embeddings across ablation models."""
    
    def __init__(self, config: ExperimentConfig, device: str):
        self.config = config
        self.device = device
        self._setup()
    
    def _setup(self):
        c = self.config.rafnet_config
        
        # Audio encoder (expensive - load once)
        self.audio_encoder = AudioEncoder(AudioEncoderConfig(
            sample_rate=c.audio.sample_rate,
            wavlm_model=c.audio.wavlm_model,
            wavlm_freeze=c.audio.wavlm_freeze,
            wavlm_output_dim=c.audio.wavlm_output_dim,
            projection_dim=c.audio.projection_dim,
            wavlm_pooling=c.audio.wavlm_pooling,
            wavlm_checkpointing=c.audio.wavlm_checkpointing,
            wavlm_mixed_precision=c.audio.wavlm_mixed_precision,
            use_handcrafted=c.audio.use_handcrafted,
            handcrafted_features=c.audio.handcrafted_features,
            handcrafted_projection_dim=c.audio.handcrafted_projection_dim,
            wavlm_stride=c.audio.wavlm_stride,
            device=self.device,
        ))
        
        # Network encoder
        self.network_encoder = NetworkEncoder(NetworkEncoderConfig(
            input_dim=c.network.input_dim,
            hidden_dims=c.network.hidden_dims,
            output_dim=c.network.output_dim,
            use_transformer=c.network.use_transformer,
            transformer_layers=c.network.transformer_layers,
            transformer_heads=c.network.transformer_heads,
            dropout=c.network.dropout,
            normalize_input=c.network.normalize_input,
            temporal_window=c.network.temporal_window,
            num_codecs=c.network.num_codecs,
            codec_embed_dim=c.network.codec_embed_dim,
        ))
        
        # Gateway 1
        self.gateway1 = Gateway1(Gateway1Config(
            audio_dim=self.audio_encoder.get_output_dim(),
            network_dim=self.network_encoder.get_output_dim(),
            hidden_dim=c.gateway1.hidden_dim,
            gate_type=c.gateway1.gate_type,
            reliability_aware=c.gateway1.reliability_aware,
            num_layers=c.gateway1.num_layers,
            use_residual_fusion=c.gateway1.use_residual_fusion,
            residual_weight=c.gateway1.residual_weight,
            gate_bias_init=c.gateway1.gate_bias_init,
            log_gate_stats=c.gateway1.log_gate_stats,
            ablation_mode=c.gateway1.ablation_mode,
        ))
        
        # Gateway 2
        self.gateway2 = Gateway2(Gateway2Config(
            input_dim=3,
            hidden_dim=c.gateway2.hidden_dim,
            gate_type=c.gateway2.gate_type,
            dropout=c.gateway2.dropout,
            num_layers=c.gateway2.num_layers,
            gate_bias_init=c.gateway2.gate_bias_init,
            log_gate_stats=c.gateway2.log_gate_stats,
            enable_classical=c.gateway2.enable_classical,
            enable_quantum=c.gateway2.enable_quantum,
            ablation_mode=c.gateway2.ablation_mode,
        ))
        
        # Calibrator
        self.calibrator = Calibrator(CalibrationConfig(
            method=c.calibration.method,
            temperature_init=c.calibration.temperature_init,
            max_iter=c.calibration.max_iter,
            lr=c.calibration.lr,
            fit_on_validation_only=c.calibration.fit_on_validation_only,
            compute_ece=c.calibration.compute_ece,
            n_bins_ece=c.calibration.n_bins_ece,
            compute_brier=c.calibration.compute_brier,
        ))
        
        # Decision gate
        self.decision_gate = DecisionGate(DecisionGateConfig(
            tau_low=c.decision_gate.tau_low,
            tau_high=c.decision_gate.tau_high,
            optimize_thresholds=self.config.decision_optimize_thresholds,
            metric_for_optimization=c.decision_gate.metric_for_optimization,
            grid_resolution=self.config.decision_grid_resolution,
            select_on_validation=c.decision_gate.select_on_validation,
            uncertainty_action=c.decision_gate.uncertainty_action,
            min_uncertainty_width=c.decision_gate.min_uncertainty_width,
            max_uncertainty_width=c.decision_gate.max_uncertainty_width,
        ))
        
        self.to(self.device)
        self.eval()
    
    def to(self, device):
        self.audio_encoder.to(device)
        self.network_encoder.to(device)
        self.gateway1.to(device)
        self.gateway2.to(device)
        self.calibrator.to(device)
        self.decision_gate.to(device)
    
    def eval(self):
        self.audio_encoder.eval()
        self.network_encoder.eval()
        self.gateway1.eval()
        self.gateway2.eval()
    
    def compute_embeddings(self, dataset, max_samples=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute z_f embeddings for a dataset."""
        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        z_f_list = []
        labels_list = []
        
        with torch.no_grad():
            for i in range(n):
                item = dataset[i]
                audio = item["audio"].unsqueeze(0).to(self.device)
                network = item["network"].unsqueeze(0).to(self.device)
                audio_mask = item.get("audio_mask", None)
                if audio_mask is not None:
                    audio_mask = audio_mask.unsqueeze(0).to(self.device)
                
                z_a = self.audio_encoder(audio, audio_mask)
                z_n = self.network_encoder(network)
                z_f, _ = self.gateway1(z_a, z_n)
                
                z_f_list.append(z_f.cpu())
                labels_list.append(item["label"])
        
        return torch.cat(z_f_list, dim=0), torch.tensor(labels_list)


def get_git_info() -> Dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd="/home/sk/rafnet", 
            stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            cwd="/home/sk/rafnet", 
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        return {"commit": "unknown", "branch": "unknown"}


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def run_ablation_on_embeddings(
    model_name: str,
    config: ExperimentConfig,
    shared: SharedComponents,
    train_z_f: torch.Tensor,
    train_labels: torch.Tensor,
    val_z_f: torch.Tensor,
    val_labels: torch.Tensor,
    test_dataset,
) -> ExperimentResult:
    """Run ablation model using pre-computed embeddings."""
    result = ExperimentResult(model_name=model_name, status="running")
    start_time = time.time()
    device = config.device
    
    try:
        details = {}
        
        # Classical branch
        classical_branch = None
        if model_name in ["M5_gateway1_classical_quantum", "M6_gateway2", "M7_complete"]:
            classical_branch = ClassicalBranch(
                ClassicalConfig(
                    model_type=config.classical_model_type,
                    xgboost_params=config.rafnet_config.classical.xgboost_params,
                    use_calibration=config.calibrate_classical,
                    calibration_method=config.rafnet_config.classical.calibration_method,
                    random_state=config.rafnet_config.classical.random_state,
                    n_jobs=config.rafnet_config.classical.n_jobs,
                    measure_latency=config.rafnet_config.classical.measure_latency,
                    calibration_cv=config.rafnet_config.classical.calibration_cv,
                    eval_metrics=config.rafnet_config.classical.eval_metrics,
                    serialization_format=config.rafnet_config.classical.serialization_format,
                ),
                config.rafnet_config.audio.projection_dim
            )
            classical_branch.fit(train_z_f, train_labels)
            if config.calibrate_classical:
                classical_branch.calibrate(val_z_f, val_labels)
            details["classical_fitted"] = True
            details["classical_calibrated"] = config.calibrate_classical
        
        # Quantum branch
        quantum_branch = None
        if model_name in ["M5_gateway1_classical_quantum", "M6_gateway2", "M7_complete"]:
            quantum_branch = QuantumBranch(
                QuantumConfig(
                    n_qubits=config.rafnet_config.quantum.n_qubits,
                    n_layers=config.rafnet_config.quantum.n_layers,
                    encoding=config.rafnet_config.quantum.encoding,
                    pca_dim=config.rafnet_config.quantum.pca_dim,
                    device=config.rafnet_config.quantum.device,
                    shots=config.rafnet_config.quantum.shots,
                    use_classical_head=config.rafnet_config.quantum.use_classical_head,
                    classical_head_hidden=config.rafnet_config.quantum.classical_head_hidden,
                    random_state=config.rafnet_config.quantum.random_state,
                    entanglement=config.rafnet_config.quantum.entanglement,
                    rotation_gates=config.rafnet_config.quantum.rotation_gates,
                    measure_latency=config.rafnet_config.quantum.measure_latency,
                    checkpointing=config.rafnet_config.quantum.checkpointing,
                    fallback_mode=config.quantum_fallback_mode,
                    serialization_format=config.rafnet_config.quantum.serialization_format,
                ),
                config.rafnet_config.audio.projection_dim
            )
            quantum_branch.fit_pca(train_z_f)
            if config.train_quantum_head:
                quantum_branch.fit_quantum(
                    val_z_f, val_labels,
                    n_epochs=config.quantum_train_epochs,
                    lr=0.01,
                    batch_size=32,
                    calibrate=config.calibrate_quantum,
                )
            details["quantum_pca_fitted"] = True
            details["quantum_head_trained"] = config.train_quantum_head
            details["quantum_calibrated"] = config.calibrate_quantum
        
        # Calibrator
        if model_name in ["M6_gateway2", "M7_complete"]:
            with torch.no_grad():
                train_p_c = classical_branch(train_z_f)
                train_p_q = quantum_branch(train_z_f)
                train_p_fused, _ = shared.gateway2(train_p_c, train_p_q)
                
                val_p_c = classical_branch(val_z_f)
                val_p_q = quantum_branch(val_z_f)
                val_p_fused, _ = shared.gateway2(val_p_c, val_p_q)
            
            shared.calibrator.fit(train_p_fused, train_labels, val_p_fused, val_labels)
            details["calibrator_fitted"] = True
        
        # Decision gate thresholds
        if model_name in ["M6_gateway2", "M7_complete"] and config.decision_optimize_thresholds:
            with torch.no_grad():
                val_p_c = classical_branch(val_z_f)
                val_p_q = quantum_branch(val_z_f)
                val_p_fused, _ = shared.gateway2(val_p_c, val_p_q)
                
                if shared.calibrator.fitted:
                    val_p_cal = shared.calibrator(val_p_fused)
                else:
                    val_p_cal = val_p_fused
            
            shared.decision_gate.optimize_thresholds(
                train_p_fused if 'train_p_fused' in locals() else val_p_fused,
                train_labels if 'train_labels' in locals() else val_labels,
                val_p_cal,
                val_labels
            )
            details["thresholds_optimized"] = True
            details["tau_low"] = shared.decision_gate.tau_low.item()
            details["tau_high"] = shared.decision_gate.tau_high.item()
        
        result.training_time_sec = time.time() - start_time
        result.fitting_details = details
        
        # Now evaluate on test set - BATCHED
        test_audio = torch.stack([test_dataset[i]['audio'] for i in range(len(test_dataset))]).to(device)
        test_network = torch.stack([test_dataset[i]['network'] for i in range(len(test_dataset))]).to(device)
        test_labels = np.array([test_dataset[i]['label'] for i in range(len(test_dataset))])
        
        # Build eval model
        model = _build_eval_model(model_name, config, shared, classical_branch, quantum_branch)
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            z_a = shared.audio_encoder(test_audio)
            z_n = shared.network_encoder(test_network)
            z_f, _ = shared.gateway1(z_a, z_n)
            
            if model_name in ["M1_audio_only", "M2_network_only", "M3_fixed_fusion", "M4_gateway1"]:
                if model_name == "M1_audio_only":
                    z_f = z_a
                elif model_name == "M2_network_only":
                    z_f = z_n
                elif model_name == "M3_fixed_fusion":
                    z_f = torch.nn.functional.linear(
                        torch.cat([z_a, z_n], dim=-1),
                        torch.nn.Linear(z_a.shape[-1] + z_n.shape[-1], config.rafnet_config.audio.projection_dim).weight
                    )
                # M4 uses gateway1 already
                
                probs = torch.sigmoid(torch.nn.Linear(z_f.shape[-1], 1, device=device)(z_f)).squeeze(-1)
            else:
                p_c = classical_branch(z_f) if classical_branch and classical_branch.is_fitted else torch.full((z_f.shape[0],), 0.5, device=device)
                p_q = quantum_branch(z_f) if quantum_branch and quantum_branch.pca_fitted else torch.full((z_f.shape[0],), 0.5, device=device)
                p_fused, _ = shared.gateway2(p_c, p_q)
                
                if shared.calibrator.fitted:
                    probs = shared.calibrator(p_fused)
                else:
                    probs = p_fused
            
            all_probs = probs.cpu().numpy()
            all_labels = test_labels
            all_preds = (all_probs > 0.5).astype(int)
            
            # Get decisions if applicable
            if hasattr(shared, 'decision_gate') and shared.decision_gate.optimized:
                all_decisions, _ = shared.decision_gate(torch.from_numpy(all_probs).to(device))
                all_decisions = all_decisions.cpu().numpy()
            else:
                all_decisions = None
        
        eval_config = EvaluationConfig(
            metrics=config.evaluation_metrics,
            n_bootstrap=0 if config.quick_mode else config.n_bootstrap,
            confidence_level=config.confidence_level,
        )
        
        metrics = evaluate_all(
            all_labels, all_probs, all_preds, all_decisions,
            config=eval_config,
            model=None,  # Skip latency measurement
            device=device,
        )
        result.metrics = metrics
        
        if config.bootstrap_ci and not config.quick_mode:
            ci = bootstrap_confidence_intervals(
                all_labels, all_probs, all_preds, all_decisions,
                config=eval_config,
                n_bootstrap=config.n_bootstrap,
                confidence_level=config.confidence_level,
            )
            result.bootstrap_ci = ci
        
        result.latency_stats = model.get_latency_stats() if hasattr(model, 'get_latency_stats') else {}
        result.model_size = model.get_model_size() if hasattr(model, 'get_model_size') else {}
        result.status = "completed"
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        import traceback
        result.fitting_details["traceback"] = traceback.format_exc()
    
    return result


def _build_eval_model(model_name, config, shared, classical_branch, quantum_branch):
    """Build evaluation model for a specific ablation."""
    class EvalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.name = model_name
            self.shared = shared
            self.classical_branch = classical_branch
            self.quantum_branch = quantum_branch
        
        def forward(self, audio, network, audio_mask=None):
            z_a = self.shared.audio_encoder(audio, audio_mask)
            z_n = self.shared.network_encoder(network)
            
            if self.name in ["M1_audio_only", "M2_network_only", "M3_fixed_fusion", "M4_gateway1"]:
                if self.name == "M1_audio_only":
                    z_f = z_a
                elif self.name == "M2_network_only":
                    z_f = z_n
                elif self.name == "M3_fixed_fusion":
                    z_f = torch.nn.functional.linear(
                        torch.cat([z_a, z_n], dim=-1),
                        torch.nn.Linear(z_a.shape[-1] + z_n.shape[-1], config.rafnet_config.audio.projection_dim).weight
                    )
                else:  # M4
                    z_f, _ = self.shared.gateway1(z_a, z_n)
                
                if self.name in ["M1_audio_only", "M2_network_only"]:
                    return torch.sigmoid(torch.nn.Linear(z_f.shape[-1], 1)(z_f)).squeeze(-1)
                elif self.name == "M3_fixed_fusion":
                    return torch.sigmoid(torch.nn.Linear(config.rafnet_config.audio.projection_dim, 1)(z_f)).squeeze(-1)
                else:  # M4
                    return torch.sigmoid(torch.nn.Linear(config.rafnet_config.audio.projection_dim, 1)(z_f)).squeeze(-1)
            
            else:
                z_f, _ = self.shared.gateway1(z_a, z_n)
                
                p_c = self.classical_branch(z_f) if self.classical_branch and self.classical_branch.is_fitted else torch.full((z_f.shape[0],), 0.5, device=z_f.device)
                p_q = self.quantum_branch(z_f) if self.quantum_branch and self.quantum_branch.pca_fitted else torch.full((z_f.shape[0],), 0.5, device=z_f.device)
                p_fused, _ = self.shared.gateway2(p_c, p_q)
                
                if hasattr(self.shared, 'calibrator') and self.shared.calibrator.fitted:
                    return self.shared.calibrator(p_fused)
                return p_fused
        
        def get_latency_stats(self):
            return {}
        
        def get_model_size(self):
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return {"total_parameters": total, "trainable_parameters": trainable}
    
    model = EvalModel()
    model.shared = shared
    model.classical_branch = classical_branch
    model.quantum_branch = quantum_branch
    model.name = model_name
    return model


def run_ablation_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    overall_start = time.time()
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.deterministic:
        torch.use_deterministic_algorithms(True)
    
    # Quick mode adjustments
    if config.quick_mode:
        n_samples = config.quick_n_samples
        config.quantum_train_epochs = 2
        config.decision_grid_resolution = 5
        config.n_bootstrap = 50
        config.split_strategy = "speaker"  # Ensure both classes in test
        # Skip M5-M7 in quick mode
        quick_model_names = [
            "M1_audio_only",
            "M2_network_only", 
            "M3_fixed_fusion",
            "M4_gateway1",
        ]
    else:
        n_samples = config.n_samples
    
    exp_dir = create_experiment_dir(config.output_dir, config.experiment_name)
    git_info = get_git_info()
    
    print("Creating dataset...")
    manifest = create_synthetic_manifest(
        n_samples=n_samples,
        audio_length=config.audio_length,
        n_speakers=config.n_speakers,
        n_generators=config.n_generators,
        n_conditions=config.n_conditions,
        seed=config.seed,
        split_ratios=config.split_ratios,
    )
    manifest = assign_splits(manifest, config.split_ratios, strategy=config.split_strategy, seed=config.seed)
    
    # Create datasets
    max_per_split = 100 if config.quick_mode else None
    train_dataset, val_dataset, test_dataset = create_datasets_from_manifest(
        manifest,
        audio_root="",
        sample_rate=config.rafnet_config.audio.sample_rate,
        network_dim=config.network_dim,
        max_samples_per_split=max_per_split,
    )
    
    dataset_version = manifest.compute_hash()
    
    # Pre-compute embeddings
    print("Computing shared embeddings...")
    shared = SharedComponents(config, config.device)
    
    print("  Train embeddings...")
    train_z_f, train_labels = shared.compute_embeddings(train_dataset, max_samples=max_per_split)
    print("  Val embeddings...")
    val_z_f, val_labels = shared.compute_embeddings(val_dataset, max_samples=max_per_split)
    
    print(f"  Train: {train_z_f.shape}, Val: {val_z_f.shape}")
    
    # Run ablation models
    if config.quick_mode:
        model_names = [
            "M1_audio_only",
            "M2_network_only", 
            "M3_fixed_fusion",
            "M4_gateway1",
        ]
    else:
        model_names = [
            "M1_audio_only",
            "M2_network_only", 
            "M3_fixed_fusion",
            "M4_gateway1",
            "M5_gateway1_classical_quantum",
            "M6_gateway2",
            "M7_complete",
        ]
    
    print(f"Running models: {model_names}")
    
    results = {}
    completed = []
    failed = []
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Running {model_name}")
        print(f"{'='*60}")
        
        result = run_ablation_on_embeddings(
            model_name, config, shared,
            train_z_f, train_labels, val_z_f, val_labels,
            test_dataset
        )
        
        results[model_name] = result.to_dict()
        
        if result.status == "completed":
            completed.append(model_name)
            print(f"  ✓ Completed")
        else:
            failed.append(model_name)
            print(f"  ✗ Failed: {result.error}")
        
        # Save intermediate
        interim_path = exp_dir / "interim_results.json"
        with open(interim_path, "w") as f:
            json.dump(results, f, indent=2)
    
    # Build metadata
    metadata = ExperimentMetadata(
        experiment_id=exp_dir.name,
        experiment_name=config.experiment_name,
        timestamp=datetime.now().isoformat(),
        git_commit=git_info["commit"],
        git_branch=git_info["branch"],
        config_hash=config.config_hash(),
        model_version="1.0.0",
        dataset_version=dataset_version,
        seed=config.seed,
        device=config.device,
        python_version=f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        total_duration_sec=time.time() - overall_start,
        models_run=model_names,
        models_completed=completed,
        models_failed=failed,
    )
    
    output = {
        "metadata": metadata.to_dict(),
        "config": config.to_dict(),
        "results": results,
    }
    
    json_path = exp_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    
    csv_rows = []
    for model_name, result in results.items():
        row = {"model": model_name, "status": result["status"]}
        row.update(result.get("metrics", {}))
        row.update({f"ci_{k}_lower": v[0] for k, v in result.get("bootstrap_ci", {}).items()})
        row.update({f"ci_{k}_upper": v[1] for k, v in result.get("bootstrap_ci", {}).items()})
        row.update(result.get("latency_stats", {}))
        row.update(result.get("model_size", {}))
        row["training_time_sec"] = result.get("training_time_sec", 0)
        if result.get("error"):
            row["error"] = result["error"]
        csv_rows.append(row)
    
    csv_path = exp_dir / "results_summary.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment completed: {exp_dir}")
    print(f"  Completed: {len(completed)}/{len(model_names)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Duration: {time.time() - overall_start:.1f}s")
    
    return {
        "experiment_dir": str(exp_dir),
        "metadata": metadata.to_dict(),
        "results": results,
    }


def run_quick_smoke() -> Dict[str, Any]:
    config = ExperimentConfig(
        quick_mode=True,
        quick_n_samples=200,
        n_speakers=20,
        n_generators=10,
        n_conditions=5,
        seed=42,
        experiment_name="rafnet_ablation_smoke",
    )
    return run_ablation_experiment(config)


if __name__ == "__main__":
    run_quick_smoke()