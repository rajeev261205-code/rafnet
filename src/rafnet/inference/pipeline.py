import time
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path

from ..config.schema import RAFNetConfig, AudioConfig, NetworkConfig, Gateway1Config, Gateway2Config
from ..config.schema import ClassicalConfig, QuantumConfig, CalibrationConfig, DecisionGateConfig
from ..audio import AudioEncoder, AudioEncoderConfig
from ..network import NetworkEncoder, NetworkEncoderConfig
from ..fusion import Gateway1, Gateway1Config as FusionGateway1Config, Gateway2, Gateway2Config as FusionGateway2Config
from ..classical import ClassicalBranch, ClassicalConfig as BranchClassicalConfig
from ..quantum import QuantumBranch, QuantumConfig as BranchQuantumConfig
from ..calibration import Calibrator, CalibrationConfig as CalibratorCalibrationConfig
from ..decision import DecisionGate, DecisionGateConfig as DecisionGateDecisionConfig


@dataclass
class InferenceOutput:
    """Complete output from the inference pipeline."""
    # Core predictions
    prediction: int  # 0=REAL, 1=UNCERTAIN, 2=SYNTHETIC
    decision_label: str  # "REAL", "UNCERTAIN", "SYNTHETIC"
    
    # Probabilities
    probability_real: float
    probability_synthetic: float
    confidence: float  # max(probability_real, probability_synthetic)
    
    # Reliability scores
    audio_reliability: Optional[float] = None
    network_reliability: Optional[float] = None
    
    # Gateway 1 (embedding fusion)
    gateway1_audio_weight: Optional[float] = None  # alpha
    gateway1_network_weight: Optional[float] = None  # 1 - alpha
    
    # Branch probabilities
    classical_probability: Optional[float] = None  # p_c
    quantum_probability: Optional[float] = None  # p_q
    
    # Gateway 2 (probability fusion)
    gateway2_classical_weight: Optional[float] = None  # beta
    gateway2_quantum_weight: Optional[float] = None  # 1 - beta
    fused_probability: Optional[float] = None  # p_fused
    
    # Calibration
    calibrated_probability: Optional[float] = None  # p_calibrated
    
    # Latency
    latency_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    
    # Model info
    model_version: str = "1.0.0"
    
    # Debug info
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                result[key] = value.item() if value.numel() == 1 else value.tolist()
            elif isinstance(value, dict):
                result[key] = {k: v.item() if isinstance(v, (np.ndarray, torch.Tensor)) and v.numel() == 1 else v 
                              for k, v in value.items()}
            else:
                result[key] = value
        return result


class InferencePipeline(nn.Module):
    """
    Complete RAFNet inference pipeline integrating all components.
    
    Pipeline:
        Audio → preprocessing → WavLM → z_a
        Network → preprocessing → NetworkEncoder → z_n
        z_a + z_n → Gateway 1 → z_f
        z_f → XGBoost → p_c
        z_f → PCA → Quantum → p_q
        p_c + p_q → Gateway 2 → p_fused
        p_fused → calibration → p_calibrated
        p_calibrated → Decision Gate → decision
    
    Features:
    - End-to-end inference with all components
    - Detailed latency tracking per component
    - Deterministic behavior with fixed seed
    - Audio-only mode (Gateway 1 + Classical + Calibration + Decision)
    - Full verification of tensor shapes and value ranges
    """
    
    def __init__(
        self,
        config: Optional[RAFNetConfig] = None,
        model_path: Optional[str] = None,
        device: str = "cpu",
        seed: int = 42,
        verify_shapes: bool = True,
    ):
        super().__init__()
        
        self.config = config or RAFNetConfig()
        self.device = device
        self.seed = seed
        self.verify_shapes = verify_shapes
        self.model_version = "1.0.0"
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize components
        self._init_audio_encoder()
        self._init_network_encoder()
        self._init_gateway1()
        self._init_classical_branch()
        self._init_quantum_branch()
        self._init_gateway2()
        self._init_calibrator()
        self._init_decision_gate()
        
        # Move to device
        self.to(device)
        self.eval()
        
        # Latency tracking
        self.latency_tracker: Dict[str, List[float]] = {}
        
        # Load pretrained weights if provided
        if model_path:
            self.load(model_path)
    
    def _init_audio_encoder(self):
        audio_config = AudioEncoderConfig(
            sample_rate=self.config.audio.sample_rate,
            wavlm_model=self.config.audio.wavlm_model,
            wavlm_freeze=self.config.audio.wavlm_freeze,
            wavlm_output_dim=self.config.audio.wavlm_output_dim,
            projection_dim=self.config.audio.projection_dim,
            wavlm_pooling=self.config.audio.wavlm_pooling,
            wavlm_checkpointing=self.config.audio.wavlm_checkpointing,
            wavlm_mixed_precision=self.config.audio.wavlm_mixed_precision,
            use_handcrafted=self.config.audio.use_handcrafted,
            handcrafted_features=self.config.audio.handcrafted_features,
            handcrafted_projection_dim=self.config.audio.handcrafted_projection_dim,
            wavlm_stride=self.config.audio.wavlm_stride,
            device=self.device,
        )
        self.audio_encoder = AudioEncoder(audio_config)
    
    def _init_network_encoder(self):
        network_config = NetworkEncoderConfig(
            input_dim=self.config.network.input_dim,
            hidden_dims=self.config.network.hidden_dims,
            output_dim=self.config.network.output_dim,
            use_transformer=self.config.network.use_transformer,
            transformer_layers=self.config.network.transformer_layers,
            transformer_heads=self.config.network.transformer_heads,
            dropout=self.config.network.dropout,
            normalize_input=self.config.network.normalize_input,
            temporal_window=self.config.network.temporal_window,
            num_codecs=self.config.network.num_codecs,
            codec_embed_dim=self.config.network.codec_embed_dim,
        )
        self.network_encoder = NetworkEncoder(network_config)
    
    def _init_gateway1(self):
        gateway1_config = FusionGateway1Config(
            audio_dim=self.audio_encoder.get_output_dim(),
            network_dim=self.network_encoder.get_output_dim(),
            hidden_dim=self.config.gateway1.hidden_dim,
            gate_type=self.config.gateway1.gate_type,
            reliability_aware=self.config.gateway1.reliability_aware,
            num_layers=self.config.gateway1.num_layers,
            use_residual_fusion=self.config.gateway1.use_residual_fusion,
            residual_weight=self.config.gateway1.residual_weight,
            gate_bias_init=self.config.gateway1.gate_bias_init,
            log_gate_stats=self.config.gateway1.log_gate_stats,
            ablation_mode=self.config.gateway1.ablation_mode,
        )
        self.gateway1 = Gateway1(gateway1_config)
    
    def _init_classical_branch(self):
        classical_config = BranchClassicalConfig(
            model_type=self.config.classical.model_type,
            xgboost_params=self.config.classical.xgboost_params,
            use_calibration=self.config.classical.use_calibration,
            calibration_method=self.config.classical.calibration_method,
            random_state=self.config.classical.random_state,
            n_jobs=self.config.classical.n_jobs,
            measure_latency=self.config.classical.measure_latency,
            calibration_cv=self.config.classical.calibration_cv,
            eval_metrics=self.config.classical.eval_metrics,
            serialization_format=self.config.classical.serialization_format,
        )
        fused_dim = self.config.audio.projection_dim
        self.classical_branch = ClassicalBranch(classical_config, fused_dim)
    
    def _init_quantum_branch(self):
        quantum_config = BranchQuantumConfig(
            n_qubits=self.config.quantum.n_qubits,
            n_layers=self.config.quantum.n_layers,
            encoding=self.config.quantum.encoding,
            pca_dim=self.config.quantum.pca_dim,
            device=self.config.quantum.device,
            shots=self.config.quantum.shots,
            use_classical_head=self.config.quantum.use_classical_head,
            classical_head_hidden=self.config.quantum.classical_head_hidden,
            random_state=self.config.quantum.random_state,
            entanglement=self.config.quantum.entanglement,
            rotation_gates=self.config.quantum.rotation_gates,
            measure_latency=self.config.quantum.measure_latency,
            checkpointing=self.config.quantum.checkpointing,
            fallback_mode=self.config.quantum.fallback_mode,
            serialization_format=self.config.quantum.serialization_format,
        )
        fused_dim = self.config.audio.projection_dim
        self.quantum_branch = QuantumBranch(quantum_config, fused_dim)
    
    def _init_gateway2(self):
        gateway2_config = FusionGateway2Config(
            input_dim=3,  # p_c, p_q, |p_c - p_q|
            hidden_dim=self.config.gateway2.hidden_dim,
            gate_type=self.config.gateway2.gate_type,
            dropout=self.config.gateway2.dropout,
            num_layers=self.config.gateway2.num_layers,
            gate_bias_init=self.config.gateway2.gate_bias_init,
            log_gate_stats=self.config.gateway2.log_gate_stats,
            enable_classical=self.config.gateway2.enable_classical,
            enable_quantum=self.config.gateway2.enable_quantum,
            ablation_mode=self.config.gateway2.ablation_mode,
        )
        self.gateway2 = Gateway2(gateway2_config)
    
    def _init_calibrator(self):
        calib_config = CalibratorCalibrationConfig(
            method=self.config.calibration.method,
            temperature_init=self.config.calibration.temperature_init,
            max_iter=self.config.calibration.max_iter,
            lr=self.config.calibration.lr,
            fit_on_validation_only=self.config.calibration.fit_on_validation_only,
            compute_ece=self.config.calibration.compute_ece,
            n_bins_ece=self.config.calibration.n_bins_ece,
            compute_brier=self.config.calibration.compute_brier,
        )
        self.calibrator = Calibrator(calib_config)
    
    def _init_decision_gate(self):
        gate_config = DecisionGateDecisionConfig(
            tau_low=self.config.decision_gate.tau_low,
            tau_high=self.config.decision_gate.tau_high,
            optimize_thresholds=self.config.decision_gate.optimize_thresholds,
            metric_for_optimization=self.config.decision_gate.metric_for_optimization,
            grid_resolution=self.config.decision_gate.grid_resolution,
            select_on_validation=self.config.decision_gate.select_on_validation,
            uncertainty_action=self.config.decision_gate.uncertainty_action,
            min_uncertainty_width=self.config.decision_gate.min_uncertainty_width,
            max_uncertainty_width=self.config.decision_gate.max_uncertainty_width,
        )
        self.decision_gate = DecisionGate(gate_config)
    
    def _verify_tensor(self, name: str, tensor: torch.Tensor, expected_shape: Optional[Tuple] = None,
                       min_val: Optional[float] = None, max_val: Optional[float] = None):
        """Verify tensor properties."""
        if not self.verify_shapes:
            return
        
        # Check for NaN/Inf
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN detected in {name}")
        if torch.isinf(tensor).any():
            raise ValueError(f"Inf detected in {name}")
        
        # Check shape
        if expected_shape is not None:
            if tensor.shape != expected_shape:
                raise ValueError(f"Shape mismatch in {name}: expected {expected_shape}, got {tensor.shape}")
        
        # Check range
        if min_val is not None and tensor.min() < min_val:
            raise ValueError(f"Value below {min_val} in {name}: min={tensor.min().item()}")
        if max_val is not None and tensor.max() > max_val:
            raise ValueError(f"Value above {max_val} in {name}: max={tensor.max().item()}")
    
    def _record_latency(self, name: str, start_time: float):
        """Record latency for a component."""
        elapsed = (time.perf_counter() - start_time) * 1000
        if name not in self.latency_tracker:
            self.latency_tracker[name] = []
        self.latency_tracker[name].append(elapsed)
        return elapsed
    
    def _get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        stats = {}
        total = 0.0
        for name, latencies in self.latency_tracker.items():
            if latencies:
                arr = np.array(latencies)
                stats[f"{name}_mean_ms"] = float(arr.mean())
                stats[f"{name}_last_ms"] = float(arr[-1])
                total += arr[-1]
        stats["total_ms"] = total
        return stats
    
    def fit_classical(self, z_fused: torch.Tensor, labels: torch.Tensor) -> "InferencePipeline":
        """Fit classical branch on fused embeddings."""
        self.classical_branch.fit(z_fused, labels)
        return self
    
    def calibrate_classical(self, z_fused: torch.Tensor, labels: torch.Tensor) -> "InferencePipeline":
        """Calibrate classical branch."""
        self.classical_branch.calibrate(z_fused, labels)
        return self
    
    def fit_quantum(self, z_fused: torch.Tensor) -> "InferencePipeline":
        """Fit quantum branch PCA."""
        self.quantum_branch.fit_pca(z_fused)
        return self
    
    def fit_calibration(self, probs: torch.Tensor, labels: torch.Tensor,
                        val_probs: Optional[torch.Tensor] = None,
                        val_labels: Optional[torch.Tensor] = None) -> "InferencePipeline":
        """Fit calibration on validation data."""
        self.calibrator.fit(probs, labels, val_probs, val_labels)
        return self
    
    def optimize_decision_thresholds(self, probs: torch.Tensor, labels: torch.Tensor,
                                     val_probs: Optional[torch.Tensor] = None,
                                     val_labels: Optional[torch.Tensor] = None) -> "InferencePipeline":
        """Optimize decision thresholds on validation data."""
        self.decision_gate.optimize_thresholds(probs, labels, val_probs, val_labels)
        return self
    
    def load(self, path: str) -> "InferencePipeline":
        """Load pretrained pipeline components."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load neural network components
        if "audio_encoder" in checkpoint:
            self.audio_encoder.load_state_dict(checkpoint["audio_encoder"])
        if "network_encoder" in checkpoint:
            self.network_encoder.load_state_dict(checkpoint["network_encoder"])
        if "gateway1" in checkpoint:
            self.gateway1.load_state_dict(checkpoint["gateway1"])
        if "gateway2" in checkpoint:
            self.gateway2.load_state_dict(checkpoint["gateway2"])
        if "quantum_branch" in checkpoint and not self.quantum_branch.is_using_fallback():
            self.quantum_branch.quantum_circuit.load_state_dict(checkpoint["quantum_branch"])
        if "quantum_head" in checkpoint:
            self.quantum_branch.classical_head.load_state_dict(checkpoint["quantum_head"])
        
        # Load classical branch (pickle)
        if "classical_branch" in checkpoint:
            import pickle
            with open(checkpoint["classical_branch"], "rb") as f:
                self.classical_branch = pickle.load(f)
        
        # Load calibrator state
        if "calibrator" in checkpoint:
            cal_data = checkpoint["calibrator"]
            if "temperature" in cal_data:
                self.calibrator._calibration_method.temperature.data = torch.tensor(cal_data["temperature"])
                self.calibrator._fitted = True
        
        # Load decision gate thresholds
        if "decision_gate" in checkpoint:
            gate_data = checkpoint["decision_gate"]
            self.decision_gate.set_thresholds(gate_data["tau_low"], gate_data["tau_high"])
            self.decision_gate.optimized = gate_data.get("optimized", False)
        
        self.model_version = checkpoint.get("model_version", "1.0.0")
        return self
    
    def save(self, path: str) -> None:
        """Save pipeline components."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "audio_encoder": self.audio_encoder.state_dict(),
            "network_encoder": self.network_encoder.state_dict(),
            "gateway1": self.gateway1.state_dict(),
            "gateway2": self.gateway2.state_dict(),
            "quantum_branch": self.quantum_branch.quantum_circuit.state_dict() if not self.quantum_branch.is_using_fallback() else None,
            "quantum_head": self.quantum_branch.classical_head.state_dict(),
            "model_version": self.model_version,
            "config": self.config,
        }
        
        # Save classical branch separately (pickle)
        import tempfile
        import pickle
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            classical_path = f.name
        self.classical_branch.save(classical_path)
        checkpoint["classical_branch"] = classical_path
        
        # Save calibrator
        if self.calibrator.fitted:
            checkpoint["calibrator"] = self.calibrator.get_fit_stats()
        
        # Save decision gate
        tau_low, tau_high = self.decision_gate.get_thresholds()
        checkpoint["decision_gate"] = {
            "tau_low": tau_low,
            "tau_high": tau_high,
            "optimized": self.decision_gate.optimized,
        }
        
        torch.save(checkpoint, path)
    
    @torch.inference_mode()
    def predict(
        self,
        audio: torch.Tensor,
        network: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        codec_indices: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> InferenceOutput:
        """
        Full inference pipeline.
        
        Args:
            audio: [B, T] or [B, 1, T] raw audio waveform
            network: [B, D_n] or [B, T, D_n] network features
            audio_mask: Optional mask for audio [B, T]
            codec_indices: Optional codec indices for network encoder
            
        Returns:
            InferenceOutput with all predictions and intermediate values
        """
        self.eval()
        self.latency_tracker.clear()
        total_start = time.perf_counter()
        
        batch_size = audio.shape[0]
        output = InferenceOutput(
            prediction=0,
            decision_label="UNKNOWN",
            probability_real=0.5,
            probability_synthetic=0.5,
            confidence=0.5,
        )
        
        try:
            # === Audio Encoding ===
            t0 = time.perf_counter()
            # Audio encoder expects [B, T] - don't unsqueeze
            z_a = self.audio_encoder(audio, audio_mask)
            self._verify_tensor("z_a", z_a, (batch_size, self.config.audio.projection_dim))
            output.latency_ms["audio_encoder_ms"] = self._record_latency("audio_encoder", t0)
            
            # === Network Encoding ===
            t0 = time.perf_counter()
            z_n = self.network_encoder(network, codec_indices=codec_indices)
            self._verify_tensor("z_n", z_n, (batch_size, self.config.network.output_dim))
            output.latency_ms["network_encoder_ms"] = self._record_latency("network_encoder", t0)
            
            # === Gateway 1: Embedding Fusion ===
            t0 = time.perf_counter()
            z_fused, gate1_info = self.gateway1(z_a, z_n, return_components=return_components)
            self._verify_tensor("z_fused", z_fused, (batch_size, self.config.audio.projection_dim))
            output.latency_ms["gateway1_ms"] = self._record_latency("gateway1", t0)
            
            # Gateway 1 weights
            alpha = gate1_info.get("alpha")
            if alpha is not None:
                output.gateway1_audio_weight = alpha[0].item() if alpha.numel() > 1 else alpha.squeeze().item()
                output.gateway1_network_weight = (1 - alpha[0]).item() if alpha.numel() > 1 else (1 - alpha).squeeze().item()
            
            # Reliability scores
            r_a = gate1_info.get("reliability_audio")
            r_n = gate1_info.get("reliability_network")
            if r_a is not None:
                output.audio_reliability = r_a[0].item() if r_a.numel() > 1 else r_a.squeeze().item()
            if r_n is not None:
                output.network_reliability = r_n[0].item() if r_n.numel() > 1 else r_n.squeeze().item()
            
            # === Classical Branch ===
            t0 = time.perf_counter()
            p_c = None
            if self.classical_branch.is_fitted:
                p_c = self.classical_branch(z_fused)
                self._verify_tensor("p_c", p_c, (batch_size,), 0.0, 1.0)
                output.classical_probability = p_c.squeeze().item()
            output.latency_ms["classical_branch_ms"] = self._record_latency("classical_branch", t0)
            
            # === Quantum Branch ===
            t0 = time.perf_counter()
            p_q = None
            if self.quantum_branch.pca_fitted:
                p_q = self.quantum_branch(z_fused)
                self._verify_tensor("p_q", p_q, (batch_size,), 0.0, 1.0)
                output.quantum_probability = p_q.squeeze().item()
            output.latency_ms["quantum_branch_ms"] = self._record_latency("quantum_branch", t0)
            
            # === Gateway 2: Probability Fusion ===
            t0 = time.perf_counter()
            if p_c is not None and p_q is not None:
                p_fused, gate2_info = self.gateway2(p_c, p_q)
                output.gateway2_classical_weight = gate2_info.get("beta", torch.tensor(0.5)).squeeze().item()
                output.gateway2_quantum_weight = 1 - output.gateway2_classical_weight
            elif p_c is not None:
                p_fused = p_c
                output.gateway2_classical_weight = 1.0
                output.gateway2_quantum_weight = 0.0
            elif p_q is not None:
                p_fused = p_q
                output.gateway2_classical_weight = 0.0
                output.gateway2_quantum_weight = 1.0
            else:
                p_fused = torch.full((batch_size,), 0.5, device=self.device)
                output.gateway2_classical_weight = 0.5
                output.gateway2_quantum_weight = 0.5
            
            self._verify_tensor("p_fused", p_fused, (batch_size,), 0.0, 1.0)
            output.fused_probability = p_fused.squeeze().item()
            output.latency_ms["gateway2_ms"] = self._record_latency("gateway2", t0)
            
            # === Calibration ===
            t0 = time.perf_counter()
            if self.calibrator.fitted:
                p_calibrated = self.calibrator(p_fused)
                self._verify_tensor("p_calibrated", p_calibrated, (batch_size,), 0.0, 1.0)
            else:
                p_calibrated = p_fused
            output.calibrated_probability = p_calibrated.squeeze().item()
            output.latency_ms["calibration_ms"] = self._record_latency("calibration", t0)
            
            # === Decision Gate ===
            t0 = time.perf_counter()
            decisions, decision_info = self.decision_gate(p_calibrated)
            self._verify_tensor("decisions", decisions, (batch_size,), 0, 2)
            output.latency_ms["decision_gate_ms"] = self._record_latency("decision_gate", t0)
            
            # Set final outputs
            pred = decisions.squeeze().item()
            output.prediction = pred
            output.decision_label = {0: "REAL", 1: "UNCERTAIN", 2: "SYNTHETIC"}[pred]
            
            prob_synthetic = output.calibrated_probability
            output.probability_synthetic = prob_synthetic
            output.probability_real = 1.0 - prob_synthetic
            output.confidence = max(output.probability_real, output.probability_synthetic)
            
            # Add debug info
            output.debug_info = {
                "gate1_info": {k: v.tolist() if isinstance(v, torch.Tensor) else v 
                              for k, v in gate1_info.items() if not isinstance(v, dict)},
                "decision_info": decision_info,
            }
            if p_c is not None:
                output.debug_info["p_c"] = p_c.squeeze().tolist()
            if p_q is not None:
                output.debug_info["p_q"] = p_q.squeeze().tolist()
            
        except Exception as e:
            output.debug_info["error"] = str(e)
            raise
        
        # Final latency
        output.total_latency_ms = (time.perf_counter() - total_start) * 1000
        output.latency_ms.update(self._get_latency_stats())
        
        return output
    
    @torch.inference_mode()
    def predict_audio_only(
        self,
        audio: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> InferenceOutput:
        """
        Audio-only inference (Gateway 1 with fixed_audio ablation + Classical + Calibration + Decision).
        """
        # Temporarily set ablation mode
        original_mode = self.gateway1.config.ablation_mode
        self.gateway1.config.ablation_mode = "fixed_audio"
        
        try:
            # Create dummy network features
            batch_size = audio.shape[0]
            dummy_network = torch.zeros(batch_size, self.config.network.input_dim, device=self.device)
            
            return self.predict(audio, dummy_network, audio_mask)
        finally:
            self.gateway1.config.ablation_mode = original_mode
    
    @torch.inference_mode()
    def predict_with_network(
        self,
        audio: torch.Tensor,
        network: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        codec_indices: Optional[torch.Tensor] = None,
    ) -> InferenceOutput:
        """
        Inference with both audio and network features.
        """
        return self.predict(audio, network, audio_mask, codec_indices)
    
    def run_smoke_test(self, seed: int = 42, verbose: bool = True) -> Dict[str, Any]:
        """
        Run end-to-end smoke test with synthetic data.
        
        Verifies:
        - Tensor shapes at every boundary
        - Probability ranges [0, 1]
        - No NaN/Inf values
        - Deterministic behavior with fixed seed
        """
        if verbose:
            print(f"\n{'='*60}")
            print("RAFNet Inference Pipeline Smoke Test")
            print(f"{'='*60}")
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate synthetic data
        batch_size = 1
        audio_len = 16000  # 1 second at 16kHz
        network_dim = self.config.network.input_dim
        
        audio = torch.randn(batch_size, audio_len, device=self.device)  # [B, T]
        network = torch.randn(batch_size, 10, network_dim, device=self.device)  # [B, T, D]
        audio_mask = torch.ones(batch_size, audio_len, device=self.device)
        
        if verbose:
            print(f"Input shapes: audio={audio.shape}, network={network.shape}")
        
        # Fit components with synthetic training data
        if verbose:
            print("\nFitting components...")
        
        # Generate training data for fitting
        n_train = 20  # Small for fast smoke test
        train_audio = torch.randn(n_train, audio_len, device=self.device)
        train_network = torch.randn(n_train, 10, network_dim, device=self.device)
        train_labels = torch.randint(0, 2, (n_train,), device=self.device).float()
        
        # Split into train/val for calibration and threshold optimization
        n_val = n_train // 2
        val_audio = train_audio[:n_val]
        val_network = train_network[:n_val]
        val_labels = train_labels[:n_val]
        train_audio = train_audio[n_val:]
        train_network = train_network[n_val:]
        train_labels = train_labels[n_val:]
        
        # Forward pass to get embeddings for fitting
        with torch.no_grad():
            train_z_a = self.audio_encoder(train_audio)
            train_z_n = self.network_encoder(train_network)
            train_z_f, _ = self.gateway1(train_z_a, train_z_n)
            
            val_z_a = self.audio_encoder(val_audio)
            val_z_n = self.network_encoder(val_network)
            val_z_f, _ = self.gateway1(val_z_a, val_z_n)
        
        # Fit classical branch
        self.fit_classical(train_z_f, train_labels.long())
        if verbose:
            print(f"  Classical branch: fitted")
        
        # Fit quantum branch
        self.fit_quantum(train_z_f)
        if verbose:
            print(f"  Quantum branch: PCA fitted")
        
        # Get probabilities for calibration
        with torch.no_grad():
            train_p_c = self.classical_branch(train_z_f)
            train_p_q = self.quantum_branch(train_z_f)
            train_p_fused, _ = self.gateway2(train_p_c, train_p_q)
            
            val_p_c = self.classical_branch(val_z_f)
            val_p_q = self.quantum_branch(val_z_f)
            val_p_fused, _ = self.gateway2(val_p_c, val_p_q)
        
        # Fit calibration
        self.fit_calibration(train_p_fused, train_labels, val_p_fused, val_labels)
        if verbose:
            print(f"  Calibration: fitted (method={self.config.calibration.method})")
        
        # Optimize decision thresholds
        self.optimize_decision_thresholds(train_p_fused, train_labels.long(), val_p_fused, val_labels.long())
        if verbose:
            print(f"  Decision gate: thresholds optimized to ({self.decision_gate.tau_low.item():.3f}, {self.decision_gate.tau_high.item():.3f})")
        
        # Run inference
        if verbose:
            print("\nRunning inference...")
        
        # Test 1: Full pipeline
        torch.manual_seed(seed)
        np.random.seed(seed)
        output1 = self.predict(audio, network, audio_mask)
        
        # Test 2: Deterministic check
        torch.manual_seed(seed)
        np.random.seed(seed)
        output2 = self.predict(audio, network, audio_mask)
        
        # Verify deterministic
        is_deterministic = (
            output1.prediction == output2.prediction and
            abs(output1.calibrated_probability - output2.calibrated_probability) < 1e-6
        )
        
        if verbose:
            print(f"\nOutput 1:")
            print(f"  prediction: {output1.decision_label} ({output1.prediction})")
            print(f"  p_calibrated: {output1.calibrated_probability:.4f}")
            print(f"  p_fused: {output1.fused_probability:.4f}")
            print(f"  p_c: {output1.classical_probability:.4f}")
            print(f"  p_q: {output1.quantum_probability:.4f}")
            print(f"  confidence: {output1.confidence:.4f}")
            print(f"  total_latency_ms: {output1.total_latency_ms:.2f}")
            print(f"\nDeterministic: {is_deterministic}")
        
        # Test 3: Audio-only
        torch.manual_seed(seed)
        np.random.seed(seed)
        audio_only_output = self.predict_audio_only(audio, audio_mask)
        
        if verbose:
            print(f"\nAudio-only output:")
            print(f"  prediction: {audio_only_output.decision_label}")
            print(f"  p_calibrated: {audio_only_output.calibrated_probability:.4f}")
        
        # Verify shapes and ranges
        checks = {
            "probability_range": 0.0 <= output1.calibrated_probability <= 1.0,
            "no_nan": not np.isnan(output1.calibrated_probability),
            "no_inf": not np.isinf(output1.calibrated_probability),
            "deterministic": is_deterministic,
            "decision_valid": output1.prediction in [0, 1, 2],
            "confidence_range": 0.0 <= output1.confidence <= 1.0,
            "probabilities_sum": abs(output1.probability_real + output1.probability_synthetic - 1.0) < 1e-6,
        }
        
        if verbose:
            print("\nVerification Checks:")
            for check, passed in checks.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {check}: {status}")
        
        all_passed = all(checks.values())
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Smoke Test: {'PASSED' if all_passed else 'FAILED'}")
            print(f"{'='*60}\n")
        
        return {
            "passed": all_passed,
            "checks": checks,
            "output": output1.to_dict(),
            "audio_only_output": audio_only_output.to_dict(),
            "latency_stats": output1.latency_ms,
        }


def create_pipeline(
    config: Optional[RAFNetConfig] = None,
    model_path: Optional[str] = None,
    device: str = "cpu",
    seed: int = 42,
) -> InferencePipeline:
    """Factory function to create inference pipeline."""
    return InferencePipeline(config, model_path, device, seed)


if __name__ == "__main__":
    # Run smoke test
    pipeline = InferencePipeline(device="cpu", seed=42)
    result = pipeline.run_smoke_test(verbose=True)
    print(f"\nTest {'PASSED' if result['passed'] else 'FAILED'}")