import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .config.schema import RAFNetConfig
from .audio import AudioEncoder, AudioEncoderConfig
from .network import NetworkEncoder, NetworkEncoderConfig
from .fusion import Gateway1, Gateway1Config, Gateway2, Gateway2Config
from .classical import ClassicalBranch, ClassicalConfig
from .quantum import QuantumBranch, QuantumConfig
from .calibration import Calibrator, CalibrationConfig
from .decision import DecisionGate, DecisionGateConfig
from .losses import RAFNetLoss, LossConfig


@dataclass
class RAFNetOutput:
    probs: torch.Tensor
    decisions: torch.Tensor
    p_c: Optional[torch.Tensor] = None
    p_q: Optional[torch.Tensor] = None
    p_fused: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None
    r_a: Optional[torch.Tensor] = None
    r_n: Optional[torch.Tensor] = None
    z_fused: Optional[torch.Tensor] = None
    gate1_info: Optional[Dict] = None
    gate2_info: Optional[Dict] = None
    decision_info: Optional[Dict] = None


class RAFNet(nn.Module):
    def __init__(self, config: RAFNetConfig):
        super().__init__()
        self.config = config
        
        audio_config = AudioEncoderConfig(
            sample_rate=config.audio.sample_rate,
            wavlm_model=config.audio.wavlm_model,
            wavlm_freeze=config.audio.wavlm_freeze,
            wavlm_output_dim=config.audio.wavlm_output_dim,
            projection_dim=config.audio.projection_dim,
            use_handcrafted=config.audio.use_handcrafted,
            handcrafted_features=config.audio.handcrafted_features,
            device=config.device,
        )
        self.audio_encoder = AudioEncoder(audio_config)
        
        network_config = NetworkEncoderConfig(
            input_dim=config.network.input_dim,
            hidden_dims=config.network.hidden_dims,
            output_dim=config.network.output_dim,
            use_transformer=config.network.use_transformer,
            transformer_layers=config.network.transformer_layers,
            transformer_heads=config.network.transformer_heads,
            dropout=config.network.dropout,
            normalize_input=config.network.normalize_input,
        )
        self.network_encoder = NetworkEncoder(network_config)
        
        gateway1_config = Gateway1Config(
            audio_dim=self.audio_encoder.get_output_dim(),
            network_dim=self.network_encoder.get_output_dim(),
            hidden_dim=config.gateway1.hidden_dim,
            gate_type=config.gateway1.gate_type,
            reliability_aware=config.gateway1.reliability_aware,
            num_layers=config.gateway1.num_layers,
            use_residual_fusion=config.gateway1.use_residual_fusion,
            residual_weight=config.gateway1.residual_weight,
            gate_bias_init=config.gateway1.gate_bias_init,
            log_gate_stats=config.gateway1.log_gate_stats,
            ablation_mode=config.gateway1.ablation_mode,
        )
        self.gateway1 = Gateway1(gateway1_config)
        
        fused_dim = config.audio.projection_dim
        
        classical_config = ClassicalConfig(
            model_type=config.classical.model_type,
            xgboost_params=config.classical.xgboost_params,
            use_calibration=config.classical.use_calibration,
        )
        self.classical_branch = ClassicalBranch(classical_config, fused_dim)
        
        quantum_config = QuantumConfig(
            n_qubits=config.quantum.n_qubits,
            n_layers=config.quantum.n_layers,
            encoding=config.quantum.encoding,
            pca_dim=config.quantum.pca_dim,
            device=config.quantum.device,
            shots=config.quantum.shots,
            use_classical_head=config.quantum.use_classical_head,
            classical_head_hidden=config.quantum.classical_head_hidden,
        )
        self.quantum_branch = QuantumBranch(quantum_config, fused_dim)
        
        gateway2_config = Gateway2Config(
            input_dim=3,
            hidden_dim=config.gateway2.hidden_dim,
            gate_type=config.gateway2.gate_type,
            dropout=config.gateway2.dropout,
            num_layers=config.gateway2.num_layers,
            gate_bias_init=config.gateway2.gate_bias_init,
            log_gate_stats=config.gateway2.log_gate_stats,
            enable_classical=config.gateway2.enable_classical,
            enable_quantum=config.gateway2.enable_quantum,
            ablation_mode=config.gateway2.ablation_mode,
        )
        self.gateway2 = Gateway2(gateway2_config)
        
        calibration_config = CalibrationConfig(
            method=config.calibration.method,
            temperature_init=config.calibration.temperature_init,
            max_iter=config.calibration.max_iter,
            lr=config.calibration.lr,
            fit_on_validation_only=config.calibration.fit_on_validation_only,
            compute_ece=config.calibration.compute_ece,
            n_bins_ece=config.calibration.n_bins_ece,
            compute_brier=config.calibration.compute_brier,
        )
        self.calibrator = Calibrator(calibration_config)
        
        decision_config = DecisionGateConfig(
            tau_low=config.decision_gate.tau_low,
            tau_high=config.decision_gate.tau_high,
            optimize_thresholds=config.decision_gate.optimize_thresholds,
            metric_for_optimization=config.decision_gate.metric_for_optimization,
            grid_resolution=config.decision_gate.grid_resolution,
            select_on_validation=config.decision_gate.select_on_validation,
            uncertainty_action=config.decision_gate.uncertainty_action,
            min_uncertainty_width=config.decision_gate.min_uncertainty_width,
            max_uncertainty_width=config.decision_gate.max_uncertainty_width,
        )
        self.decision_gate = DecisionGate(decision_config)
        
        loss_config = LossConfig(
            ce_weight=config.loss.ce_weight,
            rel_weight=config.loss.rel_weight,
            selcal_weight=config.loss.selcal_weight,
            agree_weight=config.loss.agree_weight,
        )
        self.loss_fn = RAFNetLoss(loss_config)
        
        self.classical_fitted = False
        self.quantum_fitted = False
        self.calibration_fitted = False
    
    def forward(
        self,
        audio: torch.Tensor,
        network: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> RAFNetOutput:
        z_a = self.audio_encoder(audio, audio_mask)
        z_n = self.network_encoder(network)
        
        z_fused, gate1_info = self.gateway1(z_a, z_n)
        
        p_c = None
        p_q = None
        
        if self.classical_fitted:
            p_c = self.classical_branch(z_fused)
        
        if self.quantum_fitted:
            p_q = self.quantum_branch(z_fused)
        
        if p_c is not None and p_q is not None:
            p_fused, gate2_info = self.gateway2(p_c, p_q)
        elif p_c is not None:
            p_fused = p_c
            gate2_info = {"beta": torch.ones_like(p_c), "p_c": p_c, "p_q": torch.zeros_like(p_c)}
        elif p_q is not None:
            p_fused = p_q
            gate2_info = {"beta": torch.zeros_like(p_q), "p_c": torch.zeros_like(p_q), "p_q": p_q}
        else:
            p_fused = torch.full((z_fused.shape[0],), 0.5, device=z_fused.device)
            gate2_info = {"beta": torch.full((z_fused.shape[0],), 0.5, device=z_fused.device)}
        
        if self.calibration_fitted:
            probs = self.calibrator(p_fused)
        else:
            probs = p_fused
        
        decisions, decision_info = self.decision_gate(probs)
        
        output = RAFNetOutput(
            probs=probs,
            decisions=decisions,
            p_c=p_c,
            p_q=p_q,
            p_fused=p_fused,
            alpha=gate1_info.get("alpha"),
            beta=gate2_info.get("beta"),
            r_a=gate1_info.get("reliability_audio"),
            r_n=gate1_info.get("reliability_network"),
            z_fused=z_fused,
            gate1_info=gate1_info,
            gate2_info=gate2_info,
            decision_info=decision_info,
        )
        
        return output
    
    def fit_classical(self, z_fused: torch.Tensor, labels: torch.Tensor) -> "RAFNet":
        self.classical_branch.fit(z_fused, labels)
        self.classical_fitted = True
        return self
    
    def calibrate_classical(self, z_fused: torch.Tensor, labels: torch.Tensor) -> "RAFNet":
        if not self.classical_fitted:
            raise RuntimeError("Classical branch must be fitted before calibration")
        self.classical_branch.calibrate(z_fused, labels)
        return self
    
    def fit_quantum(self, z_fused: torch.Tensor) -> "RAFNet":
        self.quantum_branch.fit_pca(z_fused)
        self.quantum_fitted = True
        return self
    
    def fit_calibration(
        self, 
        probs: torch.Tensor, 
        labels: torch.Tensor,
        validation_probs: torch.Tensor,
        validation_labels: torch.Tensor
    ) -> "RAFNet":
        self.calibrator.fit(probs, labels, validation_probs, validation_labels)
        self.calibration_fitted = True
        return self
    
    def optimize_decision_thresholds(
        self, 
        probs: torch.Tensor, 
        labels: torch.Tensor,
        validation_probs: torch.Tensor,
        validation_labels: torch.Tensor
    ) -> "RAFNet":
        self.decision_gate.optimize_thresholds(probs, labels, validation_probs, validation_labels)
        return self
    
    def compute_loss(
        self,
        output: RAFNetOutput,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.loss_fn(
            probs=output.probs,
            labels=labels,
            p_c=output.p_c,
            p_q=output.p_q,
            r_a=output.r_a,
            r_n=output.r_n,
            z_a=None,
            z_n=None,
            alpha=output.alpha,
            tau_low=self.decision_gate.tau_low.item(),
            tau_high=self.decision_gate.tau_high.item(),
        )
    
    def get_trainable_parameters(self):
        params = []
        params.extend(self.audio_encoder.parameters())
        params.extend(self.network_encoder.parameters())
        params.extend(self.gateway1.parameters())
        params.extend(self.gateway2.parameters())
        if hasattr(self.quantum_branch, 'quantum_circuit') and hasattr(self.quantum_branch.quantum_circuit, 'weights'):
            params.append(self.quantum_branch.quantum_circuit.weights)
        params.extend(self.classical_head.parameters()) if hasattr(self, 'classical_head') else None
        return params