import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from copy import deepcopy

from .metrics import evaluate_all, bootstrap_confidence_intervals, EvaluationConfig
from ..model import RAFNet, RAFNetConfig
from ..audio import AudioEncoder
from ..network import NetworkEncoder
from ..fusion import Gateway1, Gateway2
from ..classical import ClassicalBranch
from ..quantum import QuantumBranch
from ..calibration import Calibrator
from ..decision import DecisionGate


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
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)


class AblationModel(nn.Module):
    def __init__(self, name: str, config: RAFNetConfig):
        super().__init__()
        self.name = name
        self.config = config
        self._build_model()
    
    def _build_model(self):
        if self.name == "M1_audio_only":
            self.audio_encoder = AudioEncoder(self.config.audio)
            self.classifier = nn.Linear(self.config.audio.projection_dim, 1)
            
        elif self.name == "M2_network_only":
            self.network_encoder = NetworkEncoder(self.config.network)
            self.classifier = nn.Linear(self.config.network.output_dim, 1)
            
        elif self.name == "M3_fixed_fusion":
            self.audio_encoder = AudioEncoder(self.config.audio)
            self.network_encoder = NetworkEncoder(self.config.network)
            fused_dim = self.config.audio.projection_dim + self.config.network.output_dim
            self.fusion = nn.Linear(fused_dim, self.config.audio.projection_dim)
            self.classifier = nn.Linear(self.config.audio.projection_dim, 1)
            
        elif self.name == "M4_gateway1":
            self.audio_encoder = AudioEncoder(self.config.audio)
            self.network_encoder = NetworkEncoder(self.config.network)
            self.gateway1 = Gateway1(self.config.gateway1)
            self.classifier = nn.Linear(self.config.audio.projection_dim, 1)
            
        elif self.name == "M5_gateway1_classical_quantum":
            self.audio_encoder = AudioEncoder(self.config.audio)
            self.network_encoder = NetworkEncoder(self.config.network)
            self.gateway1 = Gateway1(self.config.gateway1)
            self.classical_branch = ClassicalBranch(self.config.classical, self.config.audio.projection_dim)
            self.quantum_branch = QuantumBranch(self.config.quantum, self.config.audio.projection_dim)
            self.gateway2 = Gateway2(self.config.gateway2)
            
        elif self.name == "M6_gateway2":
            self.audio_encoder = AudioEncoder(self.config.audio)
            self.network_encoder = NetworkEncoder(self.config.network)
            self.gateway1 = Gateway1(self.config.gateway1)
            self.classical_branch = ClassicalBranch(self.config.classical, self.config.audio.projection_dim)
            self.quantum_branch = QuantumBranch(self.config.quantum, self.config.audio.projection_dim)
            self.gateway2 = Gateway2(self.config.gateway2)
            self.calibrator = Calibrator(self.config.calibration)
            
        elif self.name == "M7_complete":
            from ..model import RAFNet
            self.full_model = RAFNet(self.config)
            
        else:
            raise ValueError(f"Unknown ablation model: {self.name}")
    
    def forward(self, audio: torch.Tensor, network: torch.Tensor, audio_mask: torch.Tensor = None) -> torch.Tensor:
        if self.name == "M1_audio_only":
            z = self.audio_encoder(audio, audio_mask)
            return torch.sigmoid(self.classifier(z)).squeeze(-1)
        
        elif self.name == "M2_network_only":
            z = self.network_encoder(network)
            return torch.sigmoid(self.classifier(z)).squeeze(-1)
        
        elif self.name == "M3_fixed_fusion":
            z_a = self.audio_encoder(audio, audio_mask)
            z_n = self.network_encoder(network)
            z_fused = self.fusion(torch.cat([z_a, z_n], dim=-1))
            return torch.sigmoid(self.classifier(z_fused)).squeeze(-1)
        
        elif self.name == "M4_gateway1":
            z_a = self.audio_encoder(audio, audio_mask)
            z_n = self.network_encoder(network)
            z_fused, _ = self.gateway1(z_a, z_n)
            return torch.sigmoid(self.classifier(z_fused)).squeeze(-1)
        
        elif self.name == "M5_gateway1_classical_quantum":
            z_a = self.audio_encoder(audio, audio_mask)
            z_n = self.network_encoder(network)
            z_fused, _ = self.gateway1(z_a, z_n)
            p_c = self.classical_branch(z_fused) if hasattr(self, 'classical_branch') and self.classical_branch.is_fitted else torch.full((z_fused.shape[0],), 0.5, device=z_fused.device)
            p_q = self.quantum_branch(z_fused) if hasattr(self, 'quantum_branch') and self.quantum_branch.pca_fitted else torch.full((z_fused.shape[0],), 0.5, device=z_fused.device)
            p_fused, _ = self.gateway2(p_c, p_q)
            return p_fused
        
        elif self.name == "M6_gateway2":
            z_a = self.audio_encoder(audio, audio_mask)
            z_n = self.network_encoder(network)
            z_fused, _ = self.gateway1(z_a, z_n)
            p_c = self.classical_branch(z_fused) if hasattr(self, 'classical_branch') and self.classical_branch.is_fitted else torch.full((z_fused.shape[0],), 0.5, device=z_fused.device)
            p_q = self.quantum_branch(z_fused) if hasattr(self, 'quantum_branch') and self.quantum_branch.pca_fitted else torch.full((z_fused.shape[0],), 0.5, device=z_fused.device)
            p_fused, _ = self.gateway2(p_c, p_q)
            if hasattr(self, 'calibrator') and self.calibrator.fitted:
                return self.calibrator(p_fused)
            return p_fused
        
        elif self.name == "M7_complete":
            output = self.full_model(audio, network, audio_mask)
            return output.probs
        
        return torch.full((audio.shape[0],), 0.5, device=audio.device)


def run_ablation_study(
    config: RAFNetConfig,
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    ablation_config: AblationConfig = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    if ablation_config is None:
        ablation_config = AblationConfig()
    
    results = {}
    
    for model_name in ablation_config.models:
        print(f"\nRunning ablation model: {model_name}")
        model = AblationModel(model_name, config).to(device)
        
        if model_name in ["M1_audio_only", "M2_network_only", "M3_fixed_fusion", "M4_gateway1"]:
            model.eval()
            with torch.no_grad():
                probs = model(
                    test_data["audio"].to(device),
                    test_data["network"].to(device),
                    test_data.get("audio_mask", None).to(device) if test_data.get("audio_mask") is not None else None
                ).cpu().numpy()
            
            metrics = evaluate_all(
                test_data["labels"].numpy(),
                probs,
                config=ablation_config.evaluation_config
            )
            results[model_name] = metrics
            
        elif model_name in ["M5_gateway1_classical_quantum", "M6_gateway2", "M7_complete"]:
            print(f"  {model_name} requires fitting - skipping full evaluation in ablation framework")
            results[model_name] = {"status": "requires_fitting"}
    
    return results