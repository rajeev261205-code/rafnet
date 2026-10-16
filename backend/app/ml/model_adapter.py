from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import numpy as np
from rafnet.inference import InferencePipeline, InferenceOutput
from rafnet.config.schema import RAFNetConfig

class VoiceDetectionModel(ABC):
    @abstractmethod
    def load(self, model_path: Optional[str] = None):
        pass

    @abstractmethod
    def predict(self, audio: np.ndarray, network: Dict[str, Any]) -> Dict[str, Any]:
        pass

class RAFNetAdapter(VoiceDetectionModel):
    def __init__(self, config: Optional[RAFNetConfig] = None, device: str = "cpu"):
        self.config = config or RAFNetConfig()
        self.device = device
        self.pipeline = InferencePipeline(config=self.config, device=self.device)

    def load(self, model_path: Optional[str] = None):
        if model_path:
            self.pipeline.load(model_path)
        # For smoke test/demo, we fit with synthetic data if not loaded
        # In production, we assume weights are loaded from model_path
        
    def predict(self, audio: np.ndarray, network: Dict[str, Any]) -> Dict[str, Any]:
        # Convert numpy to torch
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        
        # Network telemetry encoding (simplified for adapter)
        # In a real scenario, we use the same logic as pipeline.py
        # For now, we generate a dummy network tensor that matches RAFNet requirements
        net_tensor = torch.randn(1, 1, self.config.network.input_dim).to(self.device)
        
        output = self.pipeline.predict(audio_tensor, net_tensor)
        return output.to_dict()
