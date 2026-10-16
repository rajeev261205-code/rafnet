from typing import Optional, Dict, Any, List
import asyncio
import logging
from datetime import datetime
import numpy as np
import torch

from rafnet.inference import InferenceEngine, LatencyTarget, SimulatedStream
from rafnet.config.schema import RAFNetConfig
from rafnet.api.schemas import (
    PredictRequest, PredictResponse, HealthResponse, ModelInfoResponse,
    VersionResponse, ErrorResponse, AudioInput, NetworkTelemetryInput
)

logger = logging.getLogger(__name__)


class InferenceService:
    """Service layer for inference operations."""
    
    def __init__(
        self,
        config: Optional[RAFNetConfig] = None,
        device: str = "cpu",
        latency_target: Optional[LatencyTarget] = None,
        audio_window_seconds: float = 1.0,
        max_queue_size: int = 100,
    ):
        self.config = config or RAFNetConfig()
        self.device = device
        self.latency_target = latency_target or LatencyTarget()
        self.audio_window_seconds = audio_window_seconds
        self.max_queue_size = max_queue_size
        
        self._engine: Optional[InferenceEngine] = None
        self._is_initialized = False
        self._model_version = "1.0.0"
        self._build_timestamp = datetime.utcnow().isoformat()
        
        # Prediction history for logging
        self._prediction_log: List[Dict] = []
        self._max_log_size = 10000
        
        # Calibration status
        self._calibration_fitted = False
        self._thresholds_optimized = False
    
    async def initialize(self) -> None:
        """Initialize the inference engine."""
        if self._is_initialized:
            return
        
        logger.info("Initializing inference engine...")
        
        self._engine = InferenceEngine(
            config=self.config,
            device=self.device,
            latency_target=self.latency_target,
            audio_window_seconds=self.audio_window_seconds,
            max_queue_size=self.max_queue_size,
        )
        
        # Start engine in background thread
        await asyncio.get_event_loop().run_in_executor(None, self._engine.start)
        
        self._is_initialized = True
        logger.info("Inference engine initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the inference engine."""
        if self._engine:
            await asyncio.get_event_loop().run_in_executor(None, self._engine.stop)
        logger.info("Inference engine shut down")
    
    def is_ready(self) -> bool:
        """Check if service is ready for inference."""
        return self._is_initialized and self._engine and self._engine.is_healthy()
    
    def get_health(self) -> HealthResponse:
        """Get health status."""
        engine_healthy = self._engine.is_healthy() if self._engine else False
        components = {}
        
        if self._engine:
            components = {
                "audio_encoder": True,
                "network_encoder": True,
                "gateway1": True,
                "classical_branch": self._engine._classical_branch.is_fitted if hasattr(self._engine, '_classical_branch') else False,
                "quantum_branch": self._engine._quantum_branch.pca_fitted if hasattr(self._engine, '_quantum_branch') else False,
                "gateway2": True,
                "calibrator": self._engine._calibrator.fitted if hasattr(self._engine, '_calibrator') else False,
                "decision_gate": self._engine._decision_gate.optimized if hasattr(self._engine, '_decision_gate') else False,
            }
        
        return HealthResponse(
            status="healthy" if engine_healthy else "degraded",
            engine_healthy=engine_healthy,
            model_loaded=self._is_initialized,
            components=components,
            timestamp=datetime.utcnow(),
        )
    
    def get_model_info(self) -> ModelInfoResponse:
        """Get model information."""
        calibration_method = "none"
        if self._engine and self._engine._calibrator.fitted:
            calibration_method = self._engine._calibrator._calibration_method.method if hasattr(self._engine._calibrator, '_calibration_method') else "temperature"
        
        decision_thresholds = {}
        if self._engine and self._engine._decision_gate.optimized:
            decision_thresholds = {
                "tau_low": self._engine._decision_gate.tau_low.item(),
                "tau_high": self._engine._decision_gate.tau_high.item(),
            }
        
        components = {}
        if self._engine:
            components = {
                "audio_encoder": "WavLM + Handcrafted",
                "network_encoder": "GRU + MLP",
                "gateway1": "Reliability-aware fusion",
                "classical_branch": self.config.classical.model_type,
                "quantum_branch": f"VQC ({self.config.quantum.n_qubits} qubits, {self.config.quantum.n_layers} layers)",
                "gateway2": "Attention fusion",
                "calibrator": calibration_method,
                "decision_gate": "Three-way threshold",
            }
        
        latency_targets = {
            "preprocessing_ms": self.latency_target.preprocessing,
            "audio_encoder_ms": self.latency_target.audio_encoder,
            "network_encoder_ms": self.latency_target.network_encoder,
            "gateway1_ms": self.latency_target.gateway1,
            "classical_ms": self.latency_target.classical,
            "quantum_ms": self.latency_target.quantum,
            "gateway2_ms": self.latency_target.gateway2,
            "calibration_ms": self.latency_target.calibration,
            "decision_gate_ms": self.latency_target.decision_gate,
            "total_ms": self.latency_target.total,
        }
        
        return ModelInfoResponse(
            model_version=self._model_version,
            model_type="RAFNet",
            components=components,
            latency_targets=latency_targets,
            calibration_method=calibration_method,
            decision_thresholds=decision_thresholds,
            device=self.device,
        )
    
    def get_version(self) -> VersionResponse:
        """Get version information."""
        return VersionResponse(
            api_version="v1",
            model_version=self._model_version,
            build_timestamp=self._build_timestamp,
        )
    
    def _convert_audio(self, audio: AudioInput) -> np.ndarray:
        """Convert audio input to numpy array."""
        return np.array(audio.data, dtype=np.float32)
    
    def _convert_network(self, network: NetworkTelemetryInput) -> Dict[str, Any]:
        """Convert network telemetry to dict."""
        return {
            "packet_loss": network.packet_loss,
            "jitter": network.jitter,
            "rtt": network.rtt,
            "bitrate": network.bitrate,
            "codec": network.codec,
            "packet_timing": network.packet_timing,
            "packet_size": network.packet_size,
            "burst_loss_count": network.burst_loss_count,
            "burst_loss_duration": network.burst_loss_duration,
            "loss_burstiness": network.loss_burstiness,
        }
    
    def _build_response(self, result, audio: AudioInput) -> PredictResponse:
        """Build prediction response from engine result."""
        stage_latencies = []
        for sl in result.stage_latencies:
            stage_latencies.append({
                "stage": sl.stage_name,
                "latency_ms": sl.latency_ms,
                "target_ms": sl.target_ms,
                "within_target": sl.within_target,
            })
        
        return PredictResponse(
            decision=result.decision,
            decision_label=result.decision_label,
            probability_real=result.probability_real,
            probability_synthetic=result.probability_synthetic,
            confidence=result.confidence,
            audio_reliability=result.audio_reliability,
            network_reliability=result.network_reliability,
            gateway1_audio_weight=result.gateway1_audio_weight,
            gateway1_network_weight=result.gateway1_network_weight,
            classical_probability=result.classical_probability,
            quantum_probability=result.quantum_probability,
            gateway2_classical_weight=result.gateway2_classical_weight,
            gateway2_quantum_weight=result.gateway2_quantum_weight,
            fused_probability=result.fused_probability,
            calibrated_probability=result.calibrated_probability,
            total_latency_ms=result.total_latency_ms,
            stage_latencies=stage_latencies,
            within_total_target=result.within_total_target,
            timestamp=datetime.fromtimestamp(result.timestamp),
            frame_id=result.frame_id,
        )
    
    def _log_prediction(self, result, audio: AudioInput, request: PredictRequest) -> None:
        """Log prediction for audit trail."""
        log_entry = {
            "frame_id": result.frame_id,
            "timestamp": datetime.fromtimestamp(result.timestamp).isoformat(),
            "decision": result.decision,
            "decision_label": result.decision_label,
            "probability_synthetic": result.calibrated_probability,
            "total_latency_ms": result.total_latency_ms,
            "within_target": result.within_total_target,
            "audio_frame_id": audio.frame_id,
            "network_packet_loss": request.network.packet_loss,
            "network_jitter": request.network.jitter,
        }
        
        self._prediction_log.append(log_entry)
        if len(self._prediction_log) > self._max_log_size:
            self._prediction_log = self._prediction_log[-self._max_log_size:]
    
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """Run inference on audio and network data."""
        if not self.is_ready():
            raise RuntimeError("Inference engine not ready")
        
        # Convert inputs
        audio_data = self._convert_audio(request.audio)
        network_data = self._convert_network(request.network)
        
        # Push to engine
        self._engine.push_audio(audio_data, request.audio.sample_rate)
        self._engine.push_network(network_data)
        
        # Wait for result with timeout
        timeout = 5.0  # seconds
        start_wait = asyncio.get_event_loop().time()
        
        while True:
            result = self._engine.get_result(timeout=0.1)
            if result is not None:
                break
            
            if asyncio.get_event_loop().time() - start_wait > timeout:
                raise TimeoutError("Inference timeout")
            
            await asyncio.sleep(0.01)
        
        # Build response
        response = self._build_response(result, request.audio)
        
        # Log prediction
        self._log_prediction(result, request.audio, request)
        
        return response
    
    async def batch_predict(self, requests: List[PredictRequest]) -> List[PredictResponse]:
        """Run batch inference."""
        results = []
        for request in requests:
            result = await self.predict(request)
            results.append(result)
        return results
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics."""
        if self._engine:
            return self._engine.get_latency_stats()
        return {}
    
    def get_prediction_log(self, limit: int = 100) -> List[Dict]:
        """Get recent prediction log."""
        return self._prediction_log[-limit:]
    
    def fit_classical(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Fit classical branch on embeddings."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        
        import torch
        self._engine.fit_classical(torch.from_numpy(embeddings).float(), torch.from_numpy(labels).long())
        logger.info("Classical branch fitted")
    
    def fit_quantum(self, embeddings: np.ndarray) -> None:
        """Fit quantum branch PCA."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        
        import torch
        self._engine.fit_quantum(torch.from_numpy(embeddings).float())
        logger.info("Quantum branch PCA fitted")
    
    def fit_calibration(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        val_probs: np.ndarray,
        val_labels: np.ndarray
    ) -> None:
        """Fit calibration on validation data."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        
        import torch
        self._engine.fit_calibration(
            torch.from_numpy(probs).float(),
            torch.from_numpy(labels).float(),
            torch.from_numpy(val_probs).float(),
            torch.from_numpy(val_labels).float(),
        )
        self._calibration_fitted = True
        logger.info("Calibration fitted")
    
    def optimize_thresholds(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        val_probs: np.ndarray,
        val_labels: np.ndarray
    ) -> None:
        """Optimize decision thresholds on validation data."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        
        import torch
        self._engine.optimize_decision_thresholds(
            torch.from_numpy(probs).float(),
            torch.from_numpy(labels).long(),
            torch.from_numpy(val_probs).float(),
            torch.from_numpy(val_labels).long(),
        )
        self._thresholds_optimized = True
        logger.info("Decision thresholds optimized")


# Global service instance
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get global inference service instance."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service


async def initialize_service(
    config: Optional[RAFNetConfig] = None,
    device: str = "cpu",
    latency_target: Optional[LatencyTarget] = None,
    audio_window_seconds: float = 1.0,
    max_queue_size: int = 100,
) -> InferenceService:
    """Initialize global inference service."""
    global _inference_service
    _inference_service = InferenceService(
        config=config,
        device=device,
        latency_target=latency_target,
        audio_window_seconds=audio_window_seconds,
        max_queue_size=max_queue_size,
    )
    await _inference_service.initialize()
    return _inference_service