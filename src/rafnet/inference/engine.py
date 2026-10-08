"""
Real-time Inference Engine for RAFNet.

Processes streaming audio (WebRTC/RTP) and network telemetry
through the full RAFNet pipeline with latency instrumentation.

Pipeline:
    WebRTC/RTP → Rolling window (1-2s) → Audio Encoder
    Network Telemetry → Rolling window → Network Encoder
    → Gateway 1 → Classical → Quantum → Gateway 2 → Calibration → Decision Gate

Latency Target: < 100ms end-to-end per inference step
"""

import time
import threading
import queue
import collections
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class LatencyTarget:
    """Explicit latency targets for each pipeline stage (in milliseconds)."""
    preprocessing: float = 5.0      # Audio frame preprocessing
    audio_encoder: float = 30.0     # WavLM + projection
    network_encoder: float = 5.0    # Network telemetry encoding
    gateway1: float = 5.0           # Embedding fusion
    classical: float = 10.0         # XGBoost/Logistic inference
    quantum: float = 15.0           # PCA + VQC + head
    gateway2: float = 3.0           # Probability fusion
    calibration: float = 2.0        # Temperature/Platt/Isotonic
    decision_gate: float = 1.0      # Threshold comparison
    total: float = 76.0             # End-to-end target


@dataclass
class StageLatency:
    """Latency measurement for a single pipeline stage."""
    stage_name: str
    latency_ms: float
    target_ms: float
    within_target: bool
    timestamp: float
    
    def to_dict(self) -> Dict:
        return {
            "stage": self.stage_name,
            "latency_ms": round(self.latency_ms, 2),
            "target_ms": self.target_ms,
            "within_target": self.within_target,
            "timestamp": self.timestamp,
        }


@dataclass
class InferenceResult:
    """Complete result from one inference step."""
    decision: int  # 0=REAL, 1=UNCERTAIN, 2=SYNTHETIC
    decision_label: str
    probability_real: float
    probability_synthetic: float
    confidence: float
    audio_reliability: Optional[float]
    network_reliability: Optional[float]
    gateway1_audio_weight: Optional[float]
    gateway1_network_weight: Optional[float]
    classical_probability: Optional[float]
    quantum_probability: Optional[float]
    gateway2_classical_weight: Optional[float]
    gateway2_quantum_weight: Optional[float]
    fused_probability: Optional[float]
    calibrated_probability: float
    stage_latencies: List[StageLatency]
    total_latency_ms: float
    within_total_target: bool
    timestamp: float
    frame_id: int


class RollingBuffer:
    """Thread-safe rolling buffer for continuous signal with configurable overlap."""
    
    def __init__(self, max_samples: int, overlap_samples: int = 0):
        self.max_samples = max_samples
        self.overlap_samples = overlap_samples
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
    
    def append(self, samples: np.ndarray) -> None:
        """Append new samples to buffer."""
        with self.lock:
            self.buffer = np.concatenate([self.buffer, samples])
            if len(self.buffer) > self.max_samples:
                self.buffer = self.buffer[-self.max_samples:]
    
    def get_window(self, window_samples: int) -> np.ndarray:
        """Get latest window_samples as contiguous array."""
        with self.lock:
            if len(self.buffer) < window_samples:
                return self.buffer.copy()
            return self.buffer[-window_samples:].copy()
    
    def get_with_overlap(self, window_samples: int) -> np.ndarray:
        """Get window with overlap from previous window."""
        with self.lock:
            if len(self.buffer) < window_samples:
                return self.buffer.copy()
            start = max(0, len(self.buffer) - window_samples - self.overlap_samples)
            return self.buffer[start:].copy()
    
    def clear(self) -> None:
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)
    
    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)


class LatencyTracker:
    """Track latency for each pipeline stage with statistics."""
    
    def __init__(self, targets: LatencyTarget):
        self.targets = targets
        self.stage_data: Dict[str, List[float]] = collections.defaultdict(list)
        self.total_latencies: List[float] = []
        self.lock = threading.Lock()
    
    @contextmanager
    def measure(self, stage_name: str):
        """Context manager to measure stage latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            target = getattr(self.targets, stage_name, 100.0)
            with self.lock:
                self.stage_data[stage_name].append(elapsed)
                if stage_name == "total":
                    self.total_latencies.append(elapsed)
    
    def record(self, stage_name: str, latency_ms: float) -> StageLatency:
        """Manually record a latency measurement."""
        target = getattr(self.targets, stage_name, 100.0)
        within = latency_ms <= target
        sl = StageLatency(
            stage_name=stage_name,
            latency_ms=latency_ms,
            target_ms=target,
            within_target=within,
            timestamp=time.time(),
        )
        with self.lock:
            self.stage_data[stage_name].append(latency_ms)
        return sl
    
    def get_stats(self, stage_name: str) -> Dict:
        """Get statistics for a stage."""
        with self.lock:
            data = self.stage_data.get(stage_name, [])
        if not data:
            return {}
        arr = np.array(data)
        target = getattr(self.targets, stage_name, 100.0)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "target_ms": target,
            "within_target_pct": float((arr <= target).mean() * 100),
            "count": len(data),
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all stages."""
        return {stage: self.get_stats(stage) for stage in self.stage_data}
    
    def check_targets(self) -> Dict[str, bool]:
        """Check if all stages meet their targets on average."""
        results = {}
        for stage, target in [
            ("preprocessing", self.targets.preprocessing),
            ("audio_encoder", self.targets.audio_encoder),
            ("network_encoder", self.targets.network_encoder),
            ("gateway1", self.targets.gateway1),
            ("classical", self.targets.classical),
            ("quantum", self.targets.quantum),
            ("gateway2", self.targets.gateway2),
            ("calibration", self.targets.calibration),
            ("decision_gate", self.targets.decision_gate),
            ("total", self.targets.total),
        ]:
            stats = self.get_stats(stage)
            results[stage] = stats.get("mean_ms", float('inf')) <= target
        return results


class AudioPreprocessor:
    """Preprocess raw audio frames from WebRTC/RTP."""
    
    def __init__(self, sample_rate: int = 16000, target_db: float = -26.0):
        self.sample_rate = sample_rate
        self.target_db = target_db
        self._resampler_cache: Dict[int, Any] = {}
    
    def process(self, audio_frame: np.ndarray, input_sample_rate: int = None) -> torch.Tensor:
        """
        Process raw audio frame.
        
        Args:
            audio_frame: [T] or [1, T] audio samples (int16 or float32)
            input_sample_rate: Sample rate of input (if different from target)
        
        Returns:
            [1, T] float32 tensor at target sample rate
        """
        start = time.perf_counter()
        
        # Convert to float32
        if audio_frame.dtype == np.int16:
            audio_frame = audio_frame.astype(np.float32) / 32768.0
        elif audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32)
        
        # Ensure 2D: [1, T]
        if audio_frame.ndim == 1:
            audio_frame = audio_frame[None, :]
        
        # Resample if needed
        if input_sample_rate and input_sample_rate != self.sample_rate:
            import torchaudio.transforms as T
            key = (input_sample_rate, self.sample_rate)
            if key not in self._resampler_cache:
                self._resampler_cache[key] = T.Resample(input_sample_rate, self.sample_rate)
            resampler = self._resampler_cache[key]
            audio_tensor = torch.from_numpy(audio_frame)
            audio_tensor = resampler(audio_tensor)
            audio_frame = audio_tensor.numpy()
        
        # Normalize to target dB (simple RMS normalization)
        rms = np.sqrt(np.mean(audio_frame ** 2))
        if rms > 1e-8:
            target_rms = 10 ** (self.target_db / 20)
            audio_frame = audio_frame * (target_rms / rms)
        
        # Clamp to [-1, 1]
        audio_frame = np.clip(audio_frame, -1.0, 1.0)
        
        latency = (time.perf_counter() - start) * 1000
        return torch.from_numpy(audio_frame).float(), latency


class NetworkTelemetryBuffer:
    """Buffer and process streaming network telemetry."""
    
    def __init__(self, window_seconds: float = 1.0, telemetry_fps: int = 50):
        self.window_seconds = window_seconds
        self.telemetry_fps = telemetry_fps
        self.max_samples = int(window_seconds * telemetry_fps)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.lock = threading.Lock()
    
    def push(self, telemetry: Dict) -> None:
        """Push new telemetry sample."""
        with self.lock:
            # Ensure all expected fields present
            sample = {
                "packet_loss": telemetry.get("packet_loss", 0.0),
                "jitter": telemetry.get("jitter", 0.0),
                "rtt": telemetry.get("rtt", 0.0),
                "bitrate": telemetry.get("bitrate", 64000.0),
                "codec": telemetry.get("codec", "opus"),
                "packet_timing": telemetry.get("packet_timing"),
                "packet_size": telemetry.get("packet_size"),
                "burst_loss_count": telemetry.get("burst_loss_count"),
                "burst_loss_duration": telemetry.get("burst_loss_duration"),
                "loss_burstiness": telemetry.get("loss_burstiness"),
            }
            self.buffer.append(sample)
    
    def get_window(self) -> List[Dict]:
        """Get current window of telemetry."""
        with self.lock:
            return list(self.buffer)
    
    def is_ready(self, min_samples: int = 5) -> bool:
        """Check if enough samples for inference."""
        with self.lock:
            return len(self.buffer) >= min_samples
    
    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()


class InferenceEngine:
    """
    Real-time inference engine for RAFNet.
    
    Processes streaming audio and network telemetry through
    the complete RAFNet pipeline with full latency instrumentation.
    """
    
    def __init__(
        self,
        config: "RAFNetConfig",
        audio_window_seconds: float = 1.0,
        audio_hop_seconds: float = 0.5,
        network_window_seconds: float = 1.0,
        device: str = "cpu",
        latency_target: LatencyTarget = None,
        max_queue_size: int = 100,
    ):
        self.config = config
        self.audio_window_seconds = audio_window_seconds
        self.audio_hop_seconds = audio_hop_seconds
        self.network_window_seconds = network_window_seconds
        self.device = device
        self.targets = latency_target or LatencyTarget()
        self.max_queue_size = max_queue_size
        
        # Audio parameters
        self.sample_rate = config.audio.sample_rate
        self.audio_window_samples = int(audio_window_seconds * self.sample_rate)
        self.audio_hop_samples = int(audio_hop_seconds * self.sample_rate)
        
        # Components (loaded on demand)
        self._audio_encoder = None
        self._network_encoder = None
        self._gateway1 = None
        self._classical_branch = None
        self._quantum_branch = None
        self._gateway2 = None
        self._calibrator = None
        self._decision_gate = None
        self._audio_preprocessor = None
        self._telemetry_buffer = None
        self._telemetry_encoder = None
        
        # Rolling buffers
        max_audio_samples = int(self.sample_rate * audio_window_seconds * 2)  # 2x for safety
        overlap_samples = int(self.sample_rate * (audio_window_seconds - audio_hop_seconds))
        self._audio_buffer = RollingBuffer(
            max_samples=max_audio_samples,
            overlap_samples=overlap_samples
        )
        self._audio_frame_count = 0
        
        # Inference queue
        self._inference_queue: queue.Queue = queue.Queue(maxsize=max_queue_size * 10)
        self._result_queue: queue.Queue = queue.Queue(maxsize=max_queue_size * 10)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Latency tracking
        self.latency_tracker = LatencyTracker(self.targets)
        self._frame_counter = 0
        
        # Callbacks
        self.on_result: Optional[Callable[[InferenceResult], None]] = None
        self.on_latency_warning: Optional[Callable[[str, float, float], None]] = None
    
    def _load_components(self):
        """Lazy-load all model components."""
        if self._audio_encoder is not None:
            return
        
        # Import here to avoid circular imports
        from rafnet.audio import AudioEncoder, AudioEncoderConfig
        from rafnet.network import NetworkEncoder, NetworkEncoderConfig
        from rafnet.fusion import Gateway1, Gateway1Config, Gateway2, Gateway2Config
        from rafnet.classical import ClassicalBranch, ClassicalConfig
        from rafnet.quantum import QuantumBranch, QuantumConfig
        from rafnet.calibration import Calibrator, CalibrationConfig
        from rafnet.decision import DecisionGate, DecisionGateConfig
        from rafnet.data.dataset import NetworkTelemetryEncoder
        
        logger.info("Loading inference components...")
        
        # Audio encoder
        self._audio_preprocessor = AudioPreprocessor(self.sample_rate)
        self._audio_encoder = AudioEncoder(AudioEncoderConfig(
            sample_rate=self.config.audio.sample_rate,
            wavlm_model=self.config.audio.wavlm_model,
            wavlm_freeze=True,
            wavlm_output_dim=self.config.audio.wavlm_output_dim,
            projection_dim=self.config.audio.projection_dim,
            wavlm_pooling=self.config.audio.wavlm_pooling,
            wavlm_checkpointing=False,
            wavlm_mixed_precision=False,
            use_handcrafted=self.config.audio.use_handcrafted,
            handcrafted_features=self.config.audio.handcrafted_features,
            handcrafted_projection_dim=self.config.audio.handcrafted_projection_dim,
            wavlm_stride=self.config.audio.wavlm_stride,
            device=self.device,
        ))
        
        # Network encoder
        self._network_encoder = NetworkEncoder(NetworkEncoderConfig(
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
        ))
        self._telemetry_encoder = NetworkTelemetryEncoder(
            network_dim=self.config.network.input_dim
        )
        
        # Gateway 1
        self._gateway1 = Gateway1(Gateway1Config(
            audio_dim=self._audio_encoder.get_output_dim(),
            network_dim=self._network_encoder.get_output_dim(),
            hidden_dim=self.config.gateway1.hidden_dim,
            gate_type=self.config.gateway1.gate_type,
            reliability_aware=self.config.gateway1.reliability_aware,
            num_layers=self.config.gateway1.num_layers,
            use_residual_fusion=self.config.gateway1.use_residual_fusion,
            residual_weight=self.config.gateway1.residual_weight,
            gate_bias_init=self.config.gateway1.gate_bias_init,
            log_gate_stats=False,
            ablation_mode=self.config.gateway1.ablation_mode,
        ))
        
        # Classical branch
        self._classical_branch = ClassicalBranch(
            ClassicalConfig(
                model_type=self.config.classical.model_type,
                xgboost_params=self.config.classical.xgboost_params,
                use_calibration=self.config.classical.use_calibration,
                calibration_method=self.config.classical.calibration_method,
                random_state=self.config.classical.random_state,
                n_jobs=self.config.classical.n_jobs,
                measure_latency=True,
                calibration_cv=self.config.classical.calibration_cv,
                eval_metrics=self.config.classical.eval_metrics,
                serialization_format=self.config.classical.serialization_format,
            ),
            self.config.audio.projection_dim
        )
        
        # Quantum branch
        self._quantum_branch = QuantumBranch(
            QuantumConfig(
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
                measure_latency=True,
                checkpointing=self.config.quantum.checkpointing,
                fallback_mode=self.config.quantum.fallback_mode,
                serialization_format=self.config.quantum.serialization_format,
            ),
            self.config.audio.projection_dim
        )
        
        # Gateway 2
        self._gateway2 = Gateway2(Gateway2Config(
            input_dim=3,
            hidden_dim=self.config.gateway2.hidden_dim,
            gate_type=self.config.gateway2.gate_type,
            dropout=self.config.gateway2.dropout,
            num_layers=self.config.gateway2.num_layers,
            gate_bias_init=self.config.gateway2.gate_bias_init,
            log_gate_stats=False,
            enable_classical=self.config.gateway2.enable_classical,
            enable_quantum=self.config.gateway2.enable_quantum,
            ablation_mode=self.config.gateway2.ablation_mode,
        ))
        
        # Calibrator
        self._calibrator = Calibrator(CalibrationConfig(
            method=self.config.calibration.method,
            temperature_init=self.config.calibration.temperature_init,
            max_iter=self.config.calibration.max_iter,
            lr=self.config.calibration.lr,
            fit_on_validation_only=self.config.calibration.fit_on_validation_only,
            compute_ece=self.config.calibration.compute_ece,
            n_bins_ece=self.config.calibration.n_bins_ece,
            compute_brier=self.config.calibration.compute_brier,
        ))
        
        # Decision gate
        self._decision_gate = DecisionGate(DecisionGateConfig(
            tau_low=self.config.decision_gate.tau_low,
            tau_high=self.config.decision_gate.tau_high,
            optimize_thresholds=self.config.decision_gate.optimize_thresholds,
            metric_for_optimization=self.config.decision_gate.metric_for_optimization,
            grid_resolution=self.config.decision_gate.grid_resolution,
            select_on_validation=self.config.decision_gate.select_on_validation,
            uncertainty_action=self.config.decision_gate.uncertainty_action,
            min_uncertainty_width=self.config.decision_gate.min_uncertainty_width,
            max_uncertainty_width=self.config.decision_gate.max_uncertainty_width,
        ))
        
        # Move to device
        for component in [
            self._audio_encoder, self._network_encoder, self._gateway1,
            self._gateway2, self._calibrator, self._decision_gate
        ]:
            if hasattr(component, 'to'):
                component.to(self.device)
        
        # Set to eval mode
        for component in [
            self._audio_encoder, self._network_encoder, self._gateway1,
            self._gateway2, self._calibrator, self._decision_gate
        ]:
            if hasattr(component, 'eval'):
                component.eval()
        
        # Telemetry buffer
        self._telemetry_buffer = NetworkTelemetryBuffer(
            window_seconds=self.network_window_seconds,
            telemetry_fps=50
        )
        
        logger.info("All components loaded")
    
    def start(self) -> None:
        """Start the inference worker thread."""
        if self._running:
            return
        
        self._load_components()
        self._running = True
        self._worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._worker_thread.start()
        logger.info("Inference engine started")
    
    def stop(self) -> None:
        """Stop the inference worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info("Inference engine stopped")
    
    def push_audio(self, audio_frame: np.ndarray, input_sample_rate: int = None) -> bool:
        """
        Push audio frame to the engine.
        
        Returns True if frame was accepted, False if queue full.
        """
        try:
            self._inference_queue.put_nowait(("audio", audio_frame, input_sample_rate))
            return True
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")
            return False
    
    def push_network(self, telemetry: Dict) -> bool:
        """
        Push network telemetry to the engine.
        
        Returns True if accepted, False if queue full.
        """
        try:
            self._inference_queue.put_nowait(("network", telemetry))
            return True
        except queue.Full:
            logger.warning("Network queue full, dropping telemetry")
            return False
    
    def get_result(self, timeout: float = 0.0) -> Optional[InferenceResult]:
        """
        Get next inference result.
        
        Args:
            timeout: Seconds to wait (0 = non-blocking)
        
        Returns:
            InferenceResult or None if timeout
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest_result(self) -> Optional[InferenceResult]:
        """Get most recent result (non-blocking)."""
        result = None
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break
        return result
    
    def _inference_worker(self):
        """Background worker that processes inference queue.
        
        Implements frame dropping for real-time processing:
        - When queue builds up, only process the latest frame
        - Drop older frames to maintain real-time performance
        """
        while self._running:
            try:
                # Get next item from queue
                item_type, data, *args = self._inference_queue.get(timeout=0.1)
                
                # Frame dropping: if queue has built up, discard older frames
                # Keep only the most recent audio frame
                if item_type == "audio":
                    # Drain older audio frames, keep only latest
                    while not self._inference_queue.empty():
                        try:
                            next_item = self._inference_queue.get_nowait()
                            if next_item[0] == "audio":
                                item_type, data, *args = next_item
                            else:
                                # Put network items back
                                self._inference_queue.put(next_item)
                                break
                        except queue.Empty:
                            break
                    
                    self._process_audio_frame(data, *args)
                elif item_type == "network":
                    self._process_network_telemetry(data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Inference worker error: {e}")
    
    def _process_audio_frame(self, audio_frame: np.ndarray, input_sample_rate: int = None):
        """Process single audio frame."""
        self._frame_counter += 1
        
        # Preprocess
        with self.latency_tracker.measure("preprocessing"):
            processed_audio, preprocess_latency = self._audio_preprocessor.process(
                audio_frame, input_sample_rate
            )
        
        # Add to rolling buffer
        # processed_audio is [1, T], squeeze to [T]
        samples = processed_audio.squeeze(0).cpu().numpy()
        self._audio_buffer.append(samples)
        
        # Check if we have enough audio for a window
        if len(self._audio_buffer) >= self.audio_window_samples:
            # Get window with overlap
            window_samples = self._audio_buffer.get_window(self.audio_window_samples)
            audio_window = torch.from_numpy(window_samples).float().unsqueeze(0).to(self.device)  # [1, T]
            
            # Check if network telemetry is ready
            if self._telemetry_buffer.is_ready():
                self._run_inference(audio_window)
    
    def _process_network_telemetry(self, telemetry: Dict):
        """Process network telemetry sample."""
        was_ready = self._telemetry_buffer.is_ready()
        self._telemetry_buffer.push(telemetry)
        is_ready = self._telemetry_buffer.is_ready()
        
        # If telemetry just became ready and audio buffer is ready, trigger inference
        if not was_ready and is_ready and len(self._audio_buffer) >= self.audio_window_samples:
            # Get latest audio window and run inference
            window_samples = self._audio_buffer.get_window(self.audio_window_samples)
            audio_window = torch.from_numpy(window_samples).float().unsqueeze(0).to(self.device)
            self._run_inference(audio_window)
    
    def _run_inference(self, audio_window: torch.Tensor):
        """Run full inference pipeline on current windows."""
        frame_id = self._frame_counter
        start_total = time.perf_counter()
        stage_latencies = []
        
        try:
            # Get network window
            net_window = self._telemetry_buffer.get_window()
            
            # Encode network telemetry
            with self.latency_tracker.measure("network_encoder"):
                net_features_list = []
                net_mask_list = []
                for sample in net_window:
                    # Convert dict to SampleManifest-like object
                    from rafnet.data.manifest import SampleManifest
                    manifest = SampleManifest(
                        sample_id=f"stream_{time.time()}",
                        speaker_id="unknown",
                        generator_id="unknown",
                        class_label=0,
                        audio_path="",
                        communication_condition_id="stream",
                        codec=sample.get("codec", "opus"),
                        packet_loss=sample.get("packet_loss", 0.0),
                        jitter=sample.get("jitter", 0.0),
                        rtt=sample.get("rtt", 0.0),
                        bitrate=sample.get("bitrate", 64000.0),
                        packet_timing=sample.get("packet_timing"),
                        packet_size=sample.get("packet_size"),
                        burst_loss_count=sample.get("burst_loss_count"),
                        burst_loss_duration=sample.get("burst_loss_duration"),
                        loss_burstiness=sample.get("loss_burstiness"),
                    )
                    features, mask = self._telemetry_encoder.extract_features(manifest)
                    net_features_list.append(features)
                    net_mask_list.append(mask)
                
                net_features = torch.from_numpy(np.stack(net_features_list)).float().to(self.device)
                net_mask = torch.from_numpy(np.stack(net_mask_list)).float().to(self.device)
                
                # Temporal averaging
                net_features = net_features.mean(dim=0, keepdim=True)  # [1, D]
                net_mask = net_mask.mean(dim=0, keepdim=True)
                net_features = net_features.unsqueeze(1)  # [1, 1, D]
                net_mask = net_mask.unsqueeze(1)
            
            # Audio encoder
            with self.latency_tracker.measure("audio_encoder"):
                audio_features = self._audio_encoder(audio_window)
                audio_features = audio_features.unsqueeze(1)  # [1, 1, D] for temporal processor
            
            # Network encoder
            with self.latency_tracker.measure("network_encoder"):
                net_encoded = self._network_encoder(net_features, net_mask)
            
            # Gateway 1
            with self.latency_tracker.measure("gateway1"):
                z_fused, gate1_info = self._gateway1(audio_features.squeeze(1), net_encoded)
            
            # Classical branch
            with self.latency_tracker.measure("classical"):
                if self._classical_branch.is_fitted:
                    p_c = self._classical_branch(z_fused)
                else:
                    p_c = torch.full((z_fused.shape[0],), 0.5, device=z_fused.device)
            
            # Quantum branch
            with self.latency_tracker.measure("quantum"):
                if self._quantum_branch.pca_fitted:
                    p_q = self._quantum_branch(z_fused)
                else:
                    p_q = torch.full((z_fused.shape[0],), 0.5, device=z_fused.device)
            
            # Gateway 2
            with self.latency_tracker.measure("gateway2"):
                p_fused, gate2_info = self._gateway2(p_c, p_q)
            
            # Calibration
            with self.latency_tracker.measure("calibration"):
                if self._calibrator.fitted:
                    p_calibrated = self._calibrator(p_fused)
                else:
                    p_calibrated = p_fused
            
            # Decision gate
            with self.latency_tracker.measure("decision_gate"):
                decisions, decision_info = self._decision_gate(p_calibrated)
            
            # Build result
            total_latency = (time.perf_counter() - start_total) * 1000
            
            # Collect stage latencies
            for stage_name in [
                "preprocessing", "audio_encoder", "network_encoder",
                "gateway1", "classical", "quantum", "gateway2",
                "calibration", "decision_gate", "total"
            ]:
                stats = self.latency_tracker.get_stats(stage_name)
                if stats:
                    target = getattr(self.targets, stage_name, 100.0)
                    stage_latencies.append(StageLatency(
                        stage_name=stage_name,
                        latency_ms=stats["mean_ms"],
                        target_ms=target,
                        within_target=stats["mean_ms"] <= target,
                        timestamp=time.time(),
                    ))
            
            decision = decisions.squeeze().item()
            prob_synthetic = p_calibrated.squeeze().item()
            
            result = InferenceResult(
                decision=decision,
                decision_label={0: "REAL", 1: "UNCERTAIN", 2: "SYNTHETIC"}[decision],
                probability_real=1.0 - prob_synthetic,
                probability_synthetic=prob_synthetic,
                confidence=max(1.0 - prob_synthetic, prob_synthetic),
                audio_reliability=gate1_info.get("reliability_audio", None),
                network_reliability=gate1_info.get("reliability_network", None),
                gateway1_audio_weight=gate1_info.get("alpha", None),
                gateway1_network_weight=1.0 - gate1_info.get("alpha", None) if gate1_info.get("alpha") is not None else None,
                classical_probability=p_c.squeeze().item() if p_c is not None else None,
                quantum_probability=p_q.squeeze().item() if p_q is not None else None,
                gateway2_classical_weight=gate2_info.get("beta", None),
                gateway2_quantum_weight=1.0 - gate2_info.get("beta", None) if gate2_info.get("beta") is not None else None,
                fused_probability=p_fused.squeeze().item(),
                calibrated_probability=prob_synthetic,
                stage_latencies=stage_latencies,
                total_latency_ms=total_latency,
                within_total_target=total_latency <= self.targets.total,
                timestamp=time.time(),
                frame_id=frame_id,
            )
            
            # Put result in queue
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                logger.warning("Result queue full, dropping result")
            
            # Callbacks
            if self.on_result:
                self.on_result(result)
            
            # Check latency warnings
            if total_latency > self.targets.total:
                if self.on_latency_warning:
                    self.on_latency_warning("total", total_latency, self.targets.total)
        
        except Exception as e:
            logger.error(f"Inference error: {e}")

    def get_latency_stats(self) -> Dict:
        """Get comprehensive latency statistics."""
        stats = self.latency_tracker.get_all_stats()
        targets_met = self.latency_tracker.check_targets()
        
        return {
            "stage_stats": stats,
            "targets_met": targets_met,
            "overall_within_target": all(targets_met.values()) if targets_met else False,
            "frame_count": self._frame_counter,
        }
    
    def is_healthy(self) -> bool:
        """Check if engine is running and healthy."""
        return self._running and self._worker_thread and self._worker_thread.is_alive()


class SimulatedStream:
    """Simulated WebRTC/RTP stream for testing."""
    
    def __init__(
        self,
        engine: InferenceEngine,
        duration_seconds: float = 10.0,
        sample_rate: int = 16000,
        frame_size_ms: int = 20,
        packet_loss_rate: float = 0.0,
        network_jitter_ms: float = 1.0,
    ):
        self.engine = engine
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.frame_size_samples = int(sample_rate * frame_size_ms / 1000)
        self.packet_loss_rate = packet_loss_rate
        self.network_jitter_ms = network_jitter_ms
        
        self._running = False
        self._audio_thread: Optional[threading.Thread] = None
        self._network_thread: Optional[threading.Thread] = None
        self._results: List[InferenceResult] = []
        self._results_lock = threading.Lock()
        
        # Set up result callback
        self.engine.on_result = self._on_result
    
    def _on_result(self, result: InferenceResult):
        with self._results_lock:
            self._results.append(result)
    
    def start(self):
        """Start simulated streams."""
        self.engine.start()
        self._running = True
        
        self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._network_thread = threading.Thread(target=self._network_loop, daemon=True)
        
        self._audio_thread.start()
        self._network_thread.start()
    
    def stop(self):
        """Stop simulated streams."""
        self._running = False
        if self._audio_thread:
            self._audio_thread.join(timeout=2.0)
        if self._network_thread:
            self._network_thread.join(timeout=2.0)
        self.engine.stop()
    
    def _audio_loop(self):
        """Generate and push audio frames."""
        frame_interval = self.frame_size_ms / 1000.0
        next_frame_time = time.time()
        
        # Generate synthetic audio
        t = 0
        while self._running:
            frame_start = time.perf_counter()
            
            # Simulate packet loss
            if np.random.rand() > self.packet_loss_rate:
                # Generate synthetic audio (speech-like)
                frame = np.random.randn(self.frame_size_samples).astype(np.float32) * 0.1
                
                # Add some structure (formants)
                for f in [500, 1500, 2500]:
                    frame += 0.3 * np.sin(2 * np.pi * f * np.arange(self.frame_size_samples) / self.sample_rate + t)
                
                t += self.frame_size_samples / self.sample_rate
                
                self.engine.push_audio(frame, self.sample_rate)
            
            # Schedule next frame
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Measure actual interval
            actual_interval = (time.perf_counter() - frame_start) * 1000
    
    def _network_loop(self):
        """Generate and push network telemetry."""
        interval = 1.0 / 50.0  # 50 FPS
        next_update = time.time()
        
        base_packet_loss = 0.02
        base_jitter = 0.005
        base_rtt = 0.03
        base_bitrate = 64000
        
        while self._running:
            # Add some variation
            packet_loss = max(0, base_packet_loss + np.random.randn() * 0.01)
            jitter = max(0, base_jitter + np.random.randn() * 0.002)
            rtt = max(0.005, base_rtt + np.random.randn() * 0.005)
            bitrate = max(8000, base_bitrate + np.random.randn() * 5000)
            
            telemetry = {
                "packet_loss": packet_loss,
                "jitter": jitter,
                "rtt": rtt,
                "bitrate": bitrate,
                "codec": "opus",
                "packet_timing": np.cumsum(np.random.exponential(0.02, 50)).tolist(),
                "packet_size": np.random.normal(1200, 200, 50).clip(64, 1500).tolist(),
            }
            
            self.engine.push_network(telemetry)
            
            next_update += interval
            sleep_time = next_update - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_results(self) -> List[InferenceResult]:
        """Get all collected results."""
        with self._results_lock:
            return list(self._results)
    
    def get_latency_report(self) -> Dict:
        """Get latency report from engine."""
        return self.engine.get_latency_stats()


def run_latency_benchmark(
    config: "RAFNetConfig",
    duration_seconds: float = 30.0,
    device: str = "cpu",
    targets: LatencyTarget = None,
) -> Dict:
    """
    Run full latency benchmark with simulated stream.
    
    Returns comprehensive latency report.
    """
    engine = InferenceEngine(config, device=device, latency_target=targets)
    stream = SimulatedStream(engine, duration_seconds=duration_seconds)
    
    stream.start()
    
    # Wait for benchmark duration
    time.sleep(duration_seconds)
    
    stream.stop()
    
    # Collect results
    results = stream.get_results()
    latency_report = stream.get_latency_report()
    
    # Compute summary
    if results:
        latencies = [r.total_latency_ms for r in results]
        decisions = [r.decision for r in results]
        
        latency_report = {
            "benchmark_duration_sec": duration_seconds,
            "num_inferences": len(results),
            "latency_stats": {
                "mean_ms": float(np.mean(latencies)),
                "std_ms": float(np.std(latencies)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
            },
            "target_ms": targets.total if targets else 76.0,
            "target_met_pct": float(np.mean([r.within_total_target for r in results]) * 100),
            "decision_distribution": {
                "REAL": decisions.count(0),
                "UNCERTAIN": decisions.count(1),
                "SYNTHETIC": decisions.count(2),
            },
            "stage_stats": latency_report.get("stage_stats", {}),
            "targets_met": latency_report.get("targets_met", {}),
        }
    else:
        latency_report = {"error": "No results collected"}
    
    return latency_report


# Convenience function
def create_inference_engine(
    config_path: Optional[str] = None,
    device: str = "cpu",
    **kwargs
) -> InferenceEngine:
    """Create inference engine with optional config file."""
    from rafnet.config.schema import RAFNetConfig, load_config
    
    config = load_config(config_path) if config_path else RAFNetConfig()
    return InferenceEngine(config, device=device, **kwargs)


if __name__ == "__main__":
    # Quick test
    from rafnet.config.schema import RAFNetConfig
    
    config = RAFNetConfig()
    config.device = "cpu"
    config.classical.model_type = "logistic"  # Faster for testing
    config.quantum.fallback_mode = "classical"
    
    targets = LatencyTarget(
        preprocessing=5.0,
        audio_encoder=30.0,
        network_encoder=5.0,
        gateway1=5.0,
        classical=10.0,
        quantum=15.0,
        gateway2=3.0,
        calibration=2.0,
        decision_gate=1.0,
        total=76.0,
    )
    
    print("Running latency benchmark...")
    report = run_latency_benchmark(config, duration_seconds=10.0, device="cpu", targets=targets)
    
    print("\n=== LATENCY BENCHMARK REPORT ===")
    print(f"Duration: {report['benchmark_duration_sec']}s")
    print(f"Inferences: {report['num_inferences']}")
    print(f"\nLatency (ms):")
    for k, v in report['latency_stats'].items():
        print(f"  {k}: {v:.2f}")
    print(f"\nTarget: {report['target_ms']}ms")
    print(f"Target met: {report['target_met_pct']:.1f}%")
    print(f"\nDecisions: {report['decision_distribution']}")
    
    print("\nPer-stage targets:")
    for stage, met in report['targets_met'].items():
        status = "✓" if met else "✗"
        print(f"  {stage}: {status}")