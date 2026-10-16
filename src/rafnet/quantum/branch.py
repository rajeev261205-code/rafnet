import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pickle
import os
import time
from pathlib import Path
from contextlib import contextmanager

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None
    pnp = None

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)


def _pca_snapshot(pca) -> Optional[Dict[str, Any]]:
    """Snapshot everything sklearn needs to run transform after restore.

    Older saves persisted only components_/mean_, but sklearn >= 1.6 touches
    explained_variance_ during transform, breaking save/load round-trips.
    """
    if not bool(getattr(pca, "n_components_", 0) is not None and hasattr(pca, "components_")):
        return None
    try:
        return {
            "components": np.asarray(pca.components_),
            "mean": np.asarray(pca.mean_),
            "n_components": int(getattr(pca, "n_components_", np.shape(pca.components_)[0])),
            "n_features_in": int(getattr(pca, "n_features_in_", np.shape(pca.components_)[1])),
            "n_samples": int(getattr(pca, "n_samples_", np.shape(pca.components_)[1])),
            "singular_values": np.asarray(getattr(pca, "singular_values_", np.zeros(np.shape(pca.components_)[0]))),
            "explained_variance": np.asarray(getattr(pca, "explained_variance_", np.zeros(np.shape(pca.components_)[0]))),
            "explained_variance_ratio": np.asarray(getattr(pca, "explained_variance_ratio_", np.zeros(np.shape(pca.components_)[0]))),
            "noise_variance": float(getattr(pca, "noise_variance_", 0.0)),
        }
    except Exception:
        return None


def _pca_restore(pca, snap: Optional[Dict[str, Any]], fallback_n_samples: int = 250) -> None:
    """Restore a fitted PCA from a snapshot (or derive bookkeeping from components).

    `snap=None` handles pre-fix checkpoints that stored only components/mean:
    transform needs components_ + mean_ (present); the remaining attributes are
    algebraic bookkeeping re-derived so check_is_fitted/transform run.
    """
    if snap is not None:
        pca.components_ = np.asarray(snap["components"])
        pca.mean_ = np.asarray(snap["mean"])
        pca.n_components_ = int(snap.get("n_components", np.shape(pca.components_)[0]))
        pca.n_features_in_ = int(snap.get("n_features_in", np.shape(pca.components_)[1]))
        pca.n_samples_ = int(snap.get("n_samples", fallback_n_samples))
        n_comp = int(np.shape(pca.components_)[0])
        pca.singular_values_ = np.asarray(snap.get("singular_values", np.zeros(n_comp)), dtype=np.float64)
        pca.explained_variance_ = np.asarray(snap.get("explained_variance", np.zeros(n_comp)), dtype=np.float64)
        pca.explained_variance_ratio_ = np.asarray(snap.get("explained_variance_ratio", np.zeros(n_comp)), dtype=np.float64)
        pca.noise_variance_ = float(snap.get("noise_variance", 0.0))
        return
    # Legacy path: components_/mean_ were set directly by the caller.
    comp = getattr(pca, "components_", None)
    if comp is None:
        return
    n_comp, n_feat = int(np.shape(comp)[0]), int(np.shape(comp)[1])
    if not hasattr(pca, "n_features_in_"):
        pca.n_features_in_ = n_feat
    if not hasattr(pca, "n_samples_"):
        pca.n_samples_ = fallback_n_samples
    if not hasattr(pca, "singular_values_"):
        pca.singular_values_ = np.zeros(n_comp, dtype=np.float64)
    if not hasattr(pca, "explained_variance_"):
        pca.explained_variance_ = (pca.singular_values_ ** 2) / max(1, pca.n_samples_ - 1)
    if not hasattr(pca, "explained_variance_ratio_"):
        total = float(pca.explained_variance_.sum())
        pca.explained_variance_ratio_ = np.zeros(n_comp, dtype=np.float64) if total == 0 else pca.explained_variance_ / total
    if not hasattr(pca, "noise_variance_"):
        pca.noise_variance_ = 0.0


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
    entanglement: Literal["linear", "circular", "full"] = "circular"
    rotation_gates: Literal["RX_RY_RZ", "RY_RZ", "RX_RY"] = "RX_RY_RZ"
    measure_latency: bool = True
    checkpointing: bool = False
    fallback_mode: Literal["classical", "none"] = "classical"
    serialization_format: Literal["pickle", "joblib"] = "pickle"


class LatencyTracker:
    """Track latency statistics for quantum inference."""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.batch_sizes: List[int] = []
        self.pca_latencies: List[float] = []
        self.circuit_latencies: List[float] = []
        self.classical_head_latencies: List[float] = []
    
    def record(self, total_ms: float, batch_size: int, pca_ms: float = 0, circuit_ms: float = 0, head_ms: float = 0) -> None:
        self.latencies.append(total_ms)
        self.batch_sizes.append(batch_size)
        self.pca_latencies.append(pca_ms)
        self.circuit_latencies.append(circuit_ms)
        self.classical_head_latencies.append(head_ms)
    
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
            "pca_mean_ms": float(np.array(self.pca_latencies).mean()) if self.pca_latencies else 0,
            "circuit_mean_ms": float(np.array(self.circuit_latencies).mean()) if self.circuit_latencies else 0,
            "head_mean_ms": float(np.array(self.classical_head_latencies).mean()) if self.classical_head_latencies else 0,
        }
    
    def reset(self) -> None:
        self.latencies.clear()
        self.batch_sizes.clear()
        self.pca_latencies.clear()
        self.circuit_latencies.clear()
        self.classical_head_latencies.clear()


class QuantumCircuitBase(ABC):
    """Abstract base for quantum circuits."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_num_parameters(self) -> int:
        pass


class ClassicalFallbackCircuit(nn.Module, QuantumCircuitBase):
    """
    Classical fallback circuit for unit testing without PennyLane.
    
    Mimics the quantum circuit interface but uses classical MLP.
    Does NOT claim quantum advantage - purely for testing infrastructure.
    """
    
    def __init__(self, config: QuantumConfig):
        super().__init__()
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        
        # Classical MLP mimicking quantum parameter count
        # Each layer has n_qubits * 3 rotation parameters
        param_count = config.n_layers * config.n_qubits * 3
        
        self.fallback_net = nn.Sequential(
            nn.Linear(config.pca_dim, config.n_qubits * 2),
            nn.LayerNorm(config.n_qubits * 2),
            nn.GELU(),
            nn.Linear(config.n_qubits * 2, config.n_qubits),
        )
        
        self.pca = PCA(n_components=config.pca_dim, random_state=config.random_state)
        self.pca_fitted = False
        self.latency_tracker = LatencyTracker()
    
    def fit_pca(self, X: torch.Tensor) -> None:
        X_np = X.detach().cpu().numpy()
        self.pca.fit(X_np)
        self.pca_fitted = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pca_fitted:
            raise RuntimeError("PCA must be fitted before forward pass")
        
        start_total = time.perf_counter()
        
        # PCA timing
        start_pca = time.perf_counter()
        x_pca = self.pca.transform(x.detach().cpu().numpy())
        x_pca = torch.from_numpy(x_pca).float().to(x.device)
        pca_ms = (time.perf_counter() - start_pca) * 1000
        
        # Classical head timing
        start_head = time.perf_counter()
        batch_size = x_pca.shape[0]
        outputs = self.fallback_net(x_pca)
        head_ms = (time.perf_counter() - start_head) * 1000
        
        total_ms = (time.perf_counter() - start_total) * 1000
        self.latency_tracker.record(total_ms, batch_size, pca_ms, 0, head_ms)
        
        return outputs
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.fallback_net.parameters())
    
    def get_latency_stats(self) -> Dict[str, float]:
        return self.latency_tracker.get_stats()
    
    def reset_latency_stats(self) -> None:
        self.latency_tracker.reset()


class PennyLaneCircuit(nn.Module, QuantumCircuitBase):
    """PennyLane variational quantum circuit."""
    
    def __init__(self, config: QuantumConfig):
        super().__init__()
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane not available. Install with: pip install pennylane pennylane-lightning")
        
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        
        self.dev = qml.device(config.device, wires=config.n_qubits, shots=config.shots)
        
        # Variational parameters: [n_layers, n_qubits, 3] for RX, RY, RZ
        self.weights = nn.Parameter(
            torch.randn(config.n_layers, config.n_qubits, 3) * 0.1
        )
        
        self._build_circuit()
        
        self.pca = PCA(n_components=config.pca_dim, random_state=config.random_state)
        self.pca_fitted = False
        self.latency_tracker = LatencyTracker()
        
        self._checkpointed_circuit = None
    
    def _build_circuit(self):
        # Validate amplitude encoding requirement
        if self.config.encoding == "amplitude":
            required_dim = 2 ** self.n_qubits
            if self.config.pca_dim != required_dim:
                raise ValueError(
                    f"Amplitude encoding requires pca_dim={required_dim} "
                    f"(2^n_qubits), got {self.config.pca_dim}"
                )
        
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Angle encoding
            if self.config.encoding == "angle":
                for i in range(self.n_qubits):
                    qml.RY(inputs[i], wires=i)
            elif self.config.encoding == "amplitude":
                qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)
            
            # Variational layers
            for layer in range(self.n_layers):
                # Single-qubit rotations
                for i in range(self.n_qubits):
                    if self.config.rotation_gates in ["RX_RY_RZ", "RX_RY"]:
                        qml.RX(weights[layer, i, 0], wires=i)
                    if self.config.rotation_gates in ["RX_RY_RZ", "RY_RZ"]:
                        qml.RY(weights[layer, i, 1], wires=i)
                    if self.config.rotation_gates == "RX_RY_RZ":
                        qml.RZ(weights[layer, i, 2], wires=i)
                
                # Entanglement
                if self.config.entanglement == "linear":
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                elif self.config.entanglement == "circular":
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
                elif self.config.entanglement == "full":
                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            qml.CNOT(wires=[i, j])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.circuit = circuit
        
        # Build checkpointed version if enabled
        if self.config.checkpointing:
            self._build_checkpointed_circuit()
    
    def _build_checkpointed_circuit(self):
        """Build gradient-checkpointed version for memory efficiency."""
        # For PennyLane, we use the native diff_method="backprop" 
        # which already supports gradient checkpointing via PyTorch
        # This is a placeholder for future advanced checkpointing
        pass
    
    def fit_pca(self, X: torch.Tensor) -> None:
        X_np = X.detach().cpu().numpy()
        self.pca.fit(X_np)
        self.pca_fitted = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pca_fitted:
            raise RuntimeError("PCA must be fitted before forward pass")
        
        start_total = time.perf_counter()
        
        # PCA timing
        start_pca = time.perf_counter()
        x_pca = self.pca.transform(x.detach().cpu().numpy())
        x_pca = torch.from_numpy(x_pca).float().to(x.device)
        pca_ms = (time.perf_counter() - start_pca) * 1000
        
        # Circuit timing
        start_circuit = time.perf_counter()
        batch_size = x_pca.shape[0]
        outputs = []
        
        for i in range(batch_size):
            out = self.circuit(x_pca[i], self.weights)
            outputs.append(torch.stack(out))
        
        quantum_features = torch.stack(outputs)
        circuit_ms = (time.perf_counter() - start_circuit) * 1000
        
        total_ms = (time.perf_counter() - start_total) * 1000
        self.latency_tracker.record(total_ms, batch_size, pca_ms, circuit_ms, 0)
        
        return quantum_features
    
    def get_num_parameters(self) -> int:
        return self.weights.numel()
    
    def get_latency_stats(self) -> Dict[str, float]:
        return self.latency_tracker.get_stats()
    
    def reset_latency_stats(self) -> None:
        self.latency_tracker.reset()
    
    def get_pca_components(self) -> Optional[np.ndarray]:
        """Get fitted PCA components for serialization."""
        if self.pca_fitted:
            return self.pca.components_
        return None
    
    def get_pca_mean(self) -> Optional[np.ndarray]:
        """Get fitted PCA mean for serialization."""
        if self.pca_fitted:
            return self.pca.mean_
        return None
    
    def load_pca(self, components: np.ndarray, mean: np.ndarray) -> None:
        """Load pre-fitted PCA components."""
        self.pca.components_ = components
        self.pca.mean_ = mean
        _pca_restore(self.pca, None)
        self.pca_fitted = True


class QuantumBranch(nn.Module):
    """
    Quantum/Hybrid branch for RAFNet.
    
    Pipeline: z_f → PCA → angle encoding → VQC → classical head → p_q
    
    Features:
    - Configurable qubits, layers, encoding, entanglement
    - PCA fitting on training data only
    - PCA serialization
    - Variational parameters with gradient support
    - Classical output head
    - Latency measurement
    - Checkpointing support
    - Classical fallback for testing
    """
    
    def __init__(self, config: QuantumConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Initialize quantum circuit (with fallback)
        if PENNYLANE_AVAILABLE and config.fallback_mode != "classical":
            try:
                self.quantum_circuit = PennyLaneCircuit(config)
                self._using_fallback = False
            except Exception as e:
                if config.fallback_mode == "classical":
                    self.quantum_circuit = ClassicalFallbackCircuit(config)
                    self._using_fallback = True
                else:
                    raise e
        else:
            self.quantum_circuit = ClassicalFallbackCircuit(config)
            self._using_fallback = True
        
        # Classical output head
        if config.use_classical_head:
            self.classical_head = nn.Sequential(
                nn.Linear(config.n_qubits, config.classical_head_hidden),
                nn.LayerNorm(config.classical_head_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.classical_head_hidden, 1),
                nn.Sigmoid()
            )
        else:
            self.classical_head = nn.Sequential(
                nn.Linear(config.n_qubits, 1),
                nn.Sigmoid()
            )
        
        self.pca_fitted = False
        self._is_calibrated = False
        self._calibrator = None
        self._training_history: List[Dict] = []
        self._eval_history: List[Dict] = []
    
    def fit_pca(self, X: torch.Tensor) -> "QuantumBranch":
        """Fit PCA on training data only."""
        self.quantum_circuit.fit_pca(X)
        self.pca_fitted = True
        
        self._training_history.append({
            "pca_fitted": True,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "pca_dim": self.config.pca_dim,
            "using_fallback": self._using_fallback,
        })
        return self
    
    def fit_quantum(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor,
        n_epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 32,
        validation_split: float = 0.2,
        calibrate: bool = True,
        calibration_method: str = "sigmoid",
        verbose: bool = True,
    ) -> "QuantumBranch":
        """
        Train quantum variational parameters and classical head.
        
        This makes the quantum branch comparable to classical branch by training
        the full pipeline: PCA → VQC → classical head.
        
        Args:
            X: Training embeddings from Gateway 1 [N, D]
            y: Training labels [N]
            n_epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            validation_split: Fraction for validation (used for early stopping and calibration)
            calibrate: Whether to calibrate output probabilities
            calibration_method: "sigmoid" or "isotonic" for calibration
            verbose: Print training progress
            
        Returns:
            Self for chaining
        """
        if not self.pca_fitted:
            raise RuntimeError("PCA must be fitted before training. Call fit_pca() first.")
        
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # Split for validation (if calibrate)
        if calibrate:
            n_val = int(len(X_np) * validation_split)
            indices = np.random.permutation(len(X_np))
            val_idx, train_idx = indices[:n_val], indices[n_val:]
            X_train, y_train = X_np[train_idx], y_np[train_idx]
            X_val, y_val = X_np[val_idx], y_np[val_idx]
        else:
            X_train, y_train = X_np, y_np
            X_val, y_val = None, None
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float()
        y_train_t = torch.from_numpy(y_train).float()
        
        # Setup optimizer for quantum weights + classical head
        params = list(self.quantum_circuit.parameters()) + list(self.classical_head.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        self.train()
        for epoch in range(n_epochs):
            # Shuffle training data
            perm = torch.randperm(len(X_train_t))
            X_train_t = X_train_t[perm]
            y_train_t = y_train_t[perm]
            
            epoch_losses = []
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                
                optimizer.zero_grad()
                probs = self.forward(batch_X)
                loss = criterion(probs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            
            # Validation check
            if calibrate and X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_probs = self.forward(torch.from_numpy(X_val).float())
                    val_loss = criterion(val_probs, torch.from_numpy(y_val).float()).item()
                self.train()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break
            
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch}: train_loss={avg_loss:.4f}" + 
                      (f", val_loss={val_loss:.4f}" if calibrate else ""))
        
        self.eval()
        
        # Calibrate on validation data if requested
        if calibrate and X_val is not None:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.base import clone
            
            # Get quantum features for validation set
            with torch.no_grad():
                val_features = self.get_quantum_features(torch.from_numpy(X_val).float()).numpy()
            
            # Train calibrator on classical head output
            self._calibrator = CalibratedClassifierCV(
                clone(self._get_sklearn_head()), 
                method=calibration_method, 
                cv=3
            )
            self._calibrator.fit(val_features, y_val)
            self._is_calibrated = True
            
            if verbose:
                print(f"  Calibration: {calibration_method} on {len(X_val)} validation samples")
        
        self._training_history.append({
            "quantum_trained": True,
            "n_epochs": n_epochs,
            "final_train_loss": avg_loss,
            "calibrated": calibrate,
            "calibration_method": calibration_method if calibrate else None,
        })
        
        return self
    
    def _get_sklearn_head(self):
        """Get sklearn-compatible wrapper for classical head."""
        from sklearn.base import BaseEstimator, ClassifierMixin
        import numpy as np
        
        class QuantumHeadWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, branch):
                self.branch = branch
                self.classes_ = np.array([0, 1])
            
            def predict_proba(self, X):
                self.branch.eval()
                with torch.no_grad():
                    probs = self.branch(torch.from_numpy(X).float()).numpy()
                return np.column_stack([1 - probs, probs])
            
            def predict(self, X):
                probs = self.predict_proba(X)[:, 1]
                return (probs > 0.5).astype(int)
        
        return QuantumHeadWrapper(self)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass: z_f → PCA → VQC → classical head → p_q"""
        if not self.pca_fitted:
            raise RuntimeError("PCA must be fitted before forward pass. Call fit_pca() first.")
        
        start_head = time.perf_counter()
        quantum_features = self.quantum_circuit(X)
        output = self.classical_head(quantum_features)
        head_ms = (time.perf_counter() - start_head) * 1000
        
        # Apply calibration if available
        if self._is_calibrated and self._calibrator is not None:
            # Get quantum features for calibration
            with torch.no_grad():
                features_np = quantum_features.detach().cpu().numpy()
                calibrated_proba = self._calibrator.predict_proba(features_np)[:, 1]
                output = torch.from_numpy(calibrated_proba).float().to(X.device).unsqueeze(-1)
        
        # Update latency tracker with classical head time
        if hasattr(self.quantum_circuit, 'latency_tracker'):
            # Get the last recorded entry and update head time
            if self.quantum_circuit.latency_tracker.classical_head_latencies:
                self.quantum_circuit.latency_tracker.classical_head_latencies[-1] = head_ms
                # Recompute total
                idx = len(self.quantum_circuit.latency_tracker.latencies) - 1
                self.quantum_circuit.latency_tracker.latencies[idx] += head_ms
        
        return output.squeeze(-1)
    
    def calibrate(self, X: torch.Tensor, y: torch.Tensor, method: str = "sigmoid", cv: int = 3) -> "QuantumBranch":
        """Calibrate quantum branch output probabilities using validation data."""
        if not self.pca_fitted:
            raise RuntimeError("PCA must be fitted before calibration")
        
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # Get quantum features
        with torch.no_grad():
            features = self.get_quantum_features(X).numpy()
        
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import clone
        
        self._calibrator = CalibratedClassifierCV(
            clone(self._get_sklearn_head()),
            method=method,
            cv=cv
        )
        self._calibrator.fit(features, y_np)
        self._is_calibrated = True
        
        return self
    
    def is_calibrated(self) -> bool:
        return self._is_calibrated
    
    def get_quantum_features(self, X: torch.Tensor) -> torch.Tensor:
        """Get raw quantum features (expectation values) without classical head."""
        if not self.pca_fitted:
            raise RuntimeError("PCA must be fitted before forward pass.")
        return self.quantum_circuit(X)
    
    def evaluate(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        if not self.pca_fitted:
            raise RuntimeError("PCA must be fitted before evaluation")
        
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "logloss"]
        
        with torch.no_grad():
            probs = self.forward(X).cpu().numpy()
            y_np = y.detach().cpu().numpy()
        
        preds = (probs > 0.5).astype(int)
        
        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = float(accuracy_score(y_np, preds))
            elif metric == "precision":
                results[metric] = float(precision_score(y_np, preds, zero_division=0))
            elif metric == "recall":
                results[metric] = float(recall_score(y_np, preds, zero_division=0))
            elif metric == "f1":
                results[metric] = float(f1_score(y_np, preds, zero_division=0))
            elif metric == "roc_auc":
                try:
                    results[metric] = float(roc_auc_score(y_np, probs))
                except ValueError:
                    results[metric] = float("nan")
            elif metric == "logloss":
                try:
                    results[metric] = float(log_loss(y_np, probs))
                except ValueError:
                    results[metric] = float("nan")
            elif metric == "eer":
                results[metric] = self._compute_eer(y_np, probs)
            else:
                results[metric] = float("nan")
        
        results["latency"] = self.get_latency_stats()
        results["num_parameters"] = self.get_num_parameters()
        results["using_fallback"] = self._using_fallback
        
        self._eval_history.append(results)
        return results
    
    def _compute_eer(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute Equal Error Rate."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        return float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    
    def get_latency_stats(self) -> Dict[str, float]:
        return self.quantum_circuit.get_latency_stats()
    
    def reset_latency_stats(self) -> None:
        self.quantum_circuit.reset_latency_stats()
    
    def get_num_parameters(self) -> int:
        """Total parameter count: quantum + classical head."""
        quantum_params = self.quantum_circuit.get_num_parameters()
        head_params = sum(p.numel() for p in self.classical_head.parameters())
        return quantum_params + head_params
    
    def get_quantum_parameters(self) -> int:
        return self.quantum_circuit.get_num_parameters()
    
    def get_classical_head_parameters(self) -> int:
        return sum(p.numel() for p in self.classical_head.parameters())
    
    def save(self, path: str) -> None:
        """Save the entire branch (quantum circuit + PCA + classical head)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        save_dict = {
            "config": self.config,
            "input_dim": self.input_dim,
            "pca_fitted": self.pca_fitted,
            "using_fallback": self._using_fallback,
            "classical_head_state": self.classical_head.state_dict(),
            "training_history": self._training_history,
            "eval_history": self._eval_history,
        }
        
        if not self._using_fallback:
            save_dict["quantum_weights"] = self.quantum_circuit.weights.detach().cpu()
            save_dict["pca_components"] = self.quantum_circuit.get_pca_components()
            save_dict["pca_mean"] = self.quantum_circuit.get_pca_mean()
            save_dict["pca_snapshot"] = _pca_snapshot(self.quantum_circuit.pca)
        else:
            save_dict["fallback_net_state"] = self.quantum_circuit.fallback_net.state_dict()
            save_dict["pca_components"] = self.quantum_circuit.pca.components_ if self.quantum_circuit.pca_fitted else None
            save_dict["pca_mean"] = self.quantum_circuit.pca.mean_ if self.quantum_circuit.pca_fitted else None
            save_dict["pca_snapshot"] = _pca_snapshot(self.quantum_circuit.pca) if self.quantum_circuit.pca_fitted else None
        
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, path: str) -> "QuantumBranch":
        """Load a saved branch."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        
        branch = cls(save_dict["config"], save_dict["input_dim"])
        branch.pca_fitted = save_dict["pca_fitted"]
        branch._using_fallback = save_dict["using_fallback"]
        branch._training_history = save_dict.get("training_history", [])
        branch._eval_history = save_dict.get("eval_history", [])
        
        branch.classical_head.load_state_dict(save_dict["classical_head_state"])
        
        if not branch._using_fallback:
            branch.quantum_circuit.weights.data = save_dict["quantum_weights"]
            if save_dict["pca_components"] is not None:
                branch.quantum_circuit.load_pca(
                    save_dict["pca_components"],
                    save_dict["pca_mean"]
                )
                if save_dict.get("pca_snapshot") is not None:
                    _pca_restore(branch.quantum_circuit.pca, save_dict["pca_snapshot"])
        else:
            branch.quantum_circuit.fallback_net.load_state_dict(save_dict["fallback_net_state"])
            if save_dict["pca_components"] is not None:
                branch.quantum_circuit.pca.components_ = save_dict["pca_components"]
                branch.quantum_circuit.pca.mean_ = save_dict["pca_mean"]
                # Need to set n_components_ and other attributes for PCA to work
                branch.quantum_circuit.pca.n_components_ = save_dict["pca_components"].shape[0]
                if save_dict.get("pca_snapshot") is not None:
                    _pca_restore(branch.quantum_circuit.pca, save_dict["pca_snapshot"])
                else:
                    # Pre-fix checkpoints stored only components/mean.
                    _pca_restore(branch.quantum_circuit.pca, None)
                branch.quantum_circuit.pca_fitted = True
                # Also set the fallback circuit's pca_fitted
                branch.quantum_circuit.pca_fitted = True
        
        # Set to eval mode for deterministic inference
        branch.eval()
        
        return branch
    
    def get_training_history(self) -> List[Dict]:
        return self._training_history
    
    def get_eval_history(self) -> List[Dict]:
        return self._eval_history
    
    def is_using_fallback(self) -> bool:
        return self._using_fallback


def run_quantum_smoke_experiment(
    n_samples: int = 200,
    n_features: int = 256,
    n_qubits: int = 8,
    n_layers: int = 3,
    test_split: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a smoke experiment for the quantum branch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Quantum Branch Smoke Experiment")
        print(f"{'='*60}")
        print(f"Samples: {n_samples}, Features: {n_features}")
        print(f"Qubits: {n_qubits}, Layers: {n_layers}")
        print(f"Test split: {test_split}")
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
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
    
    # Create config
    config = QuantumConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        pca_dim=min(n_qubits, 8),
        measure_latency=True,
        fallback_mode="classical",  # Use fallback for reliable testing
    )
    
    branch = QuantumBranch(config, input_dim=n_features)
    
    # Train PCA
    start = time.time()
    branch.fit_pca(torch.from_numpy(X_train).float())
    pca_time = time.time() - start
    
    if verbose:
        print(f"PCA fitting time: {pca_time:.3f}s")
        print(f"Using fallback: {branch.is_using_fallback()}")
    
    # Evaluate (no training of quantum params in this smoke test)
    results = branch.evaluate(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).long()
    )
    
    if verbose:
        print(f"\nTest Metrics:")
        for k, v in results.items():
            if k not in ["latency", "num_parameters", "using_fallback"]:
                print(f"  {k}: {v:.4f}")
        
        if "latency" in results and results["latency"]:
            lat = results["latency"]
            print(f"\nLatency Stats:")
            print(f"  Mean: {lat.get('mean_ms', 0):.2f} ms")
            print(f"  P95:  {lat.get('p95_ms', 0):.2f} ms")
            print(f"  Circuit: {lat.get('circuit_mean_ms', 0):.2f} ms")
            print(f"  Head: {lat.get('head_mean_ms', 0):.2f} ms")
        
        print(f"\nParameters: {results.get('num_parameters', 0)}")
    
    # Test serialization
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name
    
    try:
        branch.save(temp_path)
        loaded_branch = QuantumBranch.load(temp_path)
        
        # Both in eval mode for deterministic inference
        branch.eval()
        loaded_branch.eval()
        test_proba = loaded_branch(torch.from_numpy(X_test[:5]).float())
        original_proba = branch(torch.from_numpy(X_test[:5]).float())
        
        assert torch.allclose(test_proba, original_proba), "Serialization failed!"
        if verbose:
            print(f"\nSerialization: OK")
    finally:
        os.unlink(temp_path)
    
    # Test probability prediction
    test_X = torch.from_numpy(X_test[:10]).float()
    probas = branch(test_X)
    preds = (probas > 0.5).long()
    
    if verbose:
        print(f"\nPredictions shape: {probas.shape}")
        print(f"Probabilities (first 5): {probas[:5].tolist()}")
        print(f"Predictions (first 5): {preds[:5].tolist()}")
    
    return {
        "branch": branch,
        "config": config,
        "results": results,
        "pca_time": pca_time,
        "X_test": X_test,
        "y_test": y_test,
    }


def run_quantum_config_comparison(
    configs: List[Dict],
    n_samples: int = 200,
    n_features: int = 256,
    test_split: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run comparison across multiple quantum configurations."""
    results = {}
    
    for i, cfg in enumerate(configs):
        name = cfg.get("name", f"config_{i}")
        try:
            result = run_quantum_smoke_experiment(
                n_samples=n_samples,
                n_features=n_features,
                n_qubits=cfg.get("n_qubits", 8),
                n_layers=cfg.get("n_layers", 3),
                test_split=test_split,
                seed=seed,
                verbose=False,
            )
            results[name] = {"success": True, **result}
        except Exception as e:
            results[name] = {"success": False, "error": str(e)}
            if verbose:
                print(f"\n{name} FAILED: {e}")
    
    if verbose:
        print(f"\n{'='*60}")
        print("QUANTUM CONFIG COMPARISON")
        print(f"{'='*60}")
        for name, result in results.items():
            status = "PASS" if result["success"] else "FAIL"
            if result["success"]:
                acc = result["results"].get("accuracy", 0)
                params = result["results"].get("num_parameters", 0)
                print(f"  {name:20s}: {status} (acc={acc:.4f}, params={params})")
            else:
                print(f"  {name:20s}: {status} ({result['error']})")
    
    return results


if __name__ == "__main__":
    # Run smoke experiment
    run_quantum_smoke_experiment()