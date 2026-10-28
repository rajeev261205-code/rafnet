import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

import numpy as np


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestQuantumConfig:
    def test_default_config(self):
        from rafnet.quantum.branch import QuantumConfig
        config = QuantumConfig()
        
        assert config.n_qubits == 8
        assert config.n_layers == 3
        assert config.encoding == "angle"
        assert config.pca_dim == 8
        assert config.device == "default.qubit"
        assert config.shots is None
        assert config.use_classical_head is True
        assert config.classical_head_hidden == 32
        assert config.entanglement == "circular"
        assert config.rotation_gates == "RX_RY_RZ"
        assert config.measure_latency is True
        assert config.checkpointing is False
        assert config.fallback_mode == "classical"
    
    def test_custom_config(self):
        from rafnet.quantum.branch import QuantumConfig
        config = QuantumConfig(
            n_qubits=4,
            n_layers=2,
            encoding="amplitude",
            pca_dim=4,
            entanglement="linear",
            rotation_gates="RY_RZ",
            measure_latency=False,
            fallback_mode="none",
        )
        
        assert config.n_qubits == 4
        assert config.n_layers == 2
        assert config.encoding == "amplitude"
        assert config.pca_dim == 4
        assert config.entanglement == "linear"
        assert config.rotation_gates == "RY_RZ"
        assert config.measure_latency is False
        assert config.fallback_mode == "none"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestQuantumBranchFallback:
    def test_fallback_circuit_init(self):
        from rafnet.quantum.branch import ClassicalFallbackCircuit, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4)
        circuit = ClassicalFallbackCircuit(config)
        
        assert circuit.n_qubits == 4
        assert circuit.n_layers == 2
    
    def test_fallback_fit_pca(self):
        from rafnet.quantum.branch import ClassicalFallbackCircuit, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4)
        circuit = ClassicalFallbackCircuit(config)
        
        X = torch.randn(50, 32)
        circuit.fit_pca(X)
        
        assert circuit.pca_fitted
    
    def test_fallback_forward(self):
        from rafnet.quantum.branch import ClassicalFallbackCircuit, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4)
        circuit = ClassicalFallbackCircuit(config)
        
        X = torch.randn(50, 32)
        circuit.fit_pca(X)
        
        output = circuit(X[:10])
        
        assert output.shape == (10, 4)
        assert not torch.isnan(output).any()
    
    def test_fallback_latency_tracking(self):
        from rafnet.quantum.branch import ClassicalFallbackCircuit, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, measure_latency=True)
        circuit = ClassicalFallbackCircuit(config)
        
        X = torch.randn(50, 32)
        circuit.fit_pca(X)
        
        for _ in range(5):
            _ = circuit(X[:10])
        
        stats = circuit.get_latency_stats()
        
        assert "mean_ms" in stats
        assert "p95_ms" in stats
        assert stats["total_samples"] > 0
    
    def test_fallback_parameter_count(self):
        from rafnet.quantum.branch import ClassicalFallbackCircuit, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4)
        circuit = ClassicalFallbackCircuit(config)
        
        params = circuit.get_num_parameters()
        assert params > 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestQuantumBranchPennyLane:
    def test_pennylane_circuit_init(self):
        from rafnet.quantum.branch import PennyLaneCircuit, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4)
        circuit = PennyLaneCircuit(config)
        
        assert circuit.n_qubits == 4
        assert circuit.n_layers == 2
        assert circuit.weights.shape == (2, 4, 3)
    
    def test_pennylane_fit_pca(self):
        from rafnet.quantum.branch import PennyLaneCircuit, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4)
        circuit = PennyLaneCircuit(config)
        
        X = torch.randn(50, 32)
        circuit.fit_pca(X)
        
        assert circuit.pca_fitted
    
    def test_pennylane_forward(self):
        from rafnet.quantum.branch import PennyLaneCircuit, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4)
        circuit = PennyLaneCircuit(config)
        
        X = torch.randn(50, 32)
        circuit.fit_pca(X)
        
        output = circuit(X[:10])
        
        assert output.shape == (10, 4)
        assert not torch.isnan(output).any()
    
    def test_pennylane_parameter_count(self):
        from rafnet.quantum.branch import PennyLaneCircuit, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4)
        circuit = PennyLaneCircuit(config)
        
        params = circuit.get_num_parameters()
        assert params == 4 * 2 * 3  # n_layers * n_qubits * 3
    
    def test_pennylane_entanglement_modes(self):
        from rafnet.quantum.branch import PennyLaneCircuit, QuantumConfig
        
        for entanglement in ["linear", "circular", "full"]:
            config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, entanglement=entanglement)
            circuit = PennyLaneCircuit(config)
            
            X = torch.randn(10, 32)
            circuit.fit_pca(X)
            output = circuit(X[:5])
            
            assert output.shape == (5, 4)
    
    def test_pennylane_rotation_gates(self):
        from rafnet.quantum.branch import PennyLaneCircuit, QuantumConfig
        
        for rot_gates in ["RX_RY_RZ", "RY_RZ", "RX_RY"]:
            config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, rotation_gates=rot_gates)
            circuit = PennyLaneCircuit(config)
            
            X = torch.randn(10, 32)
            circuit.fit_pca(X)
            output = circuit(X[:5])
            
            assert output.shape == (5, 4)
    
    def test_pennylane_pca_serialization(self):
        from rafnet.quantum.branch import PennyLaneCircuit, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4)
        circuit = PennyLaneCircuit(config)
        
        X = torch.randn(50, 32)
        circuit.fit_pca(X)
        
        components = circuit.get_pca_components()
        mean = circuit.get_pca_mean()
        
        assert components is not None
        assert mean is not None
        assert components.shape == (4, 32)
        assert mean.shape == (32,)
        
        # Test loading
        new_circuit = PennyLaneCircuit(config)
        new_circuit.load_pca(components, mean)
        
        assert new_circuit.pca_fitted
        assert np.allclose(new_circuit.pca.components_, components)
        assert np.allclose(new_circuit.pca.mean_, mean)
    
    def test_pennylane_amplitude_encoding(self):
        from rafnet.quantum.branch import PennyLaneCircuit, QuantumConfig
        
        # Amplitude encoding needs pca_dim = 2^n_qubits
        config = QuantumConfig(n_qubits=2, n_layers=2, pca_dim=4, encoding="amplitude")
        circuit = PennyLaneCircuit(config)
        
        X = torch.randn(50, 32)
        circuit.fit_pca(X)
        
        # Amplitude embedding requires normalized inputs
        output = circuit(torch.randn(10, 32))
        
        assert output.shape == (10, 2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestQuantumBranchIntegration:
    def test_quantum_branch_init(self):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        
        config = QuantumConfig(n_qubits=8, n_layers=3, pca_dim=8, fallback_mode="classical")
        branch = QuantumBranch(config, input_dim=256)
        
        assert branch.input_dim == 256
        assert branch.config.n_qubits == 8
        assert branch.is_using_fallback()  # Should use fallback in test env
    
    def test_quantum_branch_fit_pca(self):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, fallback_mode="classical")
        branch = QuantumBranch(config, input_dim=64)
        
        X = torch.randn(50, 64)
        branch.fit_pca(X)
        
        assert branch.pca_fitted
    
    def test_quantum_branch_forward(self):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, fallback_mode="classical")
        branch = QuantumBranch(config, input_dim=64)
        
        X = torch.randn(50, 64)
        branch.fit_pca(X)
        
        output = branch(X[:10])
        
        assert output.shape == (10,)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_quantum_branch_get_quantum_features(self):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, fallback_mode="classical")
        branch = QuantumBranch(config, input_dim=64)
        
        X = torch.randn(50, 64)
        branch.fit_pca(X)
        
        features = branch.get_quantum_features(X[:10])
        
        assert features.shape == (10, 4)
    
    def test_quantum_branch_evaluation(self):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, fallback_mode="classical")
        branch = QuantumBranch(config, input_dim=64)
        
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        X[y == 1] += 2.0
        X[y == 0] -= 2.0
        
        branch.fit_pca(X)
        results = branch.evaluate(X, y, metrics=["accuracy", "precision", "recall", "f1", "roc_auc", "logloss", "eer"])
        
        assert "accuracy" in results
        assert "f1" in results
        assert "roc_auc" in results
        assert "eer" in results
        assert "latency" in results
        assert "num_parameters" in results
        assert "using_fallback" in results
        assert results["using_fallback"] is True
    
    def test_quantum_branch_serialization(self):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        import tempfile
        import os
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, fallback_mode="classical")
        branch = QuantumBranch(config, input_dim=64)
        
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        X[y == 1] += 2.0
        
        branch.fit_pca(X)
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        
        try:
            branch.save(temp_path)
            loaded = QuantumBranch.load(temp_path)
    
            assert loaded.pca_fitted
            assert loaded.input_dim == 64
            assert loaded.is_using_fallback()
    
            # Verify predictions match (both in eval mode)
            branch.eval()
            loaded.eval()
            orig_probs = branch(X[:10])
            loaded_probs = loaded(X[:10])
            assert torch.allclose(orig_probs, loaded_probs)
        finally:
            os.unlink(temp_path)
    
    def test_quantum_branch_parameter_count(self):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, fallback_mode="classical")
        branch = QuantumBranch(config, input_dim=64)
        
        total_params = branch.get_num_parameters()
        quantum_params = branch.get_quantum_parameters()
        head_params = branch.get_classical_head_parameters()
        
        assert total_params == quantum_params + head_params
        assert quantum_params > 0
        assert head_params > 0
    
    def test_quantum_branch_latency(self):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, fallback_mode="classical", measure_latency=True)
        branch = QuantumBranch(config, input_dim=64)
        
        X = torch.randn(50, 64)
        branch.fit_pca(X)
        
        for _ in range(3):
            _ = branch(X[:10])
        
        stats = branch.get_latency_stats()
        
        assert "mean_ms" in stats
        assert stats["total_samples"] > 0
    
    def test_quantum_branch_history(self):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        
        config = QuantumConfig(n_qubits=4, n_layers=2, pca_dim=4, fallback_mode="classical")
        branch = QuantumBranch(config, input_dim=64)
        
        X = torch.randn(50, 64)
        y = torch.randint(0, 2, (50,))
        
        branch.fit_pca(X)
        
        train_hist = branch.get_training_history()
        assert len(train_hist) == 1
        assert train_hist[0]["pca_fitted"] is True
        
        eval_hist = branch.get_eval_history()
        assert len(eval_hist) == 0
        
        branch.evaluate(X, y)
        eval_hist = branch.get_eval_history()
        assert len(eval_hist) == 1


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestQuantumBranchSmoke:
    def test_smoke_experiment(self):
        from rafnet.quantum.branch import run_quantum_smoke_experiment
        
        result = run_quantum_smoke_experiment(
            n_samples=100,
            n_features=64,
            n_qubits=4,
            n_layers=2,
            test_split=0.2,
            seed=42,
            verbose=False,
        )
        
        assert "branch" in result
        assert "results" in result
        assert "accuracy" in result["results"]
        assert result["results"]["using_fallback"] is True
    
    def test_config_comparison(self):
        from rafnet.quantum.branch import run_quantum_config_comparison
        
        configs = [
            {"name": "4q_2l", "n_qubits": 4, "n_layers": 2},
            {"name": "6q_3l", "n_qubits": 6, "n_layers": 3},
        ]
        
        results = run_quantum_config_comparison(
            configs,
            n_samples=100,
            n_features=64,
            test_split=0.2,
            seed=42,
        )
        
        assert "4q_2l" in results
        assert "6q_3l" in results
        assert results["4q_2l"]["success"] is True
        assert results["6q_3l"]["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])