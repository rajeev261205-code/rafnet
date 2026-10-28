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


class TestClassicalBranch:
    @pytest.mark.skipif(True, reason="XGBoost not available in test environment")
    def test_classical_branch_init(self, config):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        classical_config = ClassicalConfig(
            model_type=config.classical.model_type,
            xgboost_params=config.classical.xgboost_params,
        )
        branch = ClassicalBranch(classical_config, 256)
        assert branch is not None
        assert branch.input_dim == 256
    
    def test_logistic_regression_model(self):
        from rafnet.classical.branch import ClassicalConfig, create_classical_model
        config = ClassicalConfig(model_type="logistic")
        model = create_classical_model(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model.fit(X, y)
        proba = model.predict_proba(X)
        preds = model.predict(X)
        
        assert proba.shape == (50, 2)
        assert preds.shape == (50,)
    
    def test_svm_model(self):
        from rafnet.classical.branch import ClassicalConfig, create_classical_model
        config = ClassicalConfig(model_type="svm")
        model = create_classical_model(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model.fit(X, y)
        proba = model.predict_proba(X)
        preds = model.predict(X)
        
        assert proba.shape == (50, 2)
        assert preds.shape == (50,)
    
    def test_random_forest_model(self):
        from rafnet.classical.branch import ClassicalConfig, create_classical_model
        config = ClassicalConfig(model_type="random_forest")
        model = create_classical_model(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model.fit(X, y)
        proba = model.predict_proba(X)
        preds = model.predict(X)
        
        assert proba.shape == (50, 2)
        assert preds.shape == (50,)
    
    def test_mlp_model(self):
        from rafnet.classical.branch import ClassicalConfig, create_classical_model
        config = ClassicalConfig(model_type="mlp")
        model = create_classical_model(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model.fit(X, y)
        proba = model.predict_proba(X)
        preds = model.predict(X)
        
        assert proba.shape == (50, 2)
        assert preds.shape == (50,)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestQuantumBranch:
    def test_quantum_branch_init(self, config):
        try:
            from rafnet.quantum.branch import QuantumBranch, QuantumConfig
            quantum_config = QuantumConfig(
                n_qubits=config.quantum.n_qubits,
                n_layers=config.quantum.n_layers,
                pca_dim=config.quantum.pca_dim,
                shots=config.quantum.shots,
            )
            branch = QuantumBranch(quantum_config, 256)
            assert branch is not None
        except ImportError:
            pytest.skip("PennyLane not available")
    
    @pytest.mark.skipif(True, reason="PennyLane not available in test environment")
    def test_quantum_branch_fit_pca(self, config):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        import torch
        
        quantum_config = QuantumConfig(
            n_qubits=4,
            n_layers=2,
            pca_dim=4,
            shots=None,
        )
        branch = QuantumBranch(quantum_config, 256)
        
        X = torch.randn(20, 256)
        branch.fit_pca(X)
        
        assert branch.pca_fitted
    
    @pytest.mark.skipif(True, reason="PennyLane not available in test environment")
    def test_quantum_branch_forward(self, config):
        from rafnet.quantum.branch import QuantumBranch, QuantumConfig
        import torch
        
        quantum_config = QuantumConfig(
            n_qubits=4,
            n_layers=2,
            pca_dim=4,
            shots=None,
        )
        branch = QuantumBranch(quantum_config, 256)
        
        X = torch.randn(20, 256)
        branch.fit_pca(X)
        
        test_X = torch.randn(5, 256)
        output = branch(test_X)
        
        assert output.shape == (5,)
        assert torch.all(output >= 0) and torch.all(output <= 1)