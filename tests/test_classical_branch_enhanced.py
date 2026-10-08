import pytest
import sys
sys.path.insert(0, '/home/sk/rafnet/src')

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
class TestClassicalBranchEnhancements:
    def test_classical_branch_fit_predict(self):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        
        config = ClassicalConfig(model_type="logistic")
        branch = ClassicalBranch(config, input_dim=64)
        
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        
        branch.fit(X, y)
        assert branch.is_fitted
        
        probs = branch(X)
        assert probs.shape == (100,)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        
        preds = branch.predict(X)
        assert preds.shape == (100,)
        assert torch.all((preds == 0) | (preds == 1))
        
        probas = branch.predict_proba(X)
        assert probas.shape == (100, 2)
        assert torch.allclose(probas.sum(dim=1), torch.ones(100))
    
    def test_classical_branch_calibration(self):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        
        config = ClassicalConfig(model_type="logistic", use_calibration=True)
        branch = ClassicalBranch(config, input_dim=64)
        
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        
        branch.fit(X, y)
        branch.calibrate(X, y)
        
        assert branch.is_calibrated()
    
    def test_classical_branch_evaluation(self):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        
        config = ClassicalConfig(model_type="logistic")
        branch = ClassicalBranch(config, input_dim=64)
        
        # Create separable data
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        X[y == 1] += 2.0
        X[y == 0] -= 2.0
        
        branch.fit(X, y)
        
        results = branch.evaluate(X, y)
        
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "roc_auc" in results
        assert "logloss" in results
        assert "latency" in results
        
        assert results["accuracy"] > 0.9
    
    def test_classical_branch_serialization(self):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        import tempfile
        import os
        
        config = ClassicalConfig(model_type="logistic")
        branch = ClassicalBranch(config, input_dim=64)
        
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        X[y == 1] += 2.0
        X[y == 0] -= 2.0
        
        branch.fit(X, y)
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        
        try:
            branch.save(temp_path)
            loaded = ClassicalBranch.load(temp_path)
            
            assert loaded.is_fitted
            assert loaded.input_dim == 64
            assert loaded.config.model_type == "logistic"
            
            # Verify predictions match
            orig_probs = branch(X[:10])
            loaded_probs = loaded(X[:10])
            assert torch.allclose(orig_probs, loaded_probs)
        finally:
            os.unlink(temp_path)
    
    def test_classical_branch_latency_tracking(self):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        
        config = ClassicalConfig(model_type="logistic", measure_latency=True)
        branch = ClassicalBranch(config, input_dim=64)
        
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        
        branch.fit(X, y)
        
        # Run inference multiple times
        for _ in range(5):
            _ = branch(X[:10])
        
        stats = branch.get_latency_stats()
        
        assert "mean_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert stats["total_samples"] > 0
    
    def test_classical_branch_reset_latency(self):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        
        config = ClassicalConfig(model_type="logistic", measure_latency=True)
        branch = ClassicalBranch(config, input_dim=64)
        
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        
        branch.fit(X, y)
        _ = branch(X[:10])
        
        stats = branch.get_latency_stats()
        assert stats["total_samples"] > 0
        
        branch.reset_latency_stats()
        stats = branch.get_latency_stats()
        assert stats.get("total_samples", 0) == 0
    
    def test_classical_branch_feature_names(self):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        
        config = ClassicalConfig(model_type="logistic")
        branch = ClassicalBranch(config, input_dim=64)
        
        feature_names = [f"feat_{i}" for i in range(64)]
        branch.set_feature_names(feature_names)
        
        assert branch.get_feature_names() == feature_names
        assert branch.model.get_feature_names() == feature_names
    
    def test_classical_branch_get_params(self):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        
        config = ClassicalConfig(model_type="logistic")
        branch = ClassicalBranch(config, input_dim=64)
        
        params = branch.get_params()
        assert "max_iter" in params
        assert "random_state" in params
    
    def test_classical_branch_history(self):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        
        config = ClassicalConfig(model_type="logistic")
        branch = ClassicalBranch(config, input_dim=64)
        
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        
        branch.fit(X, y)
        
        train_hist = branch.get_training_history()
        assert len(train_hist) == 1
        assert train_hist[0]["n_samples"] == 100
        assert train_hist[0]["n_features"] == 64
        
        eval_hist = branch.get_eval_history()
        assert len(eval_hist) == 0  # Not evaluated yet
        
        branch.evaluate(X, y)
        eval_hist = branch.get_eval_history()
        assert len(eval_hist) == 1
    
    def test_classical_branch_predict_proba_shape(self):
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        
        config = ClassicalConfig(model_type="logistic")
        branch = ClassicalBranch(config, input_dim=64)
        
        X = torch.randn(100, 64)
        y = torch.randint(0, 2, (100,))
        X[y == 1] += 2.0
        
        branch.fit(X, y)
        
        probas = branch.predict_proba(X[:5])
        assert probas.shape == (5, 2)
        assert torch.allclose(probas.sum(dim=1), torch.ones(5))
    
    def test_different_model_types_consistency(self):
        """Test that all model types produce valid outputs."""
        from rafnet.classical.branch import ClassicalBranch, ClassicalConfig
        
        for model_type in ["logistic", "svm", "random_forest", "mlp"]:
            config = ClassicalConfig(model_type=model_type)
            branch = ClassicalBranch(config, input_dim=32)
            
            X = torch.randn(50, 32)
            y = torch.randint(0, 2, (50,))
            X[y == 1] += 1.5
            X[y == 0] -= 1.5
            
            branch.fit(X, y)
            
            probs = branch(X[:5])
            assert probs.shape == (5,)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)
            
            preds = branch.predict(X[:5])
            assert preds.shape == (5,)
            
            probas = branch.predict_proba(X[:5])
            assert probas.shape == (5, 2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestClassicalConfigEnhancements:
    def test_config_defaults(self):
        from rafnet.classical.branch import ClassicalConfig
        
        config = ClassicalConfig()
        
        assert config.model_type == "xgboost"
        assert config.use_calibration is True
        assert config.calibration_method == "sigmoid"
        assert config.random_state == 42
        assert config.measure_latency is True
        assert config.calibration_cv == 3
        assert "accuracy" in config.eval_metrics
        assert "roc_auc" in config.eval_metrics
    
    def test_config_custom(self):
        from rafnet.classical.branch import ClassicalConfig
        
        config = ClassicalConfig(
            model_type="random_forest",
            use_calibration=False,
            calibration_method="isotonic",
            random_state=123,
            measure_latency=False,
            eval_metrics=["accuracy", "f1"],
        )
        
        assert config.model_type == "random_forest"
        assert config.use_calibration is False
        assert config.calibration_method == "isotonic"
        assert config.random_state == 123
        assert config.measure_latency is False
        assert config.eval_metrics == ["accuracy", "f1"]


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestBaseClassicalModelMethods:
    def test_calibrate_method(self):
        from rafnet.classical.branch import ClassicalConfig, create_classical_model
        
        config = ClassicalConfig(model_type="logistic")
        model = create_classical_model(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model.fit(X, y)
        assert not model.is_calibrated()
        
        model.calibrate(X, y, method="sigmoid", cv=3)
        assert model.is_calibrated()
    
    def test_evaluate_method(self):
        from rafnet.classical.branch import ClassicalConfig, create_classical_model
        
        config = ClassicalConfig(model_type="logistic")
        model = create_classical_model(config)
        
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        X[y == 1] += 2.0
        X[y == 0] -= 2.0
        
        model.fit(X, y)
        
        results = model.evaluate(X, y, metrics=["accuracy", "f1"])
        
        assert "accuracy" in results
        assert "f1" in results
        assert results["accuracy"] > 0.9
    
    def test_save_load_model(self):
        from rafnet.classical.branch import ClassicalConfig, create_classical_model
        import tempfile
        import os
        
        config = ClassicalConfig(model_type="logistic")
        model = create_classical_model(config)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        X[y == 1] += 2.0
        X[y == 0] -= 2.0
        
        model.fit(X, y)
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            # Load via the concrete class
            if config.model_type == "xgboost":
                from rafnet.classical.branch import XGBoostModel
                loaded = XGBoostModel.load(temp_path)
            elif config.model_type == "logistic":
                from rafnet.classical.branch import LogisticRegressionModel
                loaded = LogisticRegressionModel.load(temp_path)
            elif config.model_type == "svm":
                from rafnet.classical.branch import SVMModel
                loaded = SVMModel.load(temp_path)
            elif config.model_type == "random_forest":
                from rafnet.classical.branch import RandomForestModel
                loaded = RandomForestModel.load(temp_path)
            elif config.model_type == "mlp":
                from rafnet.classical.branch import MLPModel
                loaded = MLPModel.load(temp_path)
            
            orig_proba = model.predict_proba(X[:5])
            loaded_proba = loaded.predict_proba(X[:5])
            
            assert np.allclose(orig_proba, loaded_proba)
        finally:
            os.unlink(temp_path)


# Need to import BaseClassicalModel for the last test
from rafnet.classical.branch import BaseClassicalModel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])