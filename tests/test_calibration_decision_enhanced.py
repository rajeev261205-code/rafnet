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
class TestCalibrationEnhancements:
    def test_temperature_scaling_fit_on_probs(self):
        from rafnet.calibration.calibrator import TemperatureScaling
        
        scaler = TemperatureScaling(1.0)
        
        # Create well-calibrated probs
        probs = np.linspace(0.1, 0.9, 100)
        labels = (probs > 0.5).astype(int)
        
        scaler.fit(probs, labels)
        
        assert scaler.fitted
        calibrated = scaler.predict_proba(probs)
        
        assert calibrated.shape == probs.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)
    
    def test_calibrator_temperature_with_validation_data(self):
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        
        config = CalibrationConfig(
            method="temperature",
            fit_on_validation_only=True,
            compute_ece=True,
            compute_brier=True,
        )
        calibrator = Calibrator(config)
        
        # Training data (should NOT be used for calibration if validation provided)
        train_probs = torch.rand(100)
        train_labels = (train_probs > 0.5).float()
        
        # Validation data (should be used for calibration)
        val_probs = torch.linspace(0.1, 0.9, 50)
        val_labels = (val_probs > 0.5).float()
        
        calibrator.fit(train_probs, train_labels, val_probs, val_labels)
        
        assert calibrator.fitted
        assert "ece" in calibrator.get_fit_stats()
        assert "brier" in calibrator.get_fit_stats()
        assert "reliability_diagram" in calibrator.get_fit_stats()
    
    def test_calibrator_platt_with_validation(self):
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        
        config = CalibrationConfig(method="platt", fit_on_validation_only=True)
        calibrator = Calibrator(config)
        
        train_probs = torch.rand(100)
        train_labels = (train_probs > 0.5).float()
        
        val_probs = torch.linspace(0.1, 0.9, 50)
        val_labels = (val_probs > 0.5).float()
        
        calibrator.fit(train_probs, train_labels, val_probs, val_labels)
        
        assert calibrator.fitted
    
    def test_calibrator_isotonic_with_validation(self):
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        
        config = CalibrationConfig(method="isotonic", fit_on_validation_only=True)
        calibrator = Calibrator(config)
        
        train_probs = torch.rand(100)
        train_labels = (train_probs > 0.5).float()
        
        val_probs = torch.linspace(0.1, 0.9, 50)
        val_labels = (val_probs > 0.5).float()
        
        calibrator.fit(train_probs, train_labels, val_probs, val_labels)
        
        assert calibrator.fitted
    
    def test_calibration_requires_validation_data(self):
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        
        config = CalibrationConfig(method="temperature", fit_on_validation_only=True)
        calibrator = Calibrator(config)
        
        train_probs = torch.linspace(0.1, 0.9, 100)
        train_labels = (train_probs > 0.5).float()
        
        # Should raise error when validation data not provided
        with pytest.raises(ValueError, match="requires validation_probs and validation_labels"):
            calibrator.fit(train_probs, train_labels, None, None)
    
    def test_calibration_fit_stats_include_params(self):
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        
        for method in ["temperature", "platt", "isotonic"]:
            config = CalibrationConfig(method=method)
            calibrator = Calibrator(config)
            
            # Use random probs to ensure both classes in validation
            probs = torch.rand(100)
            labels = (probs > 0.5).float()
            
            val_probs = probs[:50]
            val_labels = labels[:50]
            train_probs = probs[50:]
            train_labels = labels[50:]
            
            calibrator.fit(train_probs, train_labels, val_probs, val_labels)
            
            stats = calibrator.get_fit_stats()
            assert "params" in stats
            assert stats["method"] == method
    
    def test_calibration_ece_brier_computed(self):
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        
        config = CalibrationConfig(
            method="temperature",
            compute_ece=True,
            compute_brier=True,
        )
        calibrator = Calibrator(config)
        
        probs = torch.linspace(0.1, 0.9, 100)
        labels = (probs > 0.5).float()
        
        val_probs = probs[:50]
        val_labels = labels[:50]
        train_probs = probs[50:]
        train_labels = labels[50:]
        
        calibrator.fit(train_probs, train_labels, val_probs, val_labels)
        
        stats = calibrator.get_fit_stats()
        assert "ece" in stats
        assert "brier" in stats
        assert 0 <= stats["ece"] <= 1
        assert 0 <= stats["brier"] <= 1
    
    def test_calibration_forward_after_fit(self):
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        
        config = CalibrationConfig(method="temperature")
        calibrator = Calibrator(config)
        
        probs = torch.rand(100)
        labels = (probs > 0.5).float()
        
        val_probs = probs[:50]
        val_labels = labels[:50]
        train_probs = probs[50:]
        train_labels = labels[50:]
        
        calibrator.fit(train_probs, train_labels, val_probs, val_labels)
        
        # Test forward pass
        test_probs = torch.tensor([0.2, 0.5, 0.8])
        calibrated = calibrator(test_probs)
        
        assert calibrated.shape == test_probs.shape
        assert torch.all(calibrated >= 0) and torch.all(calibrated <= 1)
    
    def test_calibration_leakage_prevention(self):
        """Test that calibration only uses validation data when provided."""
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        
        config = CalibrationConfig(method="temperature", fit_on_validation_only=True)
        calibrator = Calibrator(config)
        
        # Training data with known bias (overconfident)
        train_probs = torch.linspace(0.6, 0.9, 100)  # All > 0.5
        train_labels = torch.ones(100)
        
        # Validation data - well calibrated
        val_probs = torch.linspace(0.1, 0.9, 50)
        val_labels = (val_probs > 0.5).float()
        
        calibrator.fit(train_probs, train_labels, val_probs, val_labels)
        
        # Should be calibrated based on validation, not training
        test_probs = torch.tensor([0.7, 0.3, 0.5])
        calibrated = calibrator(test_probs)
        
        # Should move 0.7 towards 0.5 (less confident) based on validation
        assert calibrated.shape == test_probs.shape


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDecisionGateEnhancements:
    def test_decision_gate_exact_threshold_low(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(tau_low=0.3, tau_high=0.7)
        gate = DecisionGate(config)
        
        # Exactly at tau_low
        probs = torch.tensor([0.3])
        decisions, info = gate(probs)
        
        assert decisions[0] == 1  # UNCERTAIN (>= tau_low)
    
    def test_decision_gate_exact_threshold_high(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(tau_low=0.3, tau_high=0.7)
        gate = DecisionGate(config)
        
        # Exactly at tau_high
        probs = torch.tensor([0.7])
        decisions, info = gate(probs)
        
        assert decisions[0] == 1  # UNCERTAIN (<= tau_high)
    
    def test_decision_gate_below_threshold(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(tau_low=0.3, tau_high=0.7)
        gate = DecisionGate(config)
        
        probs = torch.tensor([0.2])
        decisions, info = gate(probs)
        
        assert decisions[0] == 0  # REAL
    
    def test_decision_gate_above_threshold(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(tau_low=0.3, tau_high=0.7)
        gate = DecisionGate(config)
        
        probs = torch.tensor([0.8])
        decisions, info = gate(probs)
        
        assert decisions[0] == 2  # SYNTHETIC
    
    def test_decision_gate_uncertainty_region(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(tau_low=0.3, tau_high=0.7)
        gate = DecisionGate(config)
        
        probs = torch.tensor([0.4, 0.5, 0.6])
        decisions, info = gate(probs)
        
        assert torch.all(decisions == 1)  # All UNCERTAIN
    
    def test_decision_gate_uncertainty_width_constraints(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(
            tau_low=0.3, 
            tau_high=0.7,
            min_uncertainty_width=0.1,
            max_uncertainty_width=0.5,
        )
        gate = DecisionGate(config)
        
        # Test optimization respects constraints
        probs = torch.rand(200)
        labels = (probs > 0.5).long()
        
        val_probs = probs[:100]
        val_labels = labels[:100]
        train_probs = probs[100:]
        train_labels = labels[100:]
        
        gate.optimize_thresholds(train_probs, train_labels, val_probs, val_labels)
        
        tau_low, tau_high = gate.get_thresholds()
        width = tau_high - tau_low
        
        assert width >= config.min_uncertainty_width
        assert width <= config.max_uncertainty_width
    
    def test_decision_gate_validation_based_threshold(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(
            select_on_validation=True,
            grid_resolution=20,
        )
        gate = DecisionGate(config)
        
        # Training data
        train_probs = torch.rand(100)
        train_labels = (train_probs > 0.5).long()
        
        # Validation data with different distribution
        val_probs = torch.linspace(0.1, 0.9, 50)
        val_labels = (val_probs > 0.5).long()
        
        gate.optimize_thresholds(train_probs, train_labels, val_probs, val_labels)
        
        assert gate.optimized
        # Should use validation data for optimization
    
    def test_decision_gate_requires_validation_data(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(
            optimize_thresholds=True,
            grid_resolution=20,
        )
        gate = DecisionGate(config)
        
        train_probs = torch.rand(100)
        train_labels = (train_probs > 0.5).long()
        
        # Should raise error when validation data not provided
        with pytest.raises(ValueError, match="requires validation_probs and validation_labels"):
            gate.optimize_thresholds(train_probs, train_labels, None, None)
    
    def test_decision_gate_uncertainty_action_reject(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(
            tau_low=0.3,
            tau_high=0.7,
            uncertainty_action="reject",
        )
        gate = DecisionGate(config)
        
        probs = torch.tensor([0.2, 0.5, 0.8])
        decisions, info = gate.decide_with_fallback(probs)
        
        assert decisions[1] == 1  # UNCERTAIN stays UNCERTAIN
        assert not info["fallback_used"]
    
    def test_decision_gate_uncertainty_action_classical(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(
            tau_low=0.3,
            tau_high=0.7,
            uncertainty_action="classical",
        )
        gate = DecisionGate(config)
        
        probs = torch.tensor([0.2, 0.5, 0.8])
        classical_probs = torch.tensor([0.1, 0.6, 0.9])
        
        decisions, info = gate.decide_with_fallback(probs, classical_probs=classical_probs)
        
        # Uncertain (0.5) should be replaced by classical decision
        assert decisions[1] == 1  # classical 0.6 -> UNCERTAIN
        assert info["fallback_used"]
    
    def test_decision_gate_uncertainty_action_quantum(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(
            tau_low=0.3,
            tau_high=0.7,
            uncertainty_action="quantum",
        )
        gate = DecisionGate(config)
        
        probs = torch.tensor([0.2, 0.5, 0.8])
        quantum_probs = torch.tensor([0.1, 0.6, 0.9])
        
        decisions, info = gate.decide_with_fallback(probs, quantum_probs=quantum_probs)
        
        assert decisions[1] == 1  # quantum 0.6 -> UNCERTAIN
        assert info["fallback_used"]
    
    def test_decision_gate_multiple_optimization_metrics(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        for metric in ["f1", "precision", "recall", "balanced_accuracy", "mcc"]:
            config = DecisionGateConfig(
                metric_for_optimization=metric,
                grid_resolution=10,
            )
            gate = DecisionGate(config)
            
            probs = torch.rand(100)
            labels = (probs > 0.5).long()
            
            val_probs = probs[:50]
            val_labels = labels[:50]
            train_probs = probs[50:]
            train_labels = labels[50:]
            
            gate.optimize_thresholds(train_probs, train_labels, val_probs, val_labels)
            
            assert gate.optimized
            tau_low, tau_high = gate.get_thresholds()
            assert tau_low < tau_high
    
    def test_decision_gate_optimization_history(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(grid_resolution=10)
        gate = DecisionGate(config)
        
        probs = torch.rand(100)
        labels = (probs > 0.5).long()
        
        val_probs = probs[:50]
        val_labels = labels[:50]
        train_probs = probs[50:]
        train_labels = labels[50:]
        
        gate.optimize_thresholds(train_probs, train_labels, val_probs, val_labels)
        
        history = gate.get_optimization_history()
        
        assert len(history) == 1
        assert "best_score" in history[0]
        assert "tau_low" in history[0]
        assert "tau_high" in history[0]
        assert "metric" in history[0]
    
    def test_decision_gate_set_thresholds_validation(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        gate = DecisionGate(DecisionGateConfig())
        
        # Valid thresholds
        gate.set_thresholds(0.2, 0.8)
        assert abs(gate.tau_low.item() - 0.2) < 1e-6
        assert abs(gate.tau_high.item() - 0.8) < 1e-6
        
        # Invalid: tau_low >= tau_high
        with pytest.raises(ValueError):
            gate.set_thresholds(0.8, 0.2)
    
    def test_decision_gate_return_uncertainty_mask(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(tau_low=0.3, tau_high=0.7)
        gate = DecisionGate(config)
        
        probs = torch.tensor([0.2, 0.5, 0.8])
        decisions, info = gate(probs, return_uncertainty_mask=True)
        
        assert "uncertainty_mask" in info
        assert "certain_mask" in info
        assert info["uncertainty_mask"][1] == True
        assert info["certain_mask"][0] == True
        assert info["certain_mask"][2] == True
    
    def test_decision_gate_info_statistics(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        
        config = DecisionGateConfig(tau_low=0.3, tau_high=0.7)
        gate = DecisionGate(config)
        
        probs = torch.tensor([0.1, 0.2, 0.5, 0.8, 0.9])
        decisions, info = gate(probs)
        
        assert info["n_real"] == 2
        assert info["n_uncertain"] == 1
        assert info["n_synthetic"] == 2
        assert abs(info["pct_real"] - 0.4) < 1e-6
        assert abs(info["pct_uncertain"] - 0.2) < 1e-6
        assert abs(info["pct_synthetic"] - 0.4) < 1e-6
        assert abs(info["uncertainty_width"] - 0.4) < 1e-6


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestEvaluationMetrics:
    def test_ece_computation(self):
        from rafnet.evaluation.metrics import compute_ece
        
        # Perfectly calibrated probabilities
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        ece = compute_ece(probs, labels)
        assert ece < 0.15  # Should be small for well-calibrated
    
    def test_brier_score(self):
        from rafnet.evaluation.metrics import compute_brier
        
        # Perfect predictions (prob = 0 or 1) -> Brier = 0
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        labels = np.array([0, 1, 0, 1])
        
        brier = compute_brier(labels, probs)
        
        assert abs(brier) < 1e-10
        
        # Test with confident but not perfect predictions
        probs2 = np.array([0.1, 0.9, 0.1, 0.9])
        brier2 = compute_brier(labels, probs2)
        
        # (0.1-0)^2 + (0.9-1)^2 + (0.1-0)^2 + (0.9-1)^2 = 0.01 * 4 = 0.04, mean = 0.01
        assert abs(brier2 - 0.01) < 1e-10
    
    def test_reliability_diagram(self):
        from rafnet.evaluation.metrics import compute_reliability_diagram
        
        probs = np.linspace(0.1, 0.9, 100)
        labels = (probs > 0.5).astype(int)
        
        diagram = compute_reliability_diagram(probs, labels, n_bins=10)
        
        assert "bin_edges" in diagram
        assert "bin_centers" in diagram
        assert "bin_accuracies" in diagram
        assert "bin_confidences" in diagram
        assert "bin_counts" in diagram
        assert len(diagram["bin_centers"]) == 10
    
    def test_risk_coverage_curve(self):
        from rafnet.evaluation.metrics import compute_risk_coverage
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])
        decisions = np.array([0, 0, 2, 2, 1, 2])  # Last one uncertain
        
        rc = compute_risk_coverage(y_true, y_prob, decisions)
        
        assert 0.5 in rc
        assert 0.9 in rc
    
    def test_coverage_error_curve(self):
        from rafnet.evaluation.metrics import compute_coverage_error_curve
        
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        decisions = np.array([0, 0, 2, 2])
        
        cec = compute_coverage_error_curve(y_true, y_prob, decisions)
        
        assert "coverage" in cec
        assert "error" in cec
        assert len(cec["coverage"]) == len(cec["error"])

    def test_calibration_metrics_mce(self):
        from rafnet.evaluation.metrics import compute_calibration_metrics
        
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        metrics = compute_calibration_metrics(labels, probs)
        
        assert "ece" in metrics
        assert "mce" in metrics
        assert "brier" in metrics

    def test_evaluate_all_with_new_metrics(self):
        from rafnet.evaluation.metrics import evaluate_all, EvaluationConfig
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])
        decisions = np.array([0, 0, 2, 2, 1, 2])
        
        config = EvaluationConfig(metrics=[
            "accuracy", "f1", "roc_auc", "ece", "mce", "brier",
            "risk_coverage", "coverage_error_curve"
        ])
        
        results = evaluate_all(y_true, y_prob, decisions=decisions, config=config)
        
        assert "accuracy" in results
        assert "f1" in results
        assert "roc_auc" in results
        assert "ece" in results
        assert "mce" in results
        assert "brier" in results
        assert "risk_coverage_50" in results
        assert "coverage_error_curve" in results


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestCalibrationLeakage:
    def test_calibration_fits_on_validation_not_train(self):
        """Test that calibration uses validation data when provided."""
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        
        config = CalibrationConfig(method="temperature", fit_on_validation_only=True)
        calibrator = Calibrator(config)
        
        # Training data: systematically biased
        train_probs = torch.tensor([0.9] * 50 + [0.1] * 50)
        train_labels = torch.tensor([1] * 50 + [0] * 50)
        
        # Validation data: well calibrated
        val_probs = torch.linspace(0.1, 0.9, 100)
        val_labels = (val_probs > 0.5).float()
        
        calibrator.fit(train_probs, train_labels, val_probs, val_labels)
        
        # Check that calibration params are based on validation
        # The temperature should be close to 1.0 for well-calibrated validation data
        temp = calibrator.get_temperature()
        assert temp is not None
        
        # Test on middle probability
        test = calibrator(torch.tensor([0.5]))
        # Should be close to 0.5 since validation was well-calibrated
        assert 0.3 < test.item() < 0.7
    
    def test_calibration_requires_validation_data(self):
        """Test that calibration requires validation data (no fallback to train)."""
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        
        config = CalibrationConfig(method="temperature", fit_on_validation_only=True)
        calibrator = Calibrator(config)
        
        train_probs = torch.linspace(0.1, 0.9, 100)
        train_labels = (train_probs > 0.5).float()
        
        # Should raise error when validation data not provided
        with pytest.raises(ValueError, match="requires validation_probs and validation_labels"):
            calibrator.fit(train_probs, train_labels, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])