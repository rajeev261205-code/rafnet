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


class TestCalibration:
    def test_platt_scaling(self):
        from rafnet.calibration.calibrator import PlattScaling
        scaler = PlattScaling()
        probs = np.random.rand(100)
        labels = (probs > 0.5).astype(int)
        
        scaler.fit(probs, labels)
        calibrated = scaler.predict_proba(probs)
        
        assert calibrated.shape == probs.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)
    
    def test_isotonic_calibration(self):
        from rafnet.calibration.calibrator import IsotonicCalibration
        scaler = IsotonicCalibration()
        probs = np.random.rand(100)
        labels = (probs > 0.5).astype(int)
        
        scaler.fit(probs, labels)
        calibrated = scaler.predict_proba(probs)
        
        assert calibrated.shape == probs.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)
        assert np.all(np.diff(calibrated[np.argsort(probs)]) >= -1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestCalibrationTorch:
    def test_temperature_scaling_init(self):
        from rafnet.calibration.calibrator import TemperatureScaling
        scaler = TemperatureScaling(1.5)
        assert scaler.temperature.item() == 1.5
    
    def test_temperature_scaling_forward(self):
        from rafnet.calibration.calibrator import TemperatureScaling
        import torch
        
        scaler = TemperatureScaling(2.0)
        logits = torch.tensor([1.0, -1.0, 2.0, -2.0])
        
        scaled = scaler(logits)
        
        assert scaled.shape == logits.shape
        assert torch.allclose(scaled, logits / 2.0)
    
    def test_temperature_scaling_fit(self):
        from rafnet.calibration.calibrator import TemperatureScaling
        import torch
        
        scaler = TemperatureScaling(1.0)
        logits = torch.randn(100) * 2
        labels = (torch.sigmoid(logits) > 0.5).float()
        
        scaler.fit(logits, labels, max_iter=50, lr=0.1)
        
        # Temperature can be negative during optimization but clamped in forward
        assert scaler.temperature.item() != 0
    
    def test_calibrator_temperature(self, config):
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        import torch
        
        calib_config = CalibrationConfig(
            method=config.calibration.method,
            temperature_init=config.calibration.temperature_init,
        )
        calibrator = Calibrator(calib_config)
        
        probs = torch.rand(100)
        labels = (probs > 0.5).float()
        
        # Split into train/val
        val_probs = probs[:50]
        val_labels = labels[:50]
        train_probs = probs[50:]
        train_labels = labels[50:]
        
        calibrator.fit(train_probs, train_labels, val_probs, val_labels)
        
        assert calibrator.fitted
        calibrated = calibrator(val_probs)
        
        assert calibrated.shape == val_probs.shape
        assert torch.all(calibrated >= 0) and torch.all(calibrated <= 1)
    
    def test_calibrator_platt(self, config):
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        import torch
        
        calib_config = CalibrationConfig(method="platt")
        calibrator = Calibrator(calib_config)
        
        probs = torch.rand(100)
        labels = (probs > 0.5).float()
        
        val_probs = probs[:50]
        val_labels = labels[:50]
        train_probs = probs[50:]
        train_labels = labels[50:]
        
        calibrator.fit(train_probs, train_labels, val_probs, val_labels)
        calibrated = calibrator(val_probs)
        
        assert calibrator.fitted
        assert calibrated.shape == val_probs.shape
    
    def test_calibrator_isotonic(self, config):
        from rafnet.calibration.calibrator import Calibrator, CalibrationConfig
        import torch
        
        calib_config = CalibrationConfig(method="isotonic")
        calibrator = Calibrator(calib_config)
        
        probs = torch.rand(100)
        labels = (probs > 0.5).float()
        
        val_probs = probs[:50]
        val_labels = labels[:50]
        train_probs = probs[50:]
        train_labels = labels[50:]
        
        calibrator.fit(train_probs, train_labels, val_probs, val_labels)
        calibrated = calibrator(val_probs)
        
        assert calibrator.fitted
        assert calibrated.shape == val_probs.shape


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDecisionGate:
    def test_decision_gate_init(self, config):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        gate_config = DecisionGateConfig(
            tau_low=config.decision_gate.tau_low,
            tau_high=config.decision_gate.tau_high,
        )
        gate = DecisionGate(gate_config)
        
        assert abs(gate.tau_low.item() - config.decision_gate.tau_low) < 1e-6
        assert abs(gate.tau_high.item() - config.decision_gate.tau_high) < 1e-6
    
    def test_decision_gate_forward(self, config):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        import torch
        
        gate_config = DecisionGateConfig(
            tau_low=0.3,
            tau_high=0.7,
        )
        gate = DecisionGate(gate_config)
        
        probs = torch.tensor([0.1, 0.5, 0.9])
        decisions, info = gate(probs)
        
        assert decisions.shape == probs.shape
        assert decisions[0] == 0  # REAL
        assert decisions[1] == 1  # UNCERTAIN
        assert decisions[2] == 2  # SYNTHETIC
        
        assert info["n_real"] == 1
        assert info["n_uncertain"] == 1
        assert info["n_synthetic"] == 1
    
    def test_decision_gate_optimize_thresholds(self, config):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        import torch
        
        gate_config = DecisionGateConfig(
            tau_low=0.3,
            tau_high=0.7,
            optimize_thresholds=True,
            metric_for_optimization="f1",
            grid_resolution=20,
        )
        gate = DecisionGate(gate_config)
        
        probs = torch.rand(200)
        labels = (probs > 0.5).long()
        
        # Split into train/val
        val_probs = probs[:100]
        val_labels = labels[:100]
        train_probs = probs[100:]
        train_labels = labels[100:]
        
        gate.optimize_thresholds(train_probs, train_labels, val_probs, val_labels)
        
        assert gate.optimized
        tau_low, tau_high = gate.get_thresholds()
        assert 0 <= tau_low < tau_high <= 1
    
    def test_decision_gate_set_thresholds(self):
        from rafnet.decision.gate import DecisionGate, DecisionGateConfig
        import torch
        
        gate = DecisionGate(DecisionGateConfig())
        
        gate.set_thresholds(0.2, 0.8)
        
        assert abs(gate.tau_low.item() - 0.2) < 1e-6
        assert abs(gate.tau_high.item() - 0.8) < 1e-6