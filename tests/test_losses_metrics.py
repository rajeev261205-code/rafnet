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


class TestMetrics:
    def test_compute_eer(self):
        from rafnet.evaluation.metrics import compute_eer
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        
        eer = compute_eer(y_true, y_score)
        
        assert 0 <= eer <= 1
    
    def test_compute_roc_curve(self):
        from rafnet.evaluation.metrics import compute_roc_curve
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        
        fpr, tpr, thresholds = compute_roc_curve(y_true, y_score)
        
        assert len(fpr) == len(tpr) == len(thresholds)
        # FPR should start at 0 and end at 1
        assert abs(fpr[0] - 0) < 1e-6
        assert abs(fpr[-1] - 1) < 1e-6
        # TPR should start at 0 and end at 1
        assert abs(tpr[0] - 0) < 1e-6
        assert abs(tpr[-1] - 1) < 1e-6
    
    def test_compute_ece(self):
        from rafnet.evaluation.metrics import compute_ece
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        
        ece = compute_ece(y_true, y_prob, n_bins=5)
        
        assert 0 <= ece <= 1
    
    def test_compute_risk_coverage(self):
        from rafnet.evaluation.metrics import compute_risk_coverage
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        decisions = np.array([0, 1, 2, 2, 1, 2, 0, 2])
        
        rc = compute_risk_coverage(y_true, y_prob, decisions, coverages=[0.5, 0.75])
        
        assert 0.5 in rc
        assert 0.75 in rc
    
    def test_evaluate_all(self):
        from rafnet.evaluation.metrics import evaluate_all
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        y_pred = (y_prob > 0.5).astype(int)
        decisions = np.random.choice([0, 1, 2], 100)
        
        metrics = evaluate_all(y_true, y_prob, y_pred, decisions)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "ece" in metrics
        assert "brier" in metrics


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestLosses:
    def test_cross_entropy_loss(self):
        from rafnet.losses.losses import CrossEntropyLoss
        import torch
        
        loss_fn = CrossEntropyLoss(weight=1.0)
        
        probs = torch.tensor([0.9, 0.1, 0.8, 0.2])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss = loss_fn(probs, labels)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_reliability_loss(self):
        from rafnet.losses.losses import ReliabilityLoss
        import torch
        
        loss_fn = ReliabilityLoss(weight=0.5)
        
        r_a = torch.tensor([[0.8], [0.3], [0.9], [0.2]])
        r_n = torch.tensor([[0.7], [0.4], [0.6], [0.1]])
        z_a = torch.randn(4, 256)
        z_n = torch.randn(4, 256)
        labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        alpha = torch.tensor([[0.7], [0.3], [0.8], [0.2]])
        
        loss = loss_fn(r_a, r_n, z_a, z_n, labels, alpha)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_selective_calibration_loss(self):
        from rafnet.losses.losses import SelectiveCalibrationLoss
        import torch
        
        loss_fn = SelectiveCalibrationLoss(weight=0.1)
        
        probs = torch.tensor([0.1, 0.5, 0.9, 0.2, 0.8])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
        
        loss = loss_fn(probs, labels, tau_low=0.3, tau_high=0.7)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_agreement_loss(self):
        from rafnet.losses.losses import AgreementLoss
        import torch
        
        loss_fn = AgreementLoss(weight=0.1)
        
        p_c = torch.tensor([0.9, 0.1, 0.8, 0.2])
        p_q = torch.tensor([0.8, 0.2, 0.7, 0.3])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        loss = loss_fn(p_c, p_q, labels)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_rafnet_loss(self, config):
        from rafnet.losses.losses import RAFNetLoss, LossConfig
        import torch
        
        loss_config = LossConfig(
            ce_weight=config.loss.ce_weight,
            rel_weight=config.loss.rel_weight,
            selcal_weight=config.loss.selcal_weight,
            agree_weight=config.loss.agree_weight,
        )
        loss_fn = RAFNetLoss(loss_config)
        
        probs = torch.rand(10)
        labels = torch.randint(0, 2, (10,)).float()
        p_c = torch.rand(10)
        p_q = torch.rand(10)
        r_a = torch.rand(10, 1)
        r_n = torch.rand(10, 1)
        z_a = torch.randn(10, 256)
        z_n = torch.randn(10, 256)
        alpha = torch.rand(10, 1)
        
        losses = loss_fn(
            probs=probs,
            labels=labels,
            p_c=p_c,
            p_q=p_q,
            r_a=r_a,
            r_n=r_n,
            z_a=z_a,
            z_n=z_n,
            alpha=alpha,
            tau_low=0.3,
            tau_high=0.7,
        )
        
        assert "total" in losses
        assert "ce" in losses
        assert "reliability" in losses
        assert "selcal" in losses
        assert "agree" in losses
        assert losses["total"].item() >= 0