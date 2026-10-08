"""
Experiment: Does Gateway 2 learn meaningful branch reliability?

Research Question:
    Does the learned gate β in Gateway 2 actually correlate with branch reliability,
    or does it simply collapse to favoring one branch (classical or quantum)?

Methodology:
    1. Generate synthetic branch probabilities with controlled reliability patterns
    2. Train Gateway 2 gate network to predict which branch is more reliable
    3. Analyze:
       - Gate distribution (β values)
       - Correlation between β and |p_c - p_q| (disagreement)
       - Conditional statistics: β when classical is correct vs wrong, quantum correct vs wrong
       - Compare against fixed fusion baselines

Expected outcomes if learning meaningful reliability:
    - β should be high when classical is more reliable (correct) and low when quantum is more reliable
    - β should correlate with disagreement (higher disagreement → gate makes meaningful choices)
    - Performance should exceed fixed fusion baselines

Expected outcomes if collapsing:
    - β distribution concentrated near 0 or 1
    - No correlation with branch correctness
    - Performance matches or worse than fixed fusion baselines
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from scipy.stats import pearsonr, spearmanr

# Add src to path
import sys
sys.path.insert(0, '/home/sk/rafnet/src')

from rafnet.fusion.gateway2 import Gateway2, Gateway2Config, FixedFusionGateway2


@dataclass
class ExperimentConfig:
    n_samples: int = 2000
    test_split: float = 0.2
    classical_reliability: float = 0.75  # Base accuracy of classical branch
    quantum_reliability: float = 0.70    # Base accuracy of quantum branch
    seed: int = 42
    gate_hidden_dim: int = 32
    gate_num_layers: int = 2
    training_epochs: int = 200


def generate_synthetic_branch_probs(
    n_samples: int,
    classical_acc: float,
    quantum_acc: float,
    correlation: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic branch probabilities with controlled reliability.
    
    Args:
        n_samples: Number of samples
        classical_acc: Target accuracy of classical branch
        quantum_acc: Target accuracy of quantum branch
        correlation: Correlation between branch errors (0 = independent, 1 = perfectly correlated)
        seed: Random seed
        
    Returns:
        p_c: Classical branch probabilities [n_samples]
        p_q: Quantum branch probabilities [n_samples]
        y_true: True labels [n_samples]
    """
    np.random.seed(seed)
    
    # Generate true labels
    y_true = np.random.randint(0, 2, n_samples)
    
    # Generate branch correctness with specified correlation
    # Use Gaussian copula
    rho = correlation
    z1 = np.random.randn(n_samples)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn(n_samples)
    
    # Convert to correctness
    class_correct = z1 < np.percentile(z1, classical_acc * 100)
    quantum_correct = z2 < np.percentile(z2, quantum_acc * 100)
    
    # Generate probabilities based on correctness
    # When correct: high confidence (0.7-0.95)
    # When wrong: low confidence (0.05-0.3)
    p_c = np.zeros(n_samples)
    p_q = np.zeros(n_samples)
    
    for i in range(n_samples):
        if class_correct[i]:
            p_c[i] = np.random.uniform(0.7, 0.95)
        else:
            p_c[i] = np.random.uniform(0.05, 0.3)
            
        if quantum_correct[i]:
            p_q[i] = np.random.uniform(0.7, 0.95)
        else:
            p_q[i] = np.random.uniform(0.05, 0.3)
    
    return p_c, p_q, y_true


def train_gateway2_gate(
    p_c: torch.Tensor,
    p_q: torch.Tensor,
    y_true: torch.Tensor,
    config: 'ExperimentConfig',
) -> Gateway2:
    """
    Train Gateway 2 gate network to predict branch reliability.
    
    Target: Prefer the branch that is more likely to be correct.
    """
    gateway2 = Gateway2(Gateway2Config(
        hidden_dim=config.gate_hidden_dim,
        num_layers=config.gate_num_layers,
    ))
    
    # Create target: which branch is correct?
    # 1 = prefer classical, 0 = prefer quantum, 0.5 = equal
    with torch.no_grad():
        class_pred = (p_c > 0.5).long()
        quantum_pred = (p_q > 0.5).long()
        y = y_true.long()
        
        class_correct = (class_pred == y)
        quantum_correct = (quantum_pred == y)
        
        target = torch.zeros_like(y, dtype=torch.float)
        target[class_correct & ~quantum_correct] = 1.0   # Classical correct, quantum wrong
        target[~class_correct & quantum_correct] = 0.0   # Quantum correct, classical wrong
        target[class_correct & quantum_correct] = 0.5    # Both correct
        target[~class_correct & ~quantum_correct] = 0.5  # Both wrong
    
    # Train
    gateway2 = Gateway2(Gateway2Config(
        hidden_dim=32,
        num_layers=2,
    ))
    optimizer = torch.optim.Adam(gateway2.gate_network.parameters(), lr=1e-3)
    
    gateway2.train()
    for epoch in range(200):
        optimizer.zero_grad()
        beta, _ = gateway2(p_c, p_q)
        loss = torch.nn.functional.binary_cross_entropy(beta.squeeze(), target)
        loss.backward()
        optimizer.step()
    
    gateway2.eval()
    return gateway2


def evaluate_fusion(p_c: np.ndarray, p_q: np.ndarray, y_true: np.ndarray, 
                    gateway2: Gateway2 = None) -> Dict:
    """Evaluate fusion strategies."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    p_c_np = p_c
    p_q_np = p_q
    
    p_c_t = torch.from_numpy(p_c_np).float()
    p_q_t = torch.from_numpy(p_q_np).float()
    
    results = {}
    
    # Fixed baselines
    for mode in ["classical_only", "quantum_only", "equal", "max", "min"]:
        fixed = FixedFusionGateway2(mode=mode)
        p_fused, _ = fixed(torch.from_numpy(p_c_np).float(), torch.from_numpy(p_q_np).float())
        p_fused_np = p_fused.numpy()
        
        preds = (p_fused_np > 0.5).astype(int)
        results[mode] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, p_fused_np)),
        }
    
    # Learned gate
    with torch.no_grad():
        p_fused, info = gateway2(p_c_t, p_q_t)
    p_fused_np = p_fused.numpy()
    beta_np = info["beta"].numpy()
    
    preds = (p_fused_np > 0.5).astype(int)
    results["learned"] = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p_fused_np)),
        "beta_stats": {
            "mean": float(info["beta"].mean()),
            "std": float(info["beta"].std()),
            "min": float(info["beta"].min()),
            "max": float(info["beta"].max()),
        }
    }
    
    return results, beta_np, info


def run_gateway2_reliability_experiment(config: 'ExperimentConfig') -> Dict:
    """Run the full Gateway 2 reliability experiment."""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    print(f"\n{'='*60}")
    print(f"Gateway 2 Reliability Experiment")
    print(f"{'='*60}")
    print(f"Samples: {config.n_samples}")
    print(f"Classical base accuracy: {config.classical_reliability}")
    print(f"Quantum base accuracy: {config.quantum_reliability}")
    
    # 1. Generate synthetic data
    print("\nGenerating synthetic branch probabilities...")
    p_c, p_q, y_true = generate_synthetic_branch_probs(
        n_samples=config.n_samples,
        classical_acc=config.classical_reliability,
        quantum_acc=config.quantum_reliability,
        correlation=0.3,
        seed=config.seed,
    )
    
    # Split
    n_test = int(config.n_samples * config.test_split)
    indices = np.random.permutation(config.n_samples)
    train_idx, test_idx = indices[n_test:], indices[:n_test]
    
    p_c_train, p_q_train, y_train = p_c[train_idx], p_q[train_idx], y_true[train_idx]
    p_c_test, p_q_test, y_test = p_c[test_idx], p_q[test_idx], y_true[test_idx]
    
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"Train class dist: {np.bincount(y_train)}")
    print(f"Test class dist: {np.bincount(y_test)}")
    
    # 2. Train Gateway 2
    print("\nTraining Gateway 2...")
    p_c_train_t = torch.from_numpy(p_c_train).float()
    p_q_train_t = torch.from_numpy(p_q_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    
    gateway2 = train_gateway2_gate(p_c_train_t, p_q_train_t, y_train_t, config)
    
    # 3. Evaluate
    print("\nEvaluating on test set...")
    results, beta_np, info = evaluate_fusion(
        p_c_test, p_q_test, y_test, gateway2
    )
    
    # 4. Analysis
    print(f"\n{'='*60}")
    print("GATE BEHAVIOR ANALYSIS")
    print(f"{'='*60}")
    
    beta = info["beta"].numpy()
    disagreement = info["disagreement"].numpy()
    
    # Correlation between beta and disagreement
    pearson_r, pearson_p = pearsonr(beta, disagreement)
    spearman_r, spearman_p = spearmanr(beta, disagreement)
    print(f"Beta vs Disagreement: Pearson r={pearson_r:.4f} (p={pearson_p:.4f}), Spearman r={spearman_r:.4f} (p={spearman_p:.4f})")
    
    # Conditional analysis
    p_c_np = p_c_test
    p_q_np = p_q_test
    y_true_np = y_test
    
    class_pred = (p_c_np > 0.5) == y_true_np
    quantum_pred = (p_q_np > 0.5) == y_true_np
    
    print(f"\nConditional Beta Analysis:")
    mask_cc_qw = class_pred & ~quantum_pred
    mask_qc_cw = ~class_pred & quantum_pred
    mask_both_c = class_pred & quantum_pred
    mask_both_w = ~class_pred & ~quantum_pred
    
    print(f"  Classical correct, Quantum wrong: beta={beta[mask_cc_qw].mean():.4f} (n={mask_cc_qw.sum()})")
    print(f"  Quantum correct, Classical wrong: beta={beta[mask_qc_cw].mean():.4f} (n={mask_qc_cw.sum()})")
    print(f"  Both correct: beta={beta[mask_both_c].mean():.4f} (n={mask_both_c.sum()})")
    print(f"  Both wrong: beta={beta[mask_both_w].mean():.4f} (n={mask_both_w.sum()})")
    
    # Gate distribution
    print(f"\nGate Distribution:")
    print(f"  Mean: {beta.mean():.4f}, Std: {beta.std():.4f}")
    print(f"  Min: {beta.min():.4f}, Max: {beta.max():.4f}")
    
    # 5. Summary comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    def compute_metrics(p_fused, y_true):
        preds = (p_fused > 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, p_fused)),
        }
    
    # Evaluate all methods
    p_c_test_t = torch.from_numpy(p_c_test).float()
    p_q_test_t = torch.from_numpy(p_q_test).float()
    
    all_results = {}
    
    # Fixed baselines
    for mode in ["classical_only", "quantum_only", "equal", "max", "min"]:
        fixed = FixedFusionGateway2(mode=mode)
        p_fused, _ = fixed(torch.from_numpy(p_c_test).float(), torch.from_numpy(p_q_test).float())
        all_results[mode] = compute_metrics(p_fused.numpy(), y_test)
    
    # Learned
    with torch.no_grad():
        p_fused_learned, info_learned = gateway2(torch.from_numpy(p_c_test).float(), torch.from_numpy(p_q_test).float())
    all_results["learned"] = compute_metrics(p_fused_learned.numpy(), y_test)
    all_results["learned"]["beta_stats"] = {
        "mean": float(info_learned["beta"].mean()),
        "std": float(info_learned["beta"].std()),
        "min": float(info_learned["beta"].min()),
        "max": float(info_learned["beta"].max()),
    }
    
    # Print comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    for mode, metrics in all_results.items():
        if mode != "learned" or "beta_stats" not in metrics:
            print(f"  {mode:15s}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
        else:
            print(f"  {mode:15s}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f} "
                  f"(β: μ={metrics['beta_stats']['mean']:.3f}, σ={metrics['beta_stats']['std']:.3f})")
    
    # Determine conclusion
    learned_acc = all_results["learned"]["accuracy"]
    best_fixed_acc = max(all_results[m]["accuracy"] for m in ["classical_only", "quantum_only", "equal"])
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    if learned_acc > best_fixed_acc + 0.01:
        print(f"✓ Gateway 2 LEARNS meaningful reliability (Acc={learned_acc:.4f} > Best Fixed={best_fixed_acc:.4f})")
        conclusion = "learns_reliability"
    elif learned_acc < best_fixed_acc - 0.01:
        print(f"✗ Gateway 2 WORSE than fixed fusion (Acc={learned_acc:.4f} < Best Fixed={best_fixed_acc:.4f})")
        conclusion = "worse_than_fixed"
    else:
        print(f"≈ Gateway 2 MATCHES fixed fusion (Acc={learned_acc:.4f} ≈ Best Fixed={best_fixed_acc:.4f})")
        conclusion = "matches_fixed"
    
    return {
        "config": config.__dict__,
        "results": all_results,
        "conclusion": conclusion,
        "gate_analysis": {
            "beta_mean": float(beta.mean()),
            "beta_std": float(beta.std()),
            "beta_min": float(beta.min()),
            "beta_max": float(beta.max()),
            "beta_disagreement_pearson_r": float(pearson_r),
            "beta_disagreement_spearman_r": float(spearman_r),
            "conditional_beta": {
                "classical_correct_quantum_wrong": float(beta[class_pred & ~quantum_pred].mean()) if (class_pred & ~quantum_pred).any() else None,
                "quantum_correct_classical_wrong": float(beta[~class_pred & quantum_pred].mean()) if (~class_pred & quantum_pred).any() else None,
                "both_correct": float(beta[class_pred & quantum_pred].mean()) if (class_pred & quantum_pred).any() else None,
                "both_wrong": float(beta[~class_pred & ~quantum_pred].mean()) if (~class_pred & ~quantum_pred).any() else None,
            }
        }
    }


if __name__ == "__main__":
    config = ExperimentConfig(
        n_samples=2000,
        classical_reliability=0.75,
        quantum_reliability=0.70,
        seed=42,
    )
    result = run_gateway2_reliability_experiment(config)
    
    # Save results
    output_path = Path("/home/sk/rafnet/experiments/gateway2_reliability_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")