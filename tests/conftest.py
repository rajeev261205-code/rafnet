import pytest
import sys
from pathlib import Path

import numpy as np

# Add src to path (repo-relative; works on any machine/OS).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


@pytest.fixture
def device():
    if HAS_TORCH:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


@pytest.fixture
def sample_audio():
    if HAS_TORCH:
        return torch.randn(2, 16000)
    return np.random.randn(2, 16000)


@pytest.fixture
def sample_network():
    if HAS_TORCH:
        return torch.randn(2, 20)
    return np.random.randn(2, 20)


@pytest.fixture
def sample_labels():
    if HAS_TORCH:
        return torch.tensor([0, 1])
    return np.array([0, 1])


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def config():
    from rafnet.config.schema import (
        RAFNetConfig, AudioConfig, NetworkConfig, Gateway1Config,
        ClassicalConfig, QuantumConfig, Gateway2Config, CalibrationConfig,
        DecisionGateConfig, LossConfig, TrainingConfig, EvaluationConfig,
        AblationConfig, DataConfig
    )
    return RAFNetConfig(
        device="cpu",
        audio=AudioConfig(wavlm_freeze=True, use_handcrafted=False),
        network=NetworkConfig(input_dim=20),
        gateway1=Gateway1Config(audio_dim=256, network_dim=256, hidden_dim=128),
        classical=ClassicalConfig(model_type="logistic"),
        quantum=QuantumConfig(n_qubits=8, n_layers=3, pca_dim=8, shots=None),
        gateway2=Gateway2Config(hidden_dim=32),
        calibration=CalibrationConfig(method="temperature"),
        decision_gate=DecisionGateConfig(optimize_thresholds=False),
        loss=LossConfig(ce_weight=1.0, rel_weight=0.0, selcal_weight=0.0, agree_weight=0.0),
        training=TrainingConfig(batch_size=2),
    )


@pytest.fixture
def synthetic_data():
    from rafnet.data.dataset import generate_synthetic_data, split_dataset
    samples = generate_synthetic_data(n_samples=100, seed=42)
    train, val, test = split_dataset(samples, seed=42)
    return {"train": train, "val": val, "test": test}


def pytest_configure(config):
    config.addinivalue_line("markers", "skip_torch: mark test as requiring torch")


@pytest.fixture(autouse=True)
def _isolate_from_real_store(monkeypatch):
    """Tests must never reach the real store: strip Supabase credentials so
    get_store() always resolves to a fresh in-memory adapter, even if the
    developer exported production env vars in their shell."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)


def pytest_runtest_setup(item):
    if "skip_torch" in item.keywords and not HAS_TORCH:
        pytest.skip("torch not available")