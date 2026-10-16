"""Tests for tools/evaluate_model.py helpers (no torch/model needed).

Uses a stub model with fixed outputs: validates the evaluation PLUMBING
(windowing, aggregation, metrics), never detector accuracy.
"""
import importlib.util


def _load():
    spec = importlib.util.spec_from_file_location("rafnet_eval_model", "tools/evaluate_model.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _wav(path, seconds=2.5, sr=16000, freq=440.0):
    import io
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(int(seconds * sr)):
            w.writeframes(struct.pack("<h", int(9000 * math.sin(2 * math.pi * freq * i / sr))))
    path.write_bytes(buf.getvalue())


class _Stub:
    def __init__(self, p=0.7):
        self.p = p

    def predict(self, samples, sr=16000):
        from rafnet.soc.model_adapter import VoiceResult

        return VoiceResult(prediction="UNCERTAIN", confidence=0.7,
                           prob_synthetic=self.p, prob_real=1 - self.p)


def test_score_file_windows_and_agg(tmp_path):
    ev = _load()
    wav = tmp_path / "s.wav"
    _wav(wav)
    entry = ev.score_file(wav, _Stub(0.7), seed=7)
    assert "error" not in entry
    assert "agg_error" not in entry, entry.get("agg_error")
    assert entry["n_windows"] >= 2
    assert entry["p_synthetic_agg"] == 0.7


def test_metrics_helpers_sane():
    ev = _load()
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.9, 0.95]
    assert ev.strict_accuracy(y, s, 0.3611, 0.8611) == 1.0
    assert ev.compute_auc(y, s) == 1.0
    assert ev.compute_eer(y, s)["eer"] == 0.0
    m = ev.metrics_block(y, s, 0.3611, 0.8611, "probe")
    assert m["confusion_3way"] == {"real": {"REAL": 2, "UNCERTAIN": 0, "SYNTHETIC": 0},
                                   "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 2}}


def test_calibration_needs_both_classes():
    ev = _load()
    r = ev.grid_calibrate([0, 0, 0], [0.1, 0.2, 0.3])
    assert r["tau_low"] is None
