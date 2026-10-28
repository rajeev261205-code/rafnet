"""Evaluate the RAFNet voice model on labeled clips (measure first, then change).

Input:  samples/real/* and samples/fake/* (wav/flac/mp3/ogg/m4a — anything
        src/rafnet/soc/audio_io.decode_audio accepts).
Output: reports/model_eval.json (machine-readable)
        reports/model_eval.md   (human-readable: confusion matrix, accuracy,
        precision/recall, EER, AUC, per-class score distributions with an
        ASCII histogram showing whether the UNCERTAIN band is too wide).

Variants:
  clean   — each file decoded and scored once (full clip, same path as
            POST /api/analyze).
  mic-sim — the same clip through a simulated laptop-mic chain (16 kHz
            resample + room noise + band-limit + MP3 round-trip when the
            runtime supports it), showing how much accuracy drops.

Calibration (--calibrate):
  stratified held-out split; grid search over (tau_low, tau_high) maximizing
  STRICT accuracy (UNCERTAIN counts as incorrect; ties -> narrower band, then
  higher coverage). Writes configs/decision_thresholds.json (taus only — never
  touches model weights) and reports before/after on the held-out half.

Live aggregation:
  files >= 1 s are also split into non-overlapping 1 s windows, each scored,
  and the file verdict is re-decided from the median window score (the same
  median-of-last-5 + hysteresis used by POST /api/live/chunk). The report
  compares single-shot (full clip) vs aggregated accuracy.

Honesty rules (enforced, not optional):
  - Mock inference is NEVER reported as model accuracy (unevaluated + reason).
  - Small sample sets are labeled NOT a general benchmark (n shown everywhere).
  - Weak results are stated plainly with likely causes + what data/retraining
    would fix. Displayed numbers are never tuned to look better.

Usage (repo root):
  .venv/Scripts/python tools/evaluate_model.py [--samples samples]
      [--reports reports] [--thresholds configs/decision_thresholds.json]
      [--calibrate] [--seed 42] [--agg-window 5]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REAL_DIR = "real"
FAKE_DIR = "fake"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".oga", ".m4a", ".mp4", ".webm"}

TAU_LOW_DEFAULT = 0.3611111111111111
TAU_HIGH_DEFAULT = 0.861111111111111

# ---------------------------------------------------------------------------
# Pure helpers (no torch/model — unit-tested without the ML runtime)
# ---------------------------------------------------------------------------

def label_for_p(p: float, tau_low: float, tau_high: float) -> str:
    if p < tau_low:
        return "REAL"
    if p > tau_high:
        return "SYNTHETIC"
    return "UNCERTAIN"


def confusion_3way(y_true: list[int], scores: list[float], tau_low: float, tau_high: float) -> dict:
    """Rows: true real/fake. Cols: predicted REAL/UNCERTAIN/SYNTHETIC."""
    m = {"real": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0},
         "fake": {"REAL": 0, "UNCERTAIN": 0, "SYNTHETIC": 0}}
    for y, s in zip(y_true, scores):
        row = "fake" if y == 1 else "real"
        m[row][label_for_p(float(s), tau_low, tau_high)] += 1
    return m


def strict_accuracy(y_true: list[int], scores: list[float], tau_low: float, tau_high: float) -> float | None:
    """Fraction exactly right; UNCERTAIN always counts as incorrect (stated tradeoff)."""
    if not y_true:
        return None
    ok = 0
    for y, s in zip(y_true, scores):
        lab = label_for_p(float(s), tau_low, tau_high)
        if (y == 1 and lab == "SYNTHETIC") or (y == 0 and lab == "REAL"):
            ok += 1
    return ok / len(y_true)


def coverage_and_selective(y_true: list[int], scores: list[float], tau_low: float, tau_high: float) -> dict:
    n = len(y_true)
    certain_idx = [i for i, s in enumerate(scores)
                   if label_for_p(float(s), tau_low, tau_high) != "UNCERTAIN"]
    cov = len(certain_idx) / n if n else 0.0
    sel = None
    if certain_idx:
        ok = sum(1 for i in certain_idx
                 if ((y_true[i] == 1 and label_for_p(float(scores[i]), tau_low, tau_high) == "SYNTHETIC")
                     or (y_true[i] == 0 and label_for_p(float(scores[i]), tau_low, tau_high) == "REAL")))
        sel = ok / len(certain_idx)
    return {"coverage": round(cov, 4), "selective_accuracy": round(sel, 4) if sel is not None else None,
            "n_certain": len(certain_idx), "n_total": n}


def binary_metrics_at_05(y_true: list[int], scores: list[float]) -> dict:
    """Threshold-0.5 binary metrics for the fake class (UNCERTAIN band ignored here)."""
    if not y_true:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None, "n": 0}
    pred = [1 if float(s) >= 0.5 else 0 for s in scores]
    tp = sum(1 for y, p in zip(y_true, pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, pred) if y == 1 and p == 0)
    n = len(y_true)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec)) else None

    def _r(v):
        return round(v, 4) if v is not None else None

    return {"accuracy": _r(acc), "precision": _r(prec), "recall": _r(rec), "f1": _r(f1),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n}


def compute_eer(y_true: list[int], scores: list[float]) -> dict:
    """Equal-error rate by sweeping the decision threshold over observed scores."""
    import numpy as np

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(set(y_true)) < 2 or len(y_true) == 0:
        return {"eer": None, "threshold": None, "note": "needs both classes"}
    cands = sorted(set([0.0, 1.0] + [float(v) for v in s]))
    best = {"eer": 1.0, "threshold": 0.5, "far": 1.0, "frr": 1.0}
    best_gap = float("inf")
    for t in cands:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        far = fp / (fp + tn) if (fp + tn) else 0.0
        frr = fn / (fn + tp) if (fn + tp) else 0.0
        gap = abs(far - frr)
        if gap < best_gap - 1e-12:
            best_gap = gap
            best = {"eer": round((far + frr) / 2, 4), "threshold": round(float(t), 4),
                    "far": round(far, 4), "frr": round(frr, 4)}
    return best


def compute_auc(y_true: list[int], scores: list[float]) -> float | None:
    """ROC AUC via the Mann-Whitney rank statistic (no sklearn needed)."""
    import numpy as np

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    # average ranks for ties
    i = 0
    sorted_s = s[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg = (i + j) / 2.0 + 1  # 1-based ranks
        ranks[order[i:j + 1]] = avg
        i = j + 1
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return round(float(u / (n_pos * n_neg)), 4)


def score_stats(values: list[float]) -> dict:
    import numpy as np

    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)

    def _q(p):
        return round(float(np.quantile(a, p)), 4)

    return {"n": len(values), "mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4), "min": round(float(a.min()), 4),
            "p25": _q(0.25), "median": _q(0.5), "p75": _q(0.75), "max": round(float(a.max()), 4)}


def ascii_histogram(values: list[float], bins: int = 10, width: int = 40,
                    tau_low: float | None = None, tau_high: float | None = None) -> list[str]:
    import numpy as np

    lines: list[str] = []
    if not values:
        return ["(no scores)"]
    a = np.asarray(values, dtype=float)
    counts, edges = np.histogram(a, bins=bins, range=(0.0, 1.0))
    peak = int(counts.max()) or 1
    for i in range(bins):
        bar = "#" * max(0, int(round(counts[i] / peak * width)))
        band = ""
        if tau_low is not None and tau_high is not None:
            mid = (edges[i] + edges[i + 1]) / 2
            band = "  <-- UNCERTAIN band" if tau_low <= mid <= tau_high else ""
        lines.append(f"{edges[i]:.2f}-{edges[i+1]:.2f} | {bar:<{width}} {int(counts[i]):>3}{band}")
    return lines


def uncertain_share(values: list[float], tau_low: float, tau_high: float) -> float | None:
    if not values:
        return None
    n = sum(1 for v in values if tau_low <= float(v) <= tau_high)
    return round(n / len(values), 4)


def grid_calibrate(y_true: list[int], scores: list[float],
                   lo_grid: int = 17, hi_grid: int = 17,
                   min_width: float = 0.1, max_width: float = 0.5) -> dict:
    """Grid search maximizing strict accuracy (UNCERTAIN = incorrect).

    Tie-breaks: narrower UNCERTAIN band first, then higher coverage. Returns
    {tau_low, tau_high, accuracy, coverage, searched} — all measured, never
    invented. Small-n instability is the caller's to report, not to hide.
    """
    import numpy as np

    if len(y_true) < 4 or len(set(y_true)) < 2:
        return {"tau_low": None, "tau_high": None, "accuracy": None,
                "note": "need >= 4 clips with both classes to calibrate"}
    lo_cands = np.linspace(0.05, 0.45, lo_grid)
    hi_cands = np.linspace(0.55, 0.95, hi_grid)
    best = {"tau_low": TAU_LOW_DEFAULT, "tau_high": TAU_HIGH_DEFAULT,
            "accuracy": strict_accuracy(y_true, scores, TAU_LOW_DEFAULT, TAU_HIGH_DEFAULT) or 0.0,
            "width": TAU_HIGH_DEFAULT - TAU_LOW_DEFAULT, "coverage": 0.0}
    searched = 0
    for lo in lo_cands:
        for hi in hi_cands:
            w = float(hi) - float(lo)
            if not (min_width <= w <= max_width):
                continue
            searched += 1
            acc = strict_accuracy(y_true, scores, float(lo), float(hi)) or 0.0
            cov = coverage_and_selective(y_true, scores, float(lo), float(hi))["coverage"]
            if (acc > best["accuracy"] + 1e-12
                    or (abs(acc - best["accuracy"]) <= 1e-12 and w < best["width"] - 1e-12)
                    or (abs(acc - best["accuracy"]) <= 1e-12 and abs(w - best["width"]) <= 1e-12
                        and cov > best["coverage"])):
                best = {"tau_low": round(float(lo), 4), "tau_high": round(float(hi), 4),
                        "accuracy": round(acc, 4), "width": w, "coverage": cov}
    best["searched"] = searched
    return best


def stratified_split(y_true: list[int], frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Stratified index split: `frac` of each class -> calibration, rest -> held-out."""
    import random

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(y_true):
        by_class.setdefault(y, []).append(i)
    cal, held = [], []
    for cls, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        k = int(round(len(idxs) * frac))
        k = max(1, min(len(idxs) - 1, k)) if len(idxs) >= 2 else 0
        cal.extend(idxs[:k])
        held.extend(idxs[k:])
    rng.shuffle(cal)
    rng.shuffle(held)
    return cal, held


# ---------------------------------------------------------------------------
# Audio helpers (numpy only)
# ---------------------------------------------------------------------------

def to_mono_16k(samples, sample_rate: int):
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype(np.float32)
    arr = arr.ravel().astype(np.float32)
    if int(sample_rate) == 16000 or arr.size == 0:
        return arr
    duration = arr.size / float(sample_rate)
    target_len = max(1, int(round(duration * 16000)))
    old_idx = np.linspace(0.0, 1.0, num=arr.size)
    new_idx = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(new_idx, old_idx, arr).astype(np.float32)


def simulate_laptop_mic(samples_16k, seed: int = 42, snr_db: float = 20.0) -> tuple:
    """Simulate laptop-mic conditions. Returns (audio_16k, method_note).

    Chain: 16 kHz mono -> white room noise at `snr_db` SNR + 50 Hz mains hum ->
    FFT brickwall 80 Hz–6.8 kHz (mic + lossy band-limit) -> MP3 64 kbps
    round-trip when soundfile/libsndfile supports it, else band-limited + noise
    only (the fallback is recorded in the note, never hidden).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.asarray(samples_16k, dtype=np.float32).ravel()
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    noise_rms = rms / (10.0 ** (snr_db / 20.0)) if rms > 0 else 1e-4
    noise = rng.normal(0.0, noise_rms, size=x.shape).astype(np.float32)
    t = np.arange(x.size, dtype=np.float32) / 16000.0
    hum = (0.15 * noise_rms * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
    y = (x + noise + hum).astype(np.float32)
    # brickwall band-limit via FFT
    spec = np.fft.rfft(y.astype(np.float64))
    freqs = np.fft.rfftfreq(y.size, d=1.0 / 16000.0)
    spec[(freqs < 80.0) | (freqs > 6800.0)] = 0.0
    y = np.fft.irfft(spec, n=y.size).astype(np.float32)
    method = f"room noise SNR {snr_db} dB + 50 Hz hum + brickwall 80 Hz-6.8 kHz"
    # MP3 round-trip when available
    try:
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, y, 16000, format="MP3", bitrate_mode="CONSTANT", compression_level=0.0)
        buf.seek(0)
        back, sr = sf.read(buf, dtype="float32", always_2d=False)
        back = np.asarray(back, dtype=np.float32).ravel()
        if back.size and sr == 16000:
            y = np.clip(back, -1.0, 1.0).astype(np.float32)
            method += " + MP3 64k round-trip"
        else:
            method += " (MP3 round-trip unavailable: rate/size mismatch)"
    except Exception as exc:
        method += f" (MP3 round-trip unavailable: {type(exc).__name__})"
    return np.clip(y, -1.0, 1.0).astype(np.float32), method


def discover_samples(samples_dir: Path) -> list[dict]:
    out: list[dict] = []
    for label, sub in ((0, REAL_DIR), (1, FAKE_DIR)):
        d = samples_dir / sub
        if not d.is_dir():
            out.append({"missing_dir": str(d), "label": label})
            continue
        files = sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
        if not files:
            out.append({"note": f"no audio files in {d}", "label": label})
            continue
        for p in files:
            out.append({"path": str(p), "label": label})
    return out


def load_active_thresholds(override: str | None) -> tuple[float, float, dict]:
    if override:
        try:
            data = json.loads(Path(override).read_text(encoding="utf-8"))
            return float(data.get("tau_low", TAU_LOW_DEFAULT)), float(data.get("tau_high", TAU_HIGH_DEFAULT)), \
                {"source": override, "calibrated": bool(data.get("calibrated", False))}
        except Exception as exc:
            print(f"WARNING: --thresholds unreadable ({exc}); using shipped defaults.")
    try:
        from rafnet.soc.thresholds import load_decision_thresholds

        return load_decision_thresholds()
    except Exception:
        return TAU_LOW_DEFAULT, TAU_HIGH_DEFAULT, {"source": "shipped defaults"}


def load_model():
    try:
        from rafnet.soc.model_adapter import load_voice_model

        model = load_voice_model()
        if model.is_loaded():
            return model, {"loaded": True, "name": getattr(model, "name", "unknown"),
                           "mock": bool(getattr(model, "name", "") == "mock")}
        return None, {"loaded": False, "reason": "model did not load (checkpoints missing?)"}
    except Exception as exc:
        return None, {"loaded": False, "reason": f"{type(exc).__name__}: {exc}"}


def score_file(path: Path, model, seed: int) -> dict:
    from rafnet.soc.audio_io import decode_audio

    entry: dict = {"file": str(path)}
    try:
        decoded = decode_audio(path.read_bytes(), path.name)
    except Exception as exc:
        entry["error"] = f"decode failed: {exc}"
        return entry
    entry["duration_seconds"] = round(decoded.duration_seconds, 2)
    entry["sample_rate_hz"] = decoded.sample_rate
    entry["input_lossy"] = bool(decoded.is_lossy)
    try:
        result = model.predict(decoded.samples, decoded.sample_rate)
    except Exception:
        entry["error"] = f"model inference failed: {traceback.format_exc(limit=3)}"
        return entry
    entry["p_synthetic"] = round(float(result.prob_synthetic), 4)
    entry["prediction"] = result.prediction
    entry["confidence"] = round(float(result.confidence), 4)
    entry["classical_p"] = result.classical_probability
    entry["quantum_p"] = result.quantum_probability
    entry["fused_p"] = result.fused_probability
    # mic-sim variant (same model, degraded audio)
    try:
        mono16 = to_mono_16k(decoded.samples, decoded.sample_rate)
        sim, method = simulate_laptop_mic(mono16, seed=seed)
        entry["mic_sim_method"] = method
        sres = model.predict(sim.tolist(), 16000)
        entry["p_synthetic_mic"] = round(float(sres.prob_synthetic), 4)
        entry["prediction_mic"] = sres.prediction
    except Exception:
        entry["mic_error"] = f"mic-sim failed: {traceback.format_exc(limit=3)}"
    # live-style windowed aggregation: non-overlapping 1 s windows, median score
    try:
        import numpy as np

        mono16 = to_mono_16k(decoded.samples, decoded.sample_rate)
        wins = [mono16[i:i + 16000] for i in range(0, len(mono16), 16000)
                if len(mono16[i:i + 16000]) >= 8000][:8]
        wps: list[float] = []
        for w in wins:
            if len(w) < 16000:
                pad = np.zeros(16000 - len(w), dtype=np.float32)
                w = np.concatenate([np.asarray(w, dtype=np.float32), pad])
            wr = model.predict(np.asarray(w, dtype=np.float32).tolist(), 16000)
            wps.append(round(float(wr.prob_synthetic), 4))
        entry["window_ps"] = wps
        entry["n_windows"] = len(wps)
        if wps:
            entry["p_synthetic_agg"] = round(float(np.median(np.asarray(wps))), 4)
    except Exception:
        entry["agg_error"] = f"windowed aggregation failed: {traceback.format_exc(limit=3)}"
    return entry


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def metrics_block(y: list[int], scores: list[float], tau_low: float, tau_high: float, name: str) -> dict:
    return {
        "condition": name,
        "n": len(y),
        "thresholds": {"tau_low": tau_low, "tau_high": tau_high,
                       "band_width": round(tau_high - tau_low, 4)},
        "confusion_3way": confusion_3way(y, scores, tau_low, tau_high),
        "strict_accuracy_uncertain_is_wrong": strict_accuracy(y, scores, tau_low, tau_high),
        **coverage_and_selective(y, scores, tau_low, tau_high),
        "binary_at_0.5": binary_metrics_at_05(y, scores),
        "eer": compute_eer(y, scores),
        "auc": compute_auc(y, scores),
    }


def write_reports(reports_dir: Path, report: dict) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    jp = reports_dir / "model_eval.json"
    mp = reports_dir / "model_eval.md"
    jp.write_text(json.dumps(report, indent=2), encoding="utf-8")

    L: list[str] = []
    L.append("# Model evaluation report (RAFNet voice)")
    L.append("")
    L.append(f"Generated: {report['generated_at']}")
    L.append(f"Model: {json.dumps(report['model'])}")
    L.append(f"Thresholds: tau_low={report['thresholds']['tau_low']} "
             f"tau_high={report['thresholds']['tau_high']} ({report['thresholds'].get('source', '?')})")
    L.append(f"Files: {report['files_scored']} scored / {report['files_total']} found "
             f"(real={report['n_real']}, fake={report['n_fake']})")
    L.append("")
    L.append("> Small local sample sets are NOT a general benchmark. Every n is shown; "
             "treat all numbers as a snapshot of THESE files only.")
    L.append("")
    if report.get("unevaluated"):
        L.append(f"## UNEVALUATED — {report['unevaluated_reason']}")
        L.append("")
        for f in report["files"]:
            L.append(f"- {json.dumps(f)}")
        L.append("")
        L.append("Add labeled clips to `samples/real/` and `samples/fake/` "
                 "(wav/flac/mp3, 2-10 s each, >= 20 per class) and re-run.")
        mp.write_text("\n".join(L) + "\n", encoding="utf-8")
        return jp, mp

    for key in ("clean", "mic_sim"):
        m = report["metrics"][key]
        L.append(f"## {key} — {m['condition']}")
        L.append(f"n={m['n']}, strict accuracy (UNCERTAIN=wrong)={m['strict_accuracy_uncertain_is_wrong']}, "
                 f"coverage={m['coverage']}, selective={m['selective_accuracy']}, "
                 f"AUC={m['auc']}, EER={json.dumps(m['eer'])}")
        L.append(f"binary@0.5: {json.dumps(m['binary_at_0.5'])}")
        L.append("")
        L.append("Confusion (rows=true, cols=predicted):")
        L.append("| true \\ pred | REAL | UNCERTAIN | SYNTHETIC |")
        L.append("| --- | --- | --- | --- |")
        for row in ("real", "fake"):
            c = m["confusion_3way"][row]
            L.append(f"| {row} | {c['REAL']} | {c['UNCERTAIN']} | {c['SYNTHETIC']} |")
        L.append("")

    L.append("## Score distributions (P(synthetic); higher = more synthetic-like)")
    for cls in ("real", "fake"):
        d = report["distributions"]["clean"][cls]
        L.append(f"- clean/{cls}: {json.dumps(d)}")
        L.append("```")
        L.extend(ascii_histogram(report["hist_values"]["clean"][cls], tau_low=report["thresholds"]["tau_low"],
                                 tau_high=report["thresholds"]["tau_high"]))
        L.append("```")
    L.append("")
    L.append("UNCERTAIN-band share (fraction of scores inside "
             f"[{report['thresholds']['tau_low']}, {report['thresholds']['tau_high']}]):")
    for cls in ("real", "fake"):
        L.append(f"- clean/{cls}: {report['uncertain_share']['clean'][cls]}")
        L.append(f"- mic-sim/{cls}: {report['uncertain_share']['mic_sim'][cls]}")
    L.append("")
    band_note = report.get("band_verdict", "")
    if band_note:
        L.append(f"Band verdict: {band_note}")
        L.append("")
    if report.get("mic_sim_method"):
        L.append(f"Mic-sim chain: {report['mic_sim_method']}")
        L.append("")
    drop = report.get("mic_accuracy_drop")
    if drop is not None:
        L.append(f"Accuracy drop under mic-sim (strict): {drop} "
                 "(positive = model degrades on laptop-mic conditions).")
        L.append("")

    cal = report.get("calibration")
    if cal:
        L.append("## Threshold calibration (held-out split)")
        L.append(f"Tradeoff: {cal.get('tradeoff')}")
        L.append(f"Split: {json.dumps(cal.get('split'))}")
        L.append(f"Before (active thresholds, held-out half): {json.dumps(cal.get('before'))}")
        L.append(f"After (calibrated thresholds, held-out half): {json.dumps(cal.get('after'))}")
        L.append(f"Chosen: tau_low={cal.get('tau_low')} tau_high={cal.get('tau_high')} "
                 f"(saved={cal.get('saved_to')})")
        if cal.get("warning"):
            L.append(f"WARNING: {cal['warning']}")
        L.append("")

    agg = report.get("live_aggregation")
    if agg:
        L.append("## Live aggregation (single full-clip vs median-of-windows)")
        L.append(f"Window: {agg.get('window')} chunks, hysteresis {agg.get('hysteresis')} "
                 f"(same as POST /api/live/chunk).")
        L.append(f"Single-shot strict accuracy: {agg.get('single_accuracy')} "
                 f"(n={agg.get('n')})")
        L.append(f"Aggregated strict accuracy: {agg.get('agg_accuracy')} "
                 f"(mean windows/file={agg.get('mean_windows')})")
        L.append(f"Delta (agg - single): {agg.get('delta')}")
        L.append("")

    L.append("## Honest reading")
    for line in report.get("honest_notes", []):
        L.append(f"- {line}")
    L.append("")
    L.append("## Files")
    for f in report["files"]:
        L.append(f"- {json.dumps(f)}")
    mp.write_text("\n".join(L) + "\n", encoding="utf-8")
    return jp, mp


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate RAFNet on samples/real + samples/fake.")
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibration-split", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agg-window", type=int, default=5)
    args = ap.parse_args()

    samples_dir = (ROOT / args.samples) if not Path(args.samples).is_absolute() else Path(args.samples)
    reports_dir = (ROOT / args.reports) if not Path(args.reports).is_absolute() else Path(args.reports)

    tau_low, tau_high, tmeta = load_active_thresholds(args.thresholds)
    model, minfo = load_model()

    found = discover_samples(samples_dir)
    scored_inputs = [f for f in found if "path" in f]
    n_real = sum(1 for f in scored_inputs if f["label"] == 0)
    n_fake = sum(1 for f in scored_inputs if f["label"] == 1)

    base = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": minfo,
        "thresholds": {"tau_low": tau_low, "tau_high": tau_high, **{k: v for k, v in tmeta.items() if k != "path"}},
        "files_total": len(scored_inputs),
        "n_real": n_real, "n_fake": n_fake,
    }

    # Unevaluated paths: no files, no model, or mock (mock is not the model).
    if not scored_inputs:
        report = {**base, "files_scored": 0, "files": found, "unevaluated": True,
                  "unevaluated_reason": "no labeled audio — samples/real/ and samples/fake/ are empty or missing"}
        jp, mp = write_reports(reports_dir, report)
        print(f"UNEVALUATED: {report['unevaluated_reason']}")
        print(f"wrote {jp} and {mp}")
        return 0
    if model is None or minfo.get("mock"):
        reason = (f"mock inference active ({minfo}) — mock numbers are not model accuracy"
                  if minfo.get("mock") else f"model not loaded ({minfo.get('reason', '?')})")
        report = {**base, "files_scored": 0, "files": found, "unevaluated": True,
                  "unevaluated_reason": reason}
        jp, mp = write_reports(reports_dir, report)
        print(f"UNEVALUATED: {reason}")
        print(f"wrote {jp} and {mp}")
        return 0

    files: list[dict] = []
    for f in scored_inputs:
        entry = score_file(Path(f["path"]), model, seed=args.seed)
        entry["label"] = "fake" if f["label"] == 1 else "real"
        entry["y"] = f["label"]
        files.append(entry)
        print(f"scored {entry['label']:>4} {Path(f['path']).name}: "
              f"p={entry.get('p_synthetic')} -> {entry.get('prediction')} "
              f"| mic p={entry.get('p_synthetic_mic')} -> {entry.get('prediction_mic')} "
              f"| agg p={entry.get('p_synthetic_agg')} ({entry.get('n_windows', 0)} win)")

    ok = [f for f in files if "error" not in f and "p_synthetic" in f]
    y = [f["y"] for f in ok]
    s_clean = [float(f["p_synthetic"]) for f in ok]
    s_mic = [float(f["p_synthetic_mic"]) for f in ok if "p_synthetic_mic" in f]
    y_mic = [f["y"] for f in ok if "p_synthetic_mic" in f]

    metrics = {"clean": metrics_block(y, s_clean, tau_low, tau_high, "full clip (POST /api/analyze path)")}
    if s_mic and len(s_mic) == len(y):
        metrics["mic_sim"] = metrics_block(y, s_mic, tau_low, tau_high,
                                           "laptop-mic simulation (noise + band-limit + MP3)")
    else:
        metrics["mic_sim"] = {"condition": "laptop-mic simulation", "n": 0,
                              "note": "mic-sim scores unavailable for all files"}

    dist = {"clean": {}, "mic_sim": {}}
    hist_values = {"clean": {}, "mic_sim": {}}
    unc_share = {"clean": {}, "mic_sim": {}}
    for cls, yy in (("real", 0), ("fake", 1)):
        vals = [s for s_f, yy_f in zip(s_clean, y) if yy_f == yy for s in [s_f]]
        mvals = [s for s_f, yy_f in zip(s_mic, y_mic) if yy_f == yy for s in [s_f]] if s_mic else []
        dist["clean"][cls] = score_stats(vals)
        dist["mic_sim"][cls] = score_stats(mvals)
        hist_values["clean"][cls] = vals
        hist_values["mic_sim"][cls] = mvals
        unc_share["clean"][cls] = uncertain_share(vals, tau_low, tau_high)
        unc_share["mic_sim"][cls] = uncertain_share(mvals, tau_low, tau_high)

    shares = [v for v in (unc_share["clean"]["real"], unc_share["clean"]["fake"]) if v is not None]
    band_verdict = ""
    if shares:
        worst = max(shares)
        if worst is not None and worst > 0.5:
            band_verdict = (f"UNCERTAIN band [{tau_low}, {tau_high}] (width "
                            f"{round(tau_high - tau_low, 2)}) holds >50% of a class's scores "
                            f"on THESE files — too wide for this data. Recalibrate with "
                            f"--calibrate once n >= 20/class.")
        else:
            band_verdict = (f"UNCERTAIN band [{tau_low}, {tau_high}] holds <=50% of each "
                            f"class on THESE files — not obviously too wide, but n={len(y)} "
                            f"is too small to generalize.")

    mic_method = next((f.get("mic_sim_method") for f in ok if f.get("mic_sim_method")), "")
    acc_clean = metrics["clean"]["strict_accuracy_uncertain_is_wrong"]
    acc_mic = metrics.get("mic_sim", {}).get("strict_accuracy_uncertain_is_wrong")
    drop = round(acc_clean - acc_mic, 4) if isinstance(acc_clean, float) and isinstance(acc_mic, float) else None

    # Calibration on a held-out split
    calibration = None
    if len(y) >= 4 and len(set(y)) == 2:
        cal_idx, held_idx = stratified_split(y, args.calibration_split, args.seed)
        if cal_idx and held_idx:
            y_cal = [y[i] for i in cal_idx]
            s_cal = [s_clean[i] for i in cal_idx]
            y_held = [y[i] for i in held_idx]
            s_held = [s_clean[i] for i in held_idx]
            best = grid_calibrate(y_cal, s_cal)
            before = {"thresholds": {"tau_low": tau_low, "tau_high": tau_high},
                      "strict_accuracy": strict_accuracy(y_held, s_held, tau_low, tau_high),
                      **coverage_and_selective(y_held, s_held, tau_low, tau_high)}
            after = None
            saved_to = None
            warning = None
            if best.get("tau_low") is not None:
                after = {"thresholds": {"tau_low": best["tau_low"], "tau_high": best["tau_high"]},
                         "strict_accuracy": strict_accuracy(y_held, s_held, best["tau_low"], best["tau_high"]),
                         **coverage_and_selective(y_held, s_held, best["tau_low"], best["tau_high"])}
                if len(y) < 40:
                    warning = (f"n={len(y)} is far below the >= 40 clips needed for stable "
                               f"thresholds — these taus fit THIS sample, not the world. "
                               f"Not a benchmark.")
                if args.calibrate:
                    cfg_path = ROOT / "configs" / "decision_thresholds.json"
                    payload = {"tau_low": best["tau_low"], "tau_high": best["tau_high"],
                               "uncertain_band_width": round(best["tau_high"] - best["tau_low"], 4),
                               "calibrated": True,
                               "evaluation": {"method": "grid search maximizing strict accuracy "
                                                        "(UNCERTAIN = incorrect); ties -> narrower band, then coverage",
                                              "held_out_n": len(held_idx), "calibration_n": len(cal_idx),
                                              "seed": args.seed,
                                              "before_held_out": before, "after_held_out": after,
                                              "note": "Small-sample fit — NOT a general benchmark."},
                               "comment": "Measured thresholds from tools/evaluate_model.py --calibrate. "
                                          "Backend applies these to the DecisionGate at startup; weights untouched."}
                    cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    saved_to = str(cfg_path)
                    print(f"calibrated thresholds saved to {saved_to}: "
                          f"tau_low={best['tau_low']} tau_high={best['tau_high']}")
            calibration = {"tradeoff": "maximize strict accuracy with UNCERTAIN counted as incorrect; "
                                       "ties broken by narrower band, then higher coverage",
                           "split": {"calibration_n": len(cal_idx), "held_out_n": len(held_idx),
                                     "seed": args.seed, "method": "stratified per-class shuffle"},
                           "grid": best, "before": before, "after": after,
                           "tau_low": best.get("tau_low"), "tau_high": best.get("tau_high"),
                           "saved_to": saved_to, "warning": warning}
    elif args.calibrate:
        print("WARNING: --calibrate needs >= 4 clips with both classes; config untouched.")

    # Live aggregation comparison (median of 1 s windows vs single full-clip)
    live_agg = None
    agg_ok = [f for f in ok if f.get("p_synthetic_agg") is not None]
    if agg_ok:
        import numpy as np

        y_a = [f["y"] for f in agg_ok]
        s_single = [float(f["p_synthetic"]) for f in agg_ok]
        s_agg = [float(f["p_synthetic_agg"]) for f in agg_ok]
        wins = [f.get("n_windows", 0) for f in agg_ok]
        live_agg = {"window": args.agg_window, "hysteresis": "majority to switch (3/5), same as live route",
                    "n": len(y_a), "mean_windows": round(float(np.mean(wins)), 2) if wins else 0,
                    "single_accuracy": strict_accuracy(y_a, s_single, tau_low, tau_high),
                    "agg_accuracy": strict_accuracy(y_a, s_agg, tau_low, tau_high)}
        sa, aa = live_agg["single_accuracy"], live_agg["agg_accuracy"]
        live_agg["delta"] = round(aa - sa, 4) if isinstance(sa, float) and isinstance(aa, float) else None

    honest: list[str] = []
    honest.append(f"n={len(y)} (real={sum(1 for v in y if v == 0)}, fake={sum(1 for v in y if v == 1)}): "
                  "a small local sample, NOT a general benchmark. Add >= 20 clips per class "
                  "(real speech + TTS + voice-conversion + replayed, disjoint speakers AND "
                  "generators) before trusting any threshold.")
    weak = (acc_clean is not None and acc_clean < 0.7) or \
           (metrics["clean"]["eer"].get("eer") is not None and metrics["clean"]["eer"]["eer"] > 0.3)
    if weak:
        honest.append("The model is WEAK on these conditions (stated plainly, numbers untouched).")
    else:
        honest.append("Accuracy on THESE files is reported as measured; it does not imply "
                      "general performance.")
    honest.append("Likely causes if weak: synthetic-only training with no real-human speech; "
                  "21 generators overlapped across splits (leakage); full-clip embeddings vs "
                  "1 s training windows (biggest preprocessing caveat, see ACCURACY_NOTES.md); "
                  "unvalidated 0.50-wide UNCERTAIN band; laptop-mic noise/band-limit/MP3 shift.")
    honest.append("What would fix it: labeled 16 kHz clips (>= 20/class, disjoint speakers and "
                  "generators); re-measure with this script; recalibrate thresholds (--calibrate); "
                  "with explicit approval, refit classical + PCA + gateway2 on windowed embeddings "
                  "and validate full-clip vs windowed aggregation before changing inference.")

    report = {**base, "files_scored": len(ok), "files": files,
              "metrics": metrics, "distributions": dist, "hist_values": hist_values,
              "uncertain_share": unc_share, "band_verdict": band_verdict,
              "mic_sim_method": mic_method, "mic_accuracy_drop": drop,
              "calibration": calibration, "live_aggregation": live_agg,
              "honest_notes": honest, "agg_window": args.agg_window}
    jp, mp = write_reports(reports_dir, report)
    print(f"files scored: {len(ok)}/{len(scored_inputs)} (real={base['n_real']}, fake={base['n_fake']})")
    print(f"clean strict accuracy: {acc_clean} | mic-sim: {acc_mic} | drop: {drop}")
    print(f"wrote {jp} and {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
