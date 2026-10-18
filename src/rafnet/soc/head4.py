"""Phase 6: WavLM-head inference (additive SOC signal).

Loads the Phase-4 head (models/head_phase4.pt, local) plus frozen
microsoft/wavlm-base-plus (local HF cache, offline) and returns P(AI) for the
stacked ensemble's 4th feature. Never blocks analysis: any failure -> None and
the caller falls back to the 3-model ensemble (models/stacking.json).

Training/inference must match: mono -> 16 kHz -> clip 12 s -> zero-mean /
unit-variance (the embedder's normalization; the base model ship exposed no
preprocessor config), layer-wise mean+std, layer-attention head.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

HEAD_DEFAULT = Path(__file__).resolve().parents[3] / "models" / "head_phase4.pt"
ENCODER_NAME = "microsoft/wavlm-base-plus"
MAX_SECONDS = 12.0
TARGET_RATE = 16000


class WavLMHead:
    def __init__(self, path: str | Path = HEAD_DEFAULT, device: str = "cpu") -> None:
        self.path = Path(path)
        self.device = device
        self._model = None
        self._encoder = None
        self._config: dict = {}
        self._band: dict = {}
        self._error: str | None = None

    def load(self) -> None:
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import torch
            from transformers import AutoModel

            if not self.path.is_file():
                raise FileNotFoundError(f"head checkpoint missing: {self.path}")
            ck = torch.load(str(self.path), map_location="cpu", weights_only=False)
            cfg = dict(ck.get("config", {}))
            import torch.nn as nn

            hid = int(cfg.get("hid", 768))
            mid = int(cfg.get("mid", 256))
            dropout = float(cfg.get("dropout", 0.2))

            class LayerAttnHead(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.layer_score = nn.Sequential(
                        nn.Linear(2 * hid, mid), nn.GELU(), nn.Linear(mid, 1))
                    self.mlp = nn.Sequential(
                        nn.Linear(2 * hid, mid), nn.GELU(), nn.Dropout(dropout),
                        nn.Linear(mid, 1))

                def forward(self, x):  # x: [B, L, 2H]
                    a = self.layer_score(x).squeeze(-1)
                    w = torch.softmax(a, dim=1)
                    agg = (w.unsqueeze(-1) * x).sum(dim=1)
                    return self.mlp(agg).squeeze(-1)

            model = LayerAttnHead()
            model.load_state_dict(ck["state_dict"])
            model.eval()
            encoder = AutoModel.from_pretrained(
                str(ck.get("encoder", ENCODER_NAME)), trust_remote_code=False)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad_(False)
            self._model = model
            self._encoder = encoder
            self._config = {"hid": hid, "mid": mid, "dropout": dropout,
                            "encoder": str(ck.get("encoder", ENCODER_NAME))}
            self._band = dict(ck.get("band", {}))
            logger.info("WavLMHead loaded from %s (band=%s)", self.path.name, self._band)
        except Exception as exc:  # additive: never crash startup
            self._error = f"{type(exc).__name__}: {exc}"
            logger.warning("WavLMHead unavailable (non-fatal): %s", self._error)

    def is_loaded(self) -> bool:
        return self._model is not None and self._encoder is not None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def band(self) -> dict:
        return self._band

    def predict_pai(self, samples, sample_rate: int = TARGET_RATE):
        """Return P(AI) in [0,1] or None when unavailable/degenerate."""
        if not self.is_loaded():
            return None
        try:
            import numpy as np
            import torch

            from .preprocess import to_model_input

            arr = to_model_input(samples, sample_rate)
            if arr.size < 1600 or not np.isfinite(arr).all():
                return None
            cap = int(TARGET_RATE * MAX_SECONDS)
            if arr.size > cap:
                arr = arr[:cap]
            norm = (arr - arr.mean()) / (arr.std() + 1e-7)
            inp = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0)
            with torch.inference_mode():
                out = self._encoder(input_values=inp, output_hidden_states=True)
                hs = torch.stack(out.hidden_states, dim=0).squeeze(1)  # L,T,H
                # Match training exactly: the embedder cached fp16 layer
                # mean/std, so round to fp16 before the head (avoids a small
                # train/inference skew from fp32-only embeddings).
                mean = hs.float().mean(dim=1).half().float()            # L,H
                std = hs.float().std(dim=1).half().float()              # L,H
                x = torch.cat([mean, std], dim=-1).unsqueeze(0)        # 1,L,2H
                logit = float(self._model(x).item())
            import math

            return round(1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, logit)))), 4)
        except Exception as exc:
            logger.warning("WavLMHead inference failed (non-fatal): %s", exc)
            return None


_HEAD: WavLMHead | None = None


def load_head(path: str | Path = HEAD_DEFAULT, device: str = "cpu") -> WavLMHead:
    global _HEAD
    if _HEAD is None:
        _HEAD = WavLMHead(path=path, device=device)
        _HEAD.load()
    return _HEAD


def get_head() -> WavLMHead | None:
    return _HEAD if (_HEAD is not None and _HEAD.is_loaded()) else None
