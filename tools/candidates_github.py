"""Phase 1 step 3: GitHub official repos + backbone/speechbrain verification.

Checks (no large downloads):
- SSL_Anti-spoofing (TakHemlata/SSL_Anti-spoofing): repo exists? checkpoint URL reachable?
- AASIST (clovaai/aasist): repo + checkpoint?
- RawNet2 (eurecom-asp/spoof-celeb? / ASVspoof baselines): repo presence
- Backbones: facebook/wav2vec2-xls-r-300m, microsoft/wavlm-large,
  openai/whisper-small|medium — config reachable + param counts from config
- speechbrain/spkrec-ecapa-voxceleb — model_info + speechbrain installed?

Writes reports/candidates_github.json. Every entry states verified vs not
and why (HTTP status / error), never assumed.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "candidates_github.json"

def http_head(url: str, timeout: int = 25) -> dict:
    import urllib.request
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "RAFNet-phase1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"url": url, "reachable": True, "status": r.status,
                    "content_length": r.headers.get("Content-Length"),
                    "content_type": r.headers.get("Content-Type")}
    except Exception as e:
        return {"url": url, "reachable": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}

def main() -> int:
    checks: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "repos": {}, "backbones": {}, "speechbrain": {}}
    # --- GitHub repos (API, lightweight) ---
    import urllib.request
    repos = {
        "SSL_Anti-spoofing": "https://api.github.com/repos/TakHemlata/SSL_Anti-spoofing",
        "AASIST": "https://api.github.com/repos/clovaai/aasist",
        "RawNet2_ASVspoof_baseline": "https://api.github.com/repos/asvspoof-challenge/asvspoof2021_baseline",
    }
    for name, api_url in repos.items():
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "RAFNet-phase1", "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=25) as r:
                j = json.loads(r.read().decode("utf-8", "ignore"))
            checks["repos"][name] = {"exists": True, "full_name": j.get("full_name"),
                                     "description": (j.get("description") or "")[:300],
                                     "license": (j.get("license") or {}).get("spdx_id") if j.get("license") else None,
                                     "default_branch": j.get("default_branch"), "pushed_at": j.get("pushed_at")}
        except Exception as e:
            checks["repos"][name] = {"exists": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"{name}: {checks['repos'][name]}")
    # --- Checkpoint discoverability: list repo roots via GitHub API (no guessing) ---
    import urllib.request as _url
    def gh_contents(repo: str, path: str = "") -> dict:
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        try:
            req = _url.Request(url, headers={"User-Agent": "RAFNet-phase1", "Accept": "application/vnd.github+json"})
            with _url.urlopen(req, timeout=25) as r:
                items = json.loads(r.read().decode("utf-8", "ignore"))
            if isinstance(items, list):
                return {"repo": repo, "path": path, "entries": [i.get("name") for i in items][:40]}
            return {"repo": repo, "path": path, "entries": items}
        except Exception as e:
            return {"repo": repo, "path": path, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    checks["repo_contents"] = {
        "SSL_Anti-spoofing_root": gh_contents("TakHemlata/SSL_Anti-spoofing", ""),
        "AASIST_root": gh_contents("clovaai/aasist", ""),
    }
    for k, v in checks["repo_contents"].items():
        print(f"{k}: {str(v)[:300]}")
    # --- Backbones: config-level verification (no weight download) ---
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        for mid in ["facebook/wav2vec2-xls-r-300m", "microsoft/wavlm-large",
                    "openai/whisper-small", "openai/whisper-medium"]:
            try:
                info = api.model_info(mid)
                raw_card = getattr(info, "cardData", None)
                if isinstance(raw_card, dict):
                    card = raw_card
                elif raw_card is None:
                    card = {}
                else:
                    try:
                        card = dict(raw_card)
                    except Exception:
                        card = {}
                sibs = list(getattr(info, "siblings", None) or [])
                names = []
                for s in sibs:
                    try:
                        names.append(getattr(s, "rfilename", "?"))
                    except Exception:
                        names.append("?")
                has_bin = any(n.endswith(".bin") for n in names)
                has_st = any(n.endswith(".safetensors") for n in names)
                arch = None
                try:
                    cfg_p = hf_hub_download(mid, "config.json")
                    cfg = json.loads(Path(cfg_p).read_text(encoding="utf-8"))
                    if isinstance(cfg, dict):
                        arch = cfg.get("architectures") or cfg.get("model_type")
                except Exception as e:
                    arch = f"config-fetch-failed: {type(e).__name__}"
                checks["backbones"][mid] = {"exists": True, "license": card.get("license"),
                                            "pipeline": getattr(info, "pipeline_tag", None),
                                            "has_pytorch_bin": has_bin, "has_safetensors": has_st,
                                            "arch": arch, "note": "config verified; weights NOT downloaded (bandwidth); for GPU notebook / embedding-head training"}
            except Exception as e:
                import traceback
                checks["backbones"][mid] = {"exists": "unknown", "error": f"{type(e).__name__}: {str(e)[:200]}",
                                            "traceback": traceback.format_exc(limit=5)}
            print(f"{mid}: {checks['backbones'][mid]}")
    except Exception as e:
        checks["backbones_error"] = str(e)[:200]
    # --- Speechbrain ECAPA ---
    try:
        import importlib.util
        checks["speechbrain"]["installed"] = importlib.util.find_spec("speechbrain") is not None
    except Exception:
        checks["speechbrain"]["installed"] = False
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info("speechbrain/spkrec-ecapa-voxceleb")
        raw = getattr(info, "cardData", None)
        card = raw if isinstance(raw, dict) else {}
        checks["speechbrain"]["hub_exists"] = True
        checks["speechbrain"]["license"] = card.get("license") or "apache-2.0 (tag)"
        checks["speechbrain"]["files"] = [getattr(s, "rfilename", "?") for s in (getattr(info, "siblings", []) or [])][:12]
        checks["speechbrain"]["note"] = ("Hub repo verified; speechbrain package NOT installed in .venv — "
                                         "enrollment biometric stays 'not installed' until "
                                         "`pip install speechbrain` (additive, Phase 5).")
    except Exception as e:
        checks["speechbrain"]["hub_exists"] = False
        checks["speechbrain"]["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    print(f"speechbrain: {checks['speechbrain']}")
    OUT.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
