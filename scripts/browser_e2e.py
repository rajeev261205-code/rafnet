"""Browser end-to-end smoke test for the running RAFNet stack.

Drives real headless Chrome (system install, no download) through:
dev-mode login -> Overview dashboard -> Analyze file upload ->
History list -> /history/:id detail -> Models page.

Requires: pip install playwright  (browsers NOT downloaded; uses channel="chrome")
Requires: backend on :8000 and frontend on :3000 (see LOCAL_RUN.md).

Usage:
  python scripts/browser_e2e.py [--base http://127.0.0.1:3000]
      [--wav <path>] [--headed] [--shots <dir>]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import wave
from pathlib import Path


def make_tone(path: Path, seconds: float = 2.0, sr: int = 16000) -> None:
    import math
    import struct

    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(n):
            v = int(9000 * math.sin(2 * math.pi * 440 * i / sr))
            w.writeframes(struct.pack("<h", v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:3000")
    ap.add_argument("--wav", default="")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--shots", default="")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: pip install playwright first (no browser download needed)", file=sys.stderr)
        return 2

    wav = Path(args.wav) if args.wav else Path(tempfile.gettempdir()) / "rafnet_e2e.wav"
    if not wav.is_file():
        make_tone(wav)
    shots = Path(args.shots) if args.shots else Path(tempfile.gettempdir()) / "rafnet_e2e_shots"
    shots.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(name)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=not args.headed)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        base = args.base.rstrip("/")
        page.goto(base + "/login", wait_until="networkidle")
        check("login page loads", "RAFNet" in page.title() or page.locator("text=RAFNet").count() > 0)
        check("dev mode labeled", page.locator("text=DEV MODE").count() > 0)
        page.fill("#login-name", "e2e-operator")
        page.click("text=Enter console")
        page.wait_for_url(base + "/", timeout=15000)
        check("dev login lands on overview", page.url.rstrip("/") == base)
        def seen(text: str, timeout: int = 30000) -> bool:
            try:
                page.wait_for_selector(f"text={text}", timeout=timeout)
                return True
            except Exception:
                return False

        check("overview heading", seen("Overview"))
        page.screenshot(path=str(shots / "01-overview.png"))

        page.goto(base + "/analyze", wait_until="networkidle")
        page.set_input_files("#analyze-file", str(wav))
        page.click("text=Start analysis")
        page.wait_for_selector("text=/REAL|UNCERTAIN|SYNTHETIC/", timeout=120000)
        body = page.inner_text("body")
        pred = next((x for x in ("REAL", "UNCERTAIN", "SYNTHETIC") if x in body), "?")
        check("analyze returns prediction", pred != "?", pred)
        page.screenshot(path=str(shots / "02-analyze.png"))

        page.goto(base + "/history", wait_until="networkidle")
        page.wait_for_selector("table.soc-table tbody tr", timeout=30000)
        first = page.locator("table.soc-table tbody tr td a").first
        href = first.get_attribute("href") or ""
        check("history row links to detail", href.startswith("/history/") and len(href) > 9, href)
        first.click()
        page.wait_for_url(base + "/history/*", timeout=15000)
        check("detail page shows model result", seen("Model result"))
        check("detail page shows verdict", seen("Fused verdict"))
        page.screenshot(path=str(shots / "03-detail.png"))

        page.goto(base + "/models", wait_until="networkidle")
        check("models page shows RAFNet", page.locator("text=RAFNet").count() > 0)
        page.screenshot(path=str(shots / "04-models.png"))

        app_errors = [e for e in errors if "favicon" not in e.lower() and "fonts.g" not in e.lower()]
        check("no app console errors", len(app_errors) == 0, "; ".join(app_errors[:3]))
        browser.close()

    print(f"shots: {shots}")
    if failures:
        print(f"\n{len(failures)} FAILURES: {failures}")
        return 1
    print("\nBROWSER E2E OK — all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
