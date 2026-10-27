"""Check (or bounded-poll) the remote Kaggle kernel, and fetch its logs.

    python scripts/kaggle_status.py                      # one status check
    python scripts/kaggle_status.py --watch --max-minutes 60 --interval 120
    python scripts/kaggle_status.py --logs               # print the tail of the run log (kernel output must exist)
"""
from __future__ import annotations

import argparse
import json

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kaggle_common import kernel_ref, load_state, redact, run_kaggle, save_state  # noqa: E402

TERMINAL = ("complete", "error", "cancel", "failed", "auth_expired")


def parse_status(text: str) -> str:
    t = text.lower()
    for key in ("complete", "error", "cancel", "running", "queued"):
        if key in t:
            return key
    return "unknown"


def check_once(ref: str) -> tuple[str, str]:
    r = run_kaggle(["kernels", "status", ref], timeout=120)
    out = redact((r.stdout + r.stderr).strip())
    low = out.lower()
    if "authentication required" in low or "permission" in low and "denied" in low:
        # OAuth tokens expire after a few hours: the remote job is NOT affected, but status/logs/output cannot be read until you sign in again
        st, out = "auth_expired", "Kaggle login expired. Run `kaggle auth login --force` and approve in the browser. (Remote jobs keep running.)"
    else:
        st = parse_status(out) if r.returncode == 0 else "unreachable"
    save_state(last_status=st, last_checked=time.strftime("%Y-%m-%d %H:%M:%S"))
    return st, out


def fetch_logs(ref: str, tail_chars: int = 6000) -> str:
    """Execution log of the latest run (works while it is running and after it finishes)."""
    r = run_kaggle(["kernels", "logs", ref], timeout=300)
    text = (r.stdout or "") + (r.stderr or "")
    try:  # Kaggle returns a JSON list of {"stream_name","time","data"} records; flatten to plain text
        text = "".join(rec.get("data", "") for rec in json.loads(text))
    except Exception:
        pass
    dest = Path(__file__).resolve().parents[1] / "artifacts" / "kaggle_run.log"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(redact(text), encoding="utf-8")
    return redact(text[-tail_chars:]) if text.strip() else "(no log text returned yet)"


def diagnose(log_text: str) -> list[str]:
    hints = []
    t = log_text.lower()
    rules = [
        ("no aptos data found", "Competition data not attached: accept the APTOS rules on kaggle.com, and keep competition_sources in kernel-metadata.json."),
        ("cannot download imagenet weights", "Enable Internet in the kernel (requires phone verification) or attach EfficientNet-B0 weights."),
        ("cuda out of memory", "Reduce batch_size or img_size in notebooks/kaggle config (CONFIGS in scripts/kaggle_submit.py)."),
        ("no module named", "A dependency is missing on Kaggle; add a pip install line at the top of train_and_evaluate.py."),
        ("non-finite training loss", "Training diverged: lower lr, or check for corrupt inputs."),
        ("timeout", "Network timeout; re-run, or attach weights to avoid downloads."),
    ]
    for needle, msg in rules:
        if needle in t:
            hints.append(msg)
    return hints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("--mode", default=None, help="smoke | full | lesion-smoke | lesion (default: the most recently pushed kernel)")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=30, help="bounded polling: give up after this long")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--logs", action="store_true")
    a = ap.parse_args()
    ref = a.ref or (load_state().get("kernels", {}).get(a.mode) if a.mode else None) or kernel_ref()
    if not ref:
        print("No kernel reference known; run `python scripts/kaggle_submit.py push` first (or pass --ref user/slug).")
        sys.exit(3)
    deadline = time.time() + a.max_minutes * 60
    while True:
        st, out = check_once(ref)
        print(time.strftime("%H:%M:%S"), ref, "->", st, "|", out.replace("\n", " ")[:200])
        if not a.watch or st in TERMINAL or st == "unreachable":
            break
        if time.time() + a.interval > deadline:
            print(f"Stopped polling after {a.max_minutes:g} min (job may still be running remotely). Re-run to continue.")
            break
        time.sleep(a.interval)
    if a.logs or st in ("error", "cancel"):
        text = fetch_logs(ref)
        print("---- log tail ----\n" + text)
        for h in diagnose(text):
            print("HINT:", h)
    sys.exit(0 if st in ("complete", "running", "queued") else 1)


if __name__ == "__main__":
    main()
