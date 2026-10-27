"""Shared helpers for the Kaggle workflow scripts. Never prints credentials."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; Kaggle logs contain Unicode
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "artifacts" / "kaggle_state.json"
KERNEL_SLUG = "dr-screening-effnet-b0-aptos"
KERNEL_TITLE = "DR Screening EffNet B0 APTOS"  # slug(title) must equal KERNEL_SLUG
COMPETITION = "aptos2019-blindness-detection"
BUILD_DIR = ROOT / "notebooks" / "kaggle" / "build"
# Large downloads (checkpoints with optimizer state, logs) go OUTSIDE the OneDrive-synced project folder.
CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "drscreen" / "kaggle_output"


def kaggle_cmd() -> list[str]:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    local = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if local.exists():
        return [str(local)]
    return [sys.executable, "-m", "kaggle"]


def run_kaggle(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(kaggle_cmd() + args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)


def credentials_present() -> dict:
    """Report WHERE credentials were found (never their values)."""
    home = Path.home() / ".kaggle"
    found = {
        "env KAGGLE_USERNAME+KAGGLE_KEY": bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")),
        "env KAGGLE_API_TOKEN": bool(os.environ.get("KAGGLE_API_TOKEN")),
        "~/.kaggle/kaggle.json": (home / "kaggle.json").is_file(),
        "~/.kaggle/access_token": (home / "access_token").is_file(),
    }
    return {"any": any(found.values()), "sources": found}


def get_username() -> str | None:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if kj.is_file():
        try:
            return json.loads(kj.read_text()).get("username")
        except Exception:
            return None
    try:  # newer CLIs may store the username in their config
        out = run_kaggle(["config", "view"], timeout=60)
        for line in (out.stdout or "").splitlines():
            if "username" in line.lower():
                return line.split(":", 1)[-1].strip() or None
    except Exception:
        pass
    return None


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.is_file() else {}


def save_state(**kw) -> dict:
    st = {**load_state(), **kw}
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=2))
    return st


def kernel_ref() -> str | None:
    st = load_state()
    if st.get("kernel_ref"):
        return st["kernel_ref"]
    u = get_username()
    return f"{u}/{KERNEL_SLUG}" if u else None


def redact(text: str) -> str:
    """Strip anything that looks like a key/token from CLI output before it is shown or logged."""
    import re

    text = re.sub(r"(?i)(key|token|password|secret)([\"'\s:=]+)[A-Za-z0-9_\-\.]{16,}", r"\1\2[REDACTED]", text)
    return re.sub(r"\b[a-f0-9]{32,}\b", "[REDACTED]", text)
