"""Drive the running interface in headless Chrome (DevTools protocol) for visual QA and the screenshots in report/figures.

    python scripts/ui_screenshots.py --url http://127.0.0.1:7861/ --image path/to/fundus.png [--out report/figures] [--chrome "C:/Program Files/Google/Chrome/Application/chrome.exe"]

Needs Chrome/Edge and `pip install websocket-client` (QA tooling only, not an app dependency). Captures: upload state, results state (full page), the
Grad-CAM / lesion tabs, the expanded viewer, a tablet width and a phone width, and prints layout checks (horizontal overflow, nested scroll containers,
clipped text). Exit code 1 if a check fails.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

CHROMES = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"]


class CDP:
    def __init__(self, port: int):
        tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=10).read())
        page = next(t for t in tabs if t["type"] == "page")
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], max_size=None, timeout=120)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m["result"]

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r["result"].get("value")

    def shot(self, path: Path, width: int, height: int = 900, full: bool = True):
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=width < 600)
        time.sleep(0.6)
        if full:
            h = int(self.js("Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)"))
            r = self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True, clip={"x": 0, "y": 0, "width": width, "height": min(h, 6000), "scale": 1})
        else:
            r = self.call("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(r["data"]))
        print("saved", path)


CHECKS = r"""
(() => {
  const out = {overflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth, nested: [], clipped: []};
  const root = document.querySelector('.gradio-container');
  root.querySelectorAll('*').forEach(el => {
    const cs = getComputedStyle(el);
    if ((cs.overflowY === 'auto' || cs.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 2 && el.clientHeight > 0) out.nested.push(el.className.toString().slice(0, 60) || el.tagName);
    if (el.children.length === 0 && el.textContent.trim() && cs.overflow === 'hidden' && cs.textOverflow === 'ellipsis' && el.scrollWidth > el.clientWidth + 1) out.clipped.push(el.textContent.trim().slice(0, 40));
  });
  return out;
})()
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7861/")
    ap.add_argument("--image", default=None, help="fundus photograph to upload for the results state (omit when the app was started with --demo-image)")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--chrome", default=next((c for c in CHROMES if Path(c).is_file()), None))
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--wait", type=float, default=40, help="seconds to wait for the analysis")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="drshot_")
    proc = subprocess.Popen([a.chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={a.port}", "--remote-allow-origins=*", f"--user-data-dir={prof}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failed = []
    try:
        for _ in range(40):
            try:
                c = CDP(a.port)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("could not attach to Chrome")
        c.call("Page.enable")
        c.call("Page.navigate", url=a.url)
        time.sleep(6)
        if a.image:  # upload state first, then results
            c.shot(out / "ui_upload_state.png", 1366, 800, full=True)
            c.call("DOM.enable")
            doc = c.call("DOM.getDocument")
            node = c.call("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#rl-input input[type=file]")["nodeId"]
            c.call("DOM.setFileInputFiles", files=[str(Path(a.image).resolve())], nodeId=node)
            time.sleep(2)
            c.js("[...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Analyse image').click()")
        t0 = time.time()
        while time.time() - t0 < a.wait and not c.js("!!document.querySelector('.rl-viewer')"):
            time.sleep(1)
        time.sleep(1.5)
        if not c.js("!!document.querySelector('.rl-viewer')"):
            failed.append("results did not appear")
        c.shot(out / "ui_results_state.png", 1366, 800, full=True)
        c.shot(out / "ui_results_first_screen.png", 1366, 800, full=False)
        for label, name in (("Grad-CAM", "gradcam"), ("Lesion evidence", "lesions"), ("Calibration", "calibration")):
            c.js(f"[...document.querySelectorAll('#rl-details button[role=tab]')].find(b => b.textContent.trim() === {json.dumps(label)}).click()")
            time.sleep(0.8)
            c.shot(out / f"ui_tab_{name}.png", 1366, 800, full=True)
        c.js("document.querySelector('#rlv-cam').click(); document.querySelector('[data-rl=expand]').click()")
        time.sleep(0.6)
        c.shot(out / "ui_viewer_expanded.png", 1366, 800, full=False)
        c.js("document.querySelector('[data-rl=expand]').click(); document.querySelector('#rlv-orig').click()")
        for w, name in ((1920, "wide"), (820, "tablet"), (390, "phone")):
            c.shot(out / f"ui_results_{name}.png", w, 900, full=True)
        for w in (1920, 1366, 1024, 820, 390):
            c.call("Emulation.setDeviceMetricsOverride", width=w, height=800, deviceScaleFactor=1, mobile=w < 600)
            time.sleep(0.5)
            r = c.js(CHECKS)
            print(f"layout @ {w}px:", r)
            if r["overflowX"] > 2:
                failed.append(f"horizontal overflow {r['overflowX']}px at {w}px")
            if r["nested"]:
                failed.append(f"nested scroll containers at {w}px: {r['nested'][:3]}")
            if r["clipped"]:
                failed.append(f"clipped text at {w}px: {r['clipped'][:3]}")
    finally:
        proc.terminate()
    print("FAILED:" if failed else "all layout checks passed", failed or "")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
