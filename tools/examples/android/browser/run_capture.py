#!/usr/bin/env python3
"""Build, run and capture one Android browser example after its result arrives."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from browser_runner import BrowserServer, RunState, demo_page
from build_bundle import DEMOS


def main() -> None:
    """Keep device use serial and preserve failures alongside successful runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=DEMOS)
    parser.add_argument("--backend", choices=("webgl2", "webgpu"), default="webgl2")
    parser.add_argument("--serial", default="emulator-5554")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    run_id = f"{args.demo}-{time.time_ns()}"
    workspace = Path(__file__).resolve().parents[4]
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    bundle = evidence / "bundles" / args.demo
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("build_bundle.py")), args.demo,
         "--backend", args.backend, "--out", str(bundle)],
        cwd=workspace, check=True,
    )
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=workspace, text=True,
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=workspace, text=True,
    ).splitlines()
    metadata = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip(),
        "backend": args.backend, "demo": args.demo, "serial": args.serial,
        "runId": run_id,
        "changed_file_sha256": {
            name: hashlib.sha256((workspace / name).read_bytes()).hexdigest()
            for name in sorted(set(changed + untracked)) if (workspace / name).is_file()
        },
        "bundle_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(bundle.iterdir()) if path.is_file()
        },
        "capture_semantics": "Two display samples after the awaited three-frame export and page report.",
    }
    (evidence / f"{args.demo}_candidate.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    state = RunState(evidence / f"{args.demo}_result.json")
    server = BrowserServer(
        ("127.0.0.1", args.port), state,
        demo_page(args.backend, args.demo, DEMOS[args.demo][args.backend]), bundle,
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def adb(*command: str) -> bytes:
        return subprocess.check_output(["adb", "-s", args.serial, *command])

    def screenshot(suffix: str) -> None:
        payload = adb("exec-out", "screencap", "-p")
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("ADB did not return a PNG screenshot")
        (evidence / f"{args.demo}{suffix}.png").write_bytes(payload)

    try:
        adb("reverse", f"tcp:{args.port}", f"tcp:{args.port}")
        adb("shell", "am", "force-stop", "com.android.browser")
        # A unique URL also avoids older browser versions reusing a cached page.
        url = f"http://localhost:{args.port}/?run={run_id}"
        adb("shell", "am", "start", "-a", "android.intent.action.VIEW", "-n",
            "com.android.browser/.BrowserActivity", "-d", url)
        deadline = time.monotonic() + args.timeout
        report = None
        while time.monotonic() < deadline:
            reports = state.snapshot()["reports"]
            matching = [item for item in reports if item.get("runId") == run_id]
            if matching:
                report = matching[-1]
                break
            time.sleep(0.25)
        snapshot = state.snapshot()
        (evidence / f"{args.demo}_result.json").write_text(
            json.dumps(snapshot, indent=2) + "\n", encoding="utf-8",
        )
        if report is None or report.get("state") != "success":
            screenshot("")
            raise RuntimeError(f"Browser did not complete {args.demo}: {snapshot}")
        if report.get("demo") != args.demo or report.get("backend") != args.backend:
            raise RuntimeError(f"Unexpected browser report: {report}")
        time.sleep(1)
        screenshot("")
        time.sleep(0.7)
        screenshot("_sample2")
        print(f"PASS {args.backend} {args.demo}: {evidence}", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        # Keep unrelated late reports as diagnostics while identifying the run
        # whose completion was actually accepted by this invocation.
        snapshot = state.snapshot()
        matching = [item for item in snapshot["reports"] if item.get("runId") == run_id]
        selected = matching[-1] if matching else None
        snapshot["selectedRunId"] = run_id
        snapshot["selectedReport"] = selected
        snapshot["state"] = selected.get("state", "waiting") if selected else "waiting"
        (evidence / f"{args.demo}_result.json").write_text(
            json.dumps(snapshot, indent=2) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
