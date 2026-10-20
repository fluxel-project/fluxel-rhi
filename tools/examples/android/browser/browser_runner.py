#!/usr/bin/env python3
"""Serve one wasm example or a capability probe to an Android browser.

The server deliberately uses only Python's standard library.  Android Browser
reports results back to the same origin, so the host receives structured
capability and completion evidence without relying on pixels or browser
automation support.  Use ``adb reverse`` and navigate Android Browser to
``http://localhost:<port>/``: localhost is a trustworthy origin in browsers
that enforce the secure-context requirement for WebGPU.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


MAX_REPORT_BYTES = 64 * 1024


class RunState:
    """Owns the latest browser report and optional durable evidence file."""

    def __init__(self, result_path: Path | None) -> None:
        self._lock = Lock()
        self._result_path = result_path
        self._report: dict[str, Any] = {
            "state": "waiting",
            "reports": [],
        }

    def add_report(self, report: dict[str, Any]) -> None:
        """Record one bounded JSON message supplied by the served page."""
        with self._lock:
            reports = self._report["reports"]
            assert isinstance(reports, list)
            reports.append(report)
            self._report["state"] = report.get("state", "reported")
            if self._result_path:
                self._result_path.parent.mkdir(parents=True, exist_ok=True)
                self._result_path.write_text(
                    json.dumps(self._report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe copy for the polling endpoint."""
        with self._lock:
            return json.loads(json.dumps(self._report))


def probe_page() -> str:
    """Return the standalone capability probe used before running a demo."""
    return """<!doctype html><meta charset=\"utf-8\"><title>Fluxel Android probe</title>
<pre id=\"result\">probing</pre><script type=\"module\">
const result = document.querySelector('#result');
async function report(payload) {
  result.textContent = JSON.stringify(payload, null, 2);
  await fetch('/report', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload)});
}
function webgl2Facts() {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2', {alpha:false, antialias:false, preserveDrawingBuffer:true});
  return gl ? {available:true, version:gl.getParameter(gl.VERSION), renderer:gl.getParameter(gl.RENDERER), vendor:gl.getParameter(gl.VENDOR), maxTextureSize:gl.getParameter(gl.MAX_TEXTURE_SIZE)} : {available:false};
}
async function webgpuFacts() {
  if (!navigator.gpu) return {available:false, reason:'navigator.gpu is absent'};
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return {available:false, reason:'requestAdapter returned null'};
  const info = await adapter.requestAdapterInfo?.().catch(error => ({error:String(error)}));
  return {available:true, info, features:[...adapter.features], limits:{maxTextureDimension2D:adapter.limits.maxTextureDimension2D}};
}
try {
  await report({state:'capability', origin:location.origin, secureContext:isSecureContext, userAgent:navigator.userAgent, webgl2:webgl2Facts(), webgpu:await webgpuFacts()});
} catch (error) { await report({state:'error', stage:'capability', error:String(error), secureContext:isSecureContext, userAgent:navigator.userAgent}); }
</script>"""


def demo_page(backend: str, demo: str, entry: str) -> str:
    """Return a page that loads one wasm-bindgen web package and verifies it."""
    mode = {
        "05_push_constants": "dynamic-uniform fallback",
        "06_compute": "raster filter fallback",
        "08_msaa": "single-sample fallback",
        "10_indirect_draw": "direct draw fallback",
        "11_occlusion_query": "visibility fallback",
        "13_multithreaded_recording": "sequential drawing fallback",
        "14_descriptor_indexing": "fixed texture-array fallback",
        "15_multiview": "sequential stereo fallback",
    }.get(demo, "example scene") if backend == "webgl2" else "example scene"
    return f"""<!doctype html><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Fluxel {demo} {backend}</title>
<style>html,body{{margin:0;background:#000;color:#fff}}canvas{{display:block;width:100%;height:auto}}pre{{margin:0;font-size:12px;white-space:pre-wrap;max-height:6em;overflow:auto}}</style><pre id=\"result\">loading</pre><canvas id=\"image\" width=\"1280\" height=\"720\"></canvas>
<script type=\"module\">
const result = document.querySelector('#result');
const canvas = document.querySelector('#image');
async function report(payload) {{
  payload.runId = new URLSearchParams(location.search).get('run');
  payload.executionMode = '{mode}';
  result.textContent = payload.state === 'success'
    ? '{demo.split('_', 1)[0]} | {backend} | {mode} — completed 3 frames; stopped for capture'
    : JSON.stringify(payload, null, 2);
  await fetch('/report', {{method:'POST', headers:{{'content-type':'application/json'}}, body:JSON.stringify(payload)}});
}}
function webgl2Readback() {{
  const gl = canvas.getContext('webgl2');
  if (!gl) throw new Error('WebGL2 context disappeared before readback');
  const pixels = new Uint8Array(canvas.width * canvas.height * 4);
  gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const background = pixels.slice(0, 4);
  let changed = 0;
  for (let i = 0; i < pixels.length; i += 4) {{
    if (pixels[i] !== background[0] || pixels[i + 1] !== background[1] || pixels[i + 2] !== background[2] || pixels[i + 3] !== background[3]) changed++;
  }}
  if (changed <= 64) throw new Error(`WebGL2 drawing buffer is blank: changed=${{changed}} first=${{[...background]}}`);
  return {{changedPixels:changed, version:gl.getParameter(gl.VERSION), renderer:gl.getParameter(gl.RENDERER)}};
}}
try {{
  {'const gl = canvas.getContext(\'webgl2\', {alpha:false, antialias:false, preserveDrawingBuffer:true}); if (!gl) throw new Error(\'WebGL2 is unavailable\');' if backend == 'webgl2' else 'if (!navigator.gpu) throw new Error(\'WebGPU is unavailable\');'}
  const module = await import('./demo.js');
  await module.default();
  if (typeof module['{entry}'] !== 'function') throw new Error('missing wasm export: {entry}');
  await module['{entry}'](canvas, 3);
  const verification = {'webgl2Readback()' if backend == 'webgl2' else '{webgpu:true}'};
  // Allow the final canvas and status to reach Android's browser compositor.
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  await report({{state:'success', backend:'{backend}', demo:'{demo}', entry:'{entry}', requestedFrames:3, canvasExtent:[canvas.width,canvas.height], userAgent:navigator.userAgent, secureContext:isSecureContext, verification}});
}} catch (error) {{
  await report({{state:'error', backend:'{backend}', demo:'{demo}', entry:'{entry}', secureContext:isSecureContext, error:String(error), stack:error?.stack || null}});
}}
</script>"""


class BrowserHandler(SimpleHTTPRequestHandler):
    """Serves a bounded bundle tree plus report/status endpoints for one run."""

    server: "BrowserServer"

    def log_message(self, format: str, *args: object) -> None:
        """Keep request logs useful but avoid SimpleHTTPRequest's HTML noise."""
        sys.stderr.write("android-browser: " + format % args + "\n")

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path.split("?", 1)[0] == "/status":
            self.send_json(self.server.state.snapshot())
            return
        if self.path.split("?", 1)[0] == "/":
            body = self.server.page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        self.serve_bundle_file()

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/report":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REPORT_BYTES:
                raise ValueError("invalid report length")
            report = json.loads(self.rfile.read(length))
            if not isinstance(report, dict):
                raise ValueError("report must be an object")
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
            return
        self.server.state.add_report(report)
        self.send_json({"accepted": True})

    def send_json(self, value: object) -> None:
        body = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_bundle_file(self) -> None:
        relative = posixpath.normpath(self.path.split("?", 1)[0]).lstrip("/")
        candidate = (self.server.bundle_dir / relative).resolve()
        try:
            candidate.relative_to(self.server.bundle_dir)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "application/wasm" if candidate.suffix == ".wasm" else mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        payload = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


class BrowserServer(ThreadingHTTPServer):
    """Carries immutable page configuration alongside stdlib HTTP state."""

    def __init__(self, address: tuple[str, int], state: RunState, page: str, bundle_dir: Path) -> None:
        super().__init__(address, BrowserHandler)
        self.state = state
        self.page = page
        self.bundle_dir = bundle_dir


def serve(args: argparse.Namespace) -> None:
    """Run one local server until Ctrl-C so adb can reverse its localhost port."""
    bundle_dir = args.bundle_dir.resolve() if args.bundle_dir else Path.cwd()
    if not bundle_dir.is_dir():
        raise SystemExit(f"bundle directory does not exist: {bundle_dir}")
    if args.mode == "demo" and not (bundle_dir / "demo.js").is_file():
        raise SystemExit(f"demo bundle must contain demo.js: {bundle_dir}")
    state = RunState(args.result)
    page = probe_page() if args.mode == "probe" else demo_page(args.backend, args.demo, args.entry)
    server = BrowserServer((args.bind, args.port), state, page, bundle_dir)
    print(f"serving {args.mode} on http://{args.bind}:{args.port}/", flush=True)
    print(f"Android URL: http://localhost:{args.port}/", flush=True)
    print(f"Status URL: http://127.0.0.1:{args.port}/status", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def poll(args: argparse.Namespace) -> None:
    """Read one server status endpoint without requiring third-party clients."""
    try:
        with urlopen(args.url, timeout=args.timeout) as response:
            print(response.read().decode("utf-8"))
    except URLError as error:
        raise SystemExit(f"status request failed: {error}") from error


def parse_args() -> argparse.Namespace:
    """Parse the narrow probe/demo serving contract used by Android evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve_parser = subcommands.add_parser("serve", help="serve a probe or one wasm demo")
    serve_parser.add_argument("--mode", choices=("probe", "demo"), required=True)
    serve_parser.add_argument("--backend", choices=("webgl2", "webgpu"), default="webgl2")
    serve_parser.add_argument("--demo", default="capability-probe")
    serve_parser.add_argument("--entry", default="")
    serve_parser.add_argument("--bundle-dir", type=Path)
    serve_parser.add_argument("--result", type=Path)
    serve_parser.add_argument("--bind", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.set_defaults(func=serve)
    poll_parser = subcommands.add_parser("poll", help="print the latest JSON report")
    poll_parser.add_argument("--url", default="http://127.0.0.1:8765/status")
    poll_parser.add_argument("--timeout", type=float, default=5.0)
    poll_parser.set_defaults(func=poll)
    args = parser.parse_args()
    if args.command == "serve" and args.mode == "demo" and not args.entry:
        parser.error("--entry is required for demo mode")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
