#!/usr/bin/env python3
"""Build the existing browser examples and capture them through ChromeDriver.

Runs only the example export entry points already present in this repository.
No example source or scene configuration is rewritten. The output records the
candidate SHA, browser adapter, diagnostics, generated module hashes and PNGs.
"""
import argparse
import base64
import functools
import hashlib
import http.server
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("01_triangle", "run_webgpu_triangle"),
    ("02_texture", "run_webgpu_texture"),
    ("03_instancing", "run_webgpu_instancing"),
    ("04_dynamic_uniform_buffer", "run_webgpu_dynamic_uniforms"),
    ("05_push_constants", "run_webgpu_push_constants"),
    ("06_compute", "run_webgpu_compute"),
    ("07_offscreen", "run_webgpu_offscreen"),
    ("08_msaa", "run_webgpu_msaa"),
    ("09_texture_mipmap", "run_webgpu_texture_mipmap"),
    ("10_indirect_draw", "run_webgpu_indirect"),
    ("11_occlusion_query", "run_webgpu_occlusion_query"),
    ("12_screenshot", "run_webgpu_screenshot"),
    ("13_multithreaded_recording", "run_webgpu_multithreaded_recording"),
    ("14_descriptor_indexing", "run_webgpu_descriptor_indexing"),
    ("15_multiview", "run_webgpu_multiview"),
]
PAGE = '''<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,">
<style>html,body{margin:0;background:#000}canvas{display:block}</style>
<canvas id="image" width="1280" height="720"></canvas>
<script type="module">
window.result={done:false,error:null};
try {
  const adapter=await navigator.gpu.requestAdapter();
  const info=adapter.info;
  window.gpuEnvironment={vendor:info.vendor,architecture:info.architecture,
    device:info.device,description:info.description,isFallbackAdapter:info.isFallbackAdapter,
    features:Array.from(adapter.features),userAgent:navigator.userAgent};
  const query=new URLSearchParams(location.search);
  const module=await import(`./${query.get('demo')}/demo.js`);
  await module.default();
  await module[query.get('entry')](document.querySelector('canvas'),3);
} catch(error) { window.result.error=String(error); }
window.result.done=true;
</script>'''


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chromedriver", required=True, type=Path)
    parser.add_argument("--chrome", required=True, type=Path)
    parser.add_argument("--toolchain", default="1.87.0")
    parser.add_argument("--cases", nargs="*", help="Default: all 15 browser exports")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    web = out / "web"
    web.mkdir()
    cases = [case for case in CASES if not args.cases or case[0] in args.cases]
    assert cases, "no selected browser examples"
    build = ["cargo", f"+{args.toolchain}", "build", "-p", "fluxel-rhi",
             "--no-default-features", "--features", "examples,webgpu",
             "--target", "wasm32-unknown-unknown"]
    for name, _entry in cases:
        build.extend(["--example", name])
    with (out / "build.stdout").open("wb") as stdout, (out / "build.stderr").open("wb") as stderr:
        subprocess.run(build, cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
    bindgen = shutil.which("wasm-bindgen")
    assert bindgen, "wasm-bindgen is required"
    for name, _entry in cases:
        subprocess.run([bindgen, str(ROOT / "target/wasm32-unknown-unknown/debug/examples" / f"{name}.wasm"),
                        "--target", "web", "--out-dir", str(web / name), "--out-name", "demo"],
                       cwd=ROOT, check=True)
    (web / "capture.html").write_text(PAGE, encoding="utf-8")
    handler = functools.partial(QuietHandler, directory=str(web))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    driver_log = (out / "driver.log").open("wb")
    driver = subprocess.Popen([str(args.chromedriver.resolve()), "--port=0"], stdout=driver_log,
                              stderr=subprocess.STDOUT, creationflags=flags)
    session = None
    report = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
              "source_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode(),
              "width": 1280, "height": 720, "frames": 3, "cases": [], "build_command": build}
    try:
        port = None
        for _ in range(100):
            text = (out / "driver.log").read_text(errors="replace")
            import re
            found = re.search(r"started successfully on port (\d+)", text)
            if found:
                port = int(found.group(1))
                break
            if driver.poll() is not None:
                raise RuntimeError(text)
            time.sleep(0.1)
        if not port:
            raise TimeoutError("ChromeDriver did not report its port")

        def command(method, path, payload=None):
            request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                      data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)["value"]
            if isinstance(result, dict) and "error" in result and "stacktrace" in result:
                raise RuntimeError(result)
            return result

        started = command("POST", "/session", {"capabilities": {"alwaysMatch": {
            "browserName": "chrome", "goog:loggingPrefs": {"browser": "ALL"},
            "goog:chromeOptions": {"binary": str(args.chrome.resolve()), "args": [
                "--headless=new", "--enable-unsafe-webgpu", "--disable-gpu-sandbox",
                "--window-size=1280,720", "--force-device-scale-factor=1"]}}}})
        session = started["sessionId"]
        report["browser"] = started["capabilities"]
        prefix = f"/session/{session}"
        command("POST", prefix + "/goog/cdp/execute", {"cmd": "Emulation.setDeviceMetricsOverride",
                "params": {"width": 1280, "height": 720, "deviceScaleFactor": 1, "mobile": False}})
        for name, entry in cases:
            command("POST", prefix + "/url", {"url": f"http://127.0.0.1:{server.server_port}/capture.html?demo={name}&entry={entry}"})
            deadline = time.monotonic() + 120
            while True:
                state = command("POST", prefix + "/execute/sync", {"script": "return window.result || null", "args": []})
                if state and state.get("done"):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(name)
                time.sleep(0.1)
            logs = command("POST", prefix + "/log", {"type": "browser"})
            environment = command("POST", prefix + "/execute/sync",
                                  {"script": "return window.gpuEnvironment", "args": []})
            png = out / f"{name}.png"
            png.write_bytes(base64.b64decode(command("GET", prefix + "/screenshot")))
            record = {"example": name, "entry": entry, "state": state, "diagnostics": logs,
                      "environment": environment,
                      "png": png.name, "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest()}
            report["cases"].append(record)
            print(f"{name}: {state}", flush=True)
            if state.get("error"):
                raise RuntimeError(record)
        report["modules"] = {p.relative_to(web).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in web.rglob("*") if p.is_file()}
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        if session:
            try:
                command("DELETE", f"/session/{session}")
            except Exception:
                pass
        driver.terminate()
        driver.wait(timeout=15)
        driver_log.close()
        server.shutdown()


if __name__ == "__main__":
    main()
