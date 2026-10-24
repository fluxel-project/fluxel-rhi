#!/usr/bin/env python3
"""Build one browser-facing Fluxel example and bind it for an Android browser.

Each invocation handles one numbered example.  It intentionally does not loop
over all demos: capture evidence must be reviewed one example at a time.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEMOS: dict[str, dict[str, str]] = {
    "01_triangle": {"webgl2": "run_webgl2_triangle", "webgpu": "run_webgpu_triangle"},
    "02_texture": {"webgl2": "run_webgl2_texture", "webgpu": "run_webgpu_texture"},
    "03_instancing": {"webgl2": "run_webgl2_instancing", "webgpu": "run_webgpu_instancing"},
    "04_dynamic_uniform_buffer": {"webgl2": "run_webgl2_dynamic_uniforms", "webgpu": "run_webgpu_dynamic_uniforms"},
    "05_push_constants": {"webgl2": "run_webgl2_push_constants", "webgpu": "run_webgpu_push_constants"},
    "06_compute": {"webgl2": "run_webgl2_compute_fallback", "webgpu": "run_webgpu_compute"},
    "07_offscreen": {"webgl2": "run_webgl2_offscreen", "webgpu": "run_webgpu_offscreen"},
    "08_msaa": {"webgl2": "run_webgl2_msaa_fallback", "webgpu": "run_webgpu_msaa"},
    "09_texture_mipmap": {"webgl2": "run_webgl2_texture_mipmap", "webgpu": "run_webgpu_texture_mipmap"},
    "10_indirect_draw": {"webgl2": "run_webgl2_indirect_fallback", "webgpu": "run_webgpu_indirect"},
    "11_occlusion_query": {"webgl2": "run_webgl2_occlusion_fallback", "webgpu": "run_webgpu_occlusion_query"},
    "12_screenshot": {"webgl2": "run_webgl2_screenshot", "webgpu": "run_webgpu_screenshot"},
    "13_multithreaded_recording": {"webgl2": "run_webgl2_multithreaded_recording", "webgpu": "run_webgpu_multithreaded_recording"},
    "14_descriptor_indexing": {"webgl2": "run_webgl2_descriptor_indexing", "webgpu": "run_webgpu_descriptor_indexing"},
    "15_multiview": {"webgl2": "run_webgl2_multiview", "webgpu": "run_webgpu_multiview"},
}


def main() -> None:
    """Build one wasm example then produce the ``demo.js`` web package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=DEMOS)
    parser.add_argument("--backend", choices=("webgl2", "webgpu"), required=True)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    profile = "release" if args.release else "debug"
    wasm = workspace / "target" / "wasm32-unknown-unknown" / profile / "examples" / f"{args.demo}.wasm"
    worker_wasm = workspace / "target" / "wasm32-unknown-unknown" / profile / "fluxel_rhi_webgpu.wasm"
    output = args.out.resolve()
    cargo = [
        "cargo", "build", "-p", "fluxel-rhi", "--no-default-features",
        "--features", f"examples,{args.backend}", "--target", "wasm32-unknown-unknown",
        "--example", args.demo,
    ]
    if args.release:
        cargo.append("--release")
    bindgen = ["wasm-bindgen", str(wasm), "--target", "web", "--out-dir", str(output), "--out-name", "demo"]
    worker_cargo = ["cargo", "build", "-p", "fluxel-rhi-webgpu", "--target", "wasm32-unknown-unknown"]
    if args.release:
        worker_cargo.append("--release")
    worker_bindgen = ["wasm-bindgen", str(worker_wasm), "--target", "web", "--out-dir", str(output), "--out-name", "fluxel_shader_worker"]
    commands = (cargo, bindgen)
    if args.backend == "webgpu":
        commands += (worker_cargo, worker_bindgen)
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=workspace, check=True)
    print(f"entry={DEMOS[args.demo][args.backend]}")
    print(f"bundle={output}")


if __name__ == "__main__":
    main()
