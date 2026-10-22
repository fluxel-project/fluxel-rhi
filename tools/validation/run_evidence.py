#!/usr/bin/env python3
"""Run validation commands and preserve raw output and Git provenance.

Examples:
  python tools/validation/run_evidence.py command --out target/evidence/check -- cargo check --workspace
  python tools/validation/run_evidence.py demos --out target/evidence/demos

Each run writes separate stdout/stderr byte streams and a JSON manifest. The
manifest records source status before and after execution; a dirty working tree
is evidence of a working tree, not proof of a frozen commit. Files are created
exclusively so an earlier run cannot be overwritten.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def snapshot():
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD", "--")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    files = {}
    for name in filter(None, untracked.split(b"\0")):
        path = ROOT / name.decode("utf-8", "surrogateescape")
        if path.is_file():
            files[str(path.relative_to(ROOT))] = sha(path.read_bytes())
    return {
        "head": git("rev-parse", "HEAD").decode("ascii").strip(),
        "status_porcelain_z_hex": status.hex(),
        "status_sha256": sha(status),
        "tracked_diff_sha256": sha(diff),
        "untracked_file_sha256": files,
    }


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_new(path, data):
    with path.open("xb") as output:
        output.write(data)


def run(prefix, command, timeout=1200):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [Path(str(prefix) + suffix) for suffix in (".stdout", ".stderr", ".json")]
    if any(p.exists() for p in paths):
        raise FileExistsError(f"Evidence already exists for {prefix}")
    before = snapshot()
    start = now()
    executable = shutil.which(command[0]) or command[0]
    try:
        result = subprocess.run([executable, *command[1:]], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout)
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, returncode = error.stdout or b"", error.stderr or b"", 124
    end = now()
    after = snapshot()
    write_new(paths[0], stdout)
    write_new(paths[1], stderr)
    manifest = {
        "cwd": str(ROOT), "command": command, "resolved_executable": executable,
        "start_utc": start, "end_utc": end, "returncode": returncode, "timeout_seconds": timeout,
        "environment": {key: os.environ[key] for key in (
            "VK_INSTANCE_LAYERS", "VK_SCREENSHOT_FRAMES", "VK_SCREENSHOT_DIR",
            "VK_SCREENSHOT_FORMAT", "FLUXEL_VULKAN_CAPTURE", "FLUXEL_VULKAN_CAPTURE_PASS"
        ) if key in os.environ},
        "stdout": {"path": paths[0].name, "bytes": len(stdout), "sha256": sha(stdout)},
        "stderr": {"path": paths[1].name, "bytes": len(stderr), "sha256": sha(stderr)},
        "source_before": before, "source_after": after,
    }
    write_new(paths[2], (json.dumps(manifest, indent=2, ensure_ascii=True) + "\n").encode("utf-8"))
    print(f"{prefix}: exit {returncode}", flush=True)
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("command", "demos"))
    parser.add_argument("--out", required=True, type=Path, help="Output prefix for command, directory for demos")
    args, remaining = parser.parse_known_args()
    if args.mode == "command":
        command = remaining[1:] if remaining[:1] == ["--"] else remaining
        if not command:
            parser.error("command mode requires a command after --")
        return run(args.out, command)
    if remaining:
        parser.error("demos mode takes no command")
    cargo = (ROOT / "crates/rhi/Cargo.toml").read_text(encoding="utf-8")
    names = re.findall(r'^name\s*=\s*"(\d\d_[^"]+)"\s*$', cargo, re.MULTILINE)
    if len(names) != 23 or [int(n[:2]) for n in names] != list(range(1, 24)):
        raise ValueError("Expected examples 01 through 23 in crates/rhi/Cargo.toml")
    failures = 0
    for name in names:
        failures += run(args.out / name, ["cargo", "run", "-p", "fluxel-rhi", "--no-default-features", "--features", "examples,vulkan", "--example", name, "--", "--backend", "vulkan", "--frames", "3"], timeout=120) != 0
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
