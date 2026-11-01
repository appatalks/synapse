#!/usr/bin/env python3
"""Build a pinned, opt-in Linux/NVIDIA research backend without system changes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synapse_app.telemetry import SystemProbe
from synapse_app.updates import download

COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
RESEARCH = "https://raw.githubusercontent.com/appatalks/unified-memory-scheduler/6b952cc5c1d88f44dd50e4097b9ad3d8f24effdc"
PATCH_SHA = "990bd2ce296c3ad65918b684428c6a6340674619a9f1dc8a1ddee1d72a39d88f"
PLAN_SHA = "6a679bec76f1a410ed65fa79c186d164a388fc76419d96dcad5978782343949c"


def checked_file(url, destination, expected):
    body = download(url, 4 * 1024 * 1024)
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("Pinned research artifact checksum mismatch")
    destination.write_bytes(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This optional executor is Linux/NVIDIA-only.")
    missing = [name for name in ("git", "cmake", "c++", "nvcc", "nvidia-smi") if not shutil.which(name)]
    if missing:
        parser.error("Missing build requirements: " + ", ".join(missing) +
                     ". Install them yourself, or use the standard app. This installer never installs drivers/toolkits.")
    root = args.prefix.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("Backend directory already contains files. Keep it for diagnosis or choose another prefix.")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    total_ram, _ = SystemProbe().memory()
    vram = int(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.total",
                                      "--format=csv,noheader,nounits"], text=True, timeout=10).strip())
    if vram < 4096:
        parser.error("At least4GiB of NVIDIA VRAM is required for this optional setup; fit is still model-dependent.")
    print("Building the experimental engine. Quality checks remain unresolved; speedups are not universal.", flush=True)
    print("No model weights, drivers or system packages will be installed.", flush=True)
    source, build = root / "source", root / "build"
    subprocess.run(["git", "init", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "remote", "add", "origin",
                    "https://github.com/ggml-org/llama.cpp.git"], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", COMMIT], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError("Backend source does not match the pinned commit")
    patch = root / "hybrid-experts.patch"
    checked_file(RESEARCH + "/third_party/patches/llama-b11669-hybrid-experts.patch", patch, PATCH_SHA)
    subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_CUDA=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_NATIVE=OFF",
                    "-DGGML_CPU_ALL_VARIANTS=ON", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "llama-server", "--parallel",
                    str(min(8, os.cpu_count() or 1))], check=True)
    executable, plugin = build / "bin/llama-server", build / "bin/libggml-cuda.so"
    if not executable.is_file() or not plugin.is_file():
        raise ValueError("Build completed without the expected matching server/CUDA plugin")
    profile = {"schema_version": 2, "backend_commit": COMMIT, "executable": str(executable),
               "cuda_plugin": str(plugin), "plan": "auto", "context": 16384,
               "cache_mib": min(8192, vram // 2), "generic_cache_mib": min(8192, vram // 2),
               "transfer": "staged", "generic_transfer": "staged", "decode_pipeline": True, "async_ids": True}
    if vram >= 16000 and total_ram >= 96 * 1024 ** 3:
        plan = root / "trained-hot944.txt"
        checked_file(RESEARCH + "/benchmarks/results/gpt120-trained-hot944.txt", plan, PLAN_SHA)
        profile.update(plan=str(plan), cache_mib=12000, transfer="direct", generic_transfer="direct",
                       model_layout={"architecture": "gpt-oss", "file_type": 38, "block_count": 36,
                                     "embedding_length": 2880, "expert_feed_forward_length": 2880,
                                     "expert_count": 128, "expert_used_count": 4})
    # The application validates the descriptor binding, actual tensors, memory and activation again at load time.
    from synapse_app.app import read_hybrid_profile
    path = root / "hybrid-profile.json"
    path.write_text(json.dumps(profile, indent=2))
    path.chmod(0o600)
    read_hybrid_profile(path)
    print("Pinned backend/profile installed. Actual model fit and quality are not guaranteed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Accelerated setup failed: {error}") from error
