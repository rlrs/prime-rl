#!/usr/bin/env bash
set -euo pipefail

# Experimental overlay builder for upstream vLLM main on ROCm 7.2.
#
# This does not reuse the container's ROCm/Torch stack. Instead it:
# - binds a separate ROCm 7.2 user-space/toolchain tree
# - creates an isolated overlay venv (no --system-site-packages)
# - installs the ROCm 7.2 PyTorch stack into that venv
# - builds vLLM main from source against the bound ROCm toolchain
#
# The builder writes generated runtime metadata that the LUMI launchers can consume:
# - overlay-runtime.env
# - overlay-runtime.binds
# - overlay-sidecar-manifest.json
# - overlay-runtime-check.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
PROJECT_SCRATCH=/pfs/lustrep4/scratch/project_465002183

: "${SIF:=$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}"
: "${OVERLAY_DIR:=$REPO_ROOT/overlay_vllm_rocm72_main}"
: "${ROCM_USERLAND_DIR:=$PROJECT_SCRATCH/rasmus/rocm-userland-exp/untar-7.2.1/rocm-7.2.1}"
: "${ROCM_USERLAND_VERSION:=7.2.1}"
: "${ROCM_BIND_PATH:=/rocm72}"
: "${PYTORCH_ROCM_ARCH:=gfx90a}"

: "${VLLM_REPO:=https://github.com/vllm-project/vllm.git}"
: "${VLLM_REF:=edcc37a8cee26813fe868b9fc267c3cba5818ff7}"

: "${PYTORCH_WHEEL_INDEX:=https://download.pytorch.org/whl/rocm7.2}"
: "${TORCH_VERSION:=2.11.0+rocm7.2}"
: "${TORCHVISION_VERSION:=0.26.0+rocm7.2}"
: "${TORCHAUDIO_VERSION:=2.11.0+rocm7.2}"
: "${TRITON_VERSION:=3.6.0}"
: "${TRANSFORMERS_VERSION:=5.5.0}"
: "${CMAKE_VERSION_SPEC:=cmake>=3.26.1,<4}"
: "${MAX_JOBS:=64}"
: "${VLLM_BUILD_JOBS:=8}"
: "${CONFIG_SMOKE_MODEL:=google/gemma-3-1b-it}"
: "${INSTALL_FLUENT_ALIGNMENT_DA:=0}"
: "${FLUENT_ALIGNMENT_DA_DIR:=$HOME/git/environments/environments/fluent-alignment-da}"

if [[ ! -f "$SIF" ]]; then
  echo "FATAL: SIF not found: $SIF" >&2
  exit 1
fi
if [[ ! -d "$ROCM_USERLAND_DIR" ]]; then
  echo "FATAL: ROCm userland dir not found: $ROCM_USERLAND_DIR" >&2
  exit 1
fi
if [[ "$INSTALL_FLUENT_ALIGNMENT_DA" == "1" && ! -f "$FLUENT_ALIGNMENT_DA_DIR/pyproject.toml" ]]; then
  echo "FATAL: fluent-alignment-da not found at: $FLUENT_ALIGNMENT_DA_DIR" >&2
  exit 1
fi

ROCM_USERLAND_DIR="$(readlink -f "$ROCM_USERLAND_DIR" 2>/dev/null || printf '%s' "$ROCM_USERLAND_DIR")"
if [[ ! -d "$ROCM_USERLAND_DIR/share/amd_smi" ]]; then
  echo "FATAL: ROCm userland missing share/amd_smi: $ROCM_USERLAND_DIR" >&2
  exit 1
fi
if [[ ! -d "$ROCM_USERLAND_DIR/lib/llvm/bin" ]]; then
  echo "FATAL: ROCm userland missing lib/llvm/bin: $ROCM_USERLAND_DIR" >&2
  exit 1
fi

mkdir -p "$OVERLAY_DIR"/{build,cache,src,venv,runtime}
chmod 700 "$OVERLAY_DIR"

echo "+ SIF: $SIF"
echo "+ Overlay: $OVERLAY_DIR"
echo "+ ROCm userland: $ROCM_USERLAND_DIR"
echo "+ ROCm bind path: $ROCM_BIND_PATH"
echo "+ ROCm arch: $PYTORCH_ROCM_ARCH"
echo "+ vLLM ref: $VLLM_REF"
echo "+ Torch wheel index: $PYTORCH_WHEEL_INDEX"
echo "+ Torch: $TORCH_VERSION"
echo "+ torchvision: $TORCHVISION_VERSION"
echo "+ torchaudio: $TORCHAUDIO_VERSION"
echo "+ triton/triton-rocm: $TRITON_VERSION"
echo "+ transformers: $TRANSFORMERS_VERSION"
echo "+ vLLM build jobs: $VLLM_BUILD_JOBS"
if [[ "$INSTALL_FLUENT_ALIGNMENT_DA" == "1" ]]; then
  echo "+ fluent-alignment-da: $FLUENT_ALIGNMENT_DA_DIR"
fi

export VLLM_REPO VLLM_REF
export PYTORCH_WHEEL_INDEX
export TORCH_VERSION TORCHVISION_VERSION TORCHAUDIO_VERSION TRITON_VERSION
export TRANSFORMERS_VERSION CMAKE_VERSION_SPEC MAX_JOBS VLLM_BUILD_JOBS CONFIG_SMOKE_MODEL
export ROCM_BIND_PATH ROCM_USERLAND_VERSION
export PYTORCH_ROCM_ARCH INSTALL_FLUENT_ALIGNMENT_DA

SINGULARITY_BIND_ARGS=(
  -B "$BASE_DIR:$LAIFS_APPL_DIR"
  -B "$PROJECT_SCRATCH:/scratch/project_465002183"
  -B "$OVERLAY_DIR:/overlay"
  -B "$ROCM_USERLAND_DIR:$ROCM_BIND_PATH"
)
if [[ "$INSTALL_FLUENT_ALIGNMENT_DA" == "1" ]]; then
  SINGULARITY_BIND_ARGS+=(-B "$FLUENT_ALIGNMENT_DA_DIR:/env-src/fluent-alignment-da")
fi

singularity exec --rocm \
  "${SINGULARITY_BIND_ARGS[@]}" \
  "$SIF" bash -eu -s <<'INSIDE'
if [[ ! -d /overlay/venv/vllm-min ]]; then
  python3 -m venv /overlay/venv/vllm-min
fi
source /overlay/venv/vllm-min/bin/activate

OVERLAY_SITE="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

export PIP_USER=0
unset PYTHONUSERBASE

export XDG_CACHE_HOME=/overlay/cache
export PIP_CACHE_DIR=/overlay/cache/pip
export UV_CACHE_DIR=/overlay/cache/uv
export TRITON_CACHE_DIR=/overlay/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/overlay/cache/torchinductor
export PYTORCH_KERNEL_CACHE_PATH=/overlay/cache/torch-kernels
export TMPDIR=/overlay/cache/tmp
export HOME=/overlay/cache/home
export HF_HOME=/scratch/project_465002183/.cache/huggingface
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTORCH_KERNEL_CACHE_PATH" \
  "$TMPDIR" "$HOME" "$HF_HOME"

mkdir -p /overlay/runtime
cat > /overlay/runtime/sitecustomize.py <<'PY'
import os
import sys


def _matches(entry: str, prefix: str) -> bool:
    normalized_entry = entry.rstrip("/")
    normalized_prefix = prefix.rstrip("/")
    return normalized_entry == normalized_prefix or normalized_entry.startswith(
        normalized_prefix + "/"
    )


preferred = os.environ.get("OVERLAY_PREFERRED_SITE", "").strip()
strip_prefixes = [
    prefix.strip()
    for prefix in os.environ.get("OVERLAY_STRIP_SITE_PREFIXES", "").split(":")
    if prefix.strip()
]

head = sys.path[:1]
tail = sys.path[1:]
new_path = list(head)
seen = set(head)

if preferred and preferred not in seen:
    new_path.append(preferred)
    seen.add(preferred)

for entry in tail:
    if any(_matches(entry, prefix) for prefix in strip_prefixes):
        continue
    if entry not in seen:
        new_path.append(entry)
        seen.add(entry)

sys.path[:] = new_path
PY

export PYTHONNOUSERSITE=1
unset PYTHONHOME
export OVERLAY_PREFERRED_SITE="${OVERLAY_SITE}"
export OVERLAY_STRIP_SITE_PREFIXES=/opt/venv/lib/python3.12/site-packages
export PYTHONPATH="/overlay/runtime:${OVERLAY_SITE}"

export PATH="${ROCM_BIND_PATH}/bin:${ROCM_BIND_PATH}/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="${ROCM_BIND_PATH}/lib:${ROCM_BIND_PATH}/lib64:${ROCM_BIND_PATH}/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_PATH="${ROCM_BIND_PATH}"
export ROCM_HOME="${ROCM_BIND_PATH}"
export HIP_PATH="${ROCM_BIND_PATH}"
export HSA_PATH="${ROCM_BIND_PATH}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}"
export VLLM_TARGET_DEVICE=rocm
export FETCHCONTENT_BASE_DIR=/overlay/src/vllm/.deps

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

manifest = {
    "type": "rocm_userland_sidecar",
    "version": os.environ["ROCM_USERLAND_VERSION"],
    "bind_path": os.environ["ROCM_BIND_PATH"],
    "required_at_runtime": True,
}

for name in ("amdclang++", "clang++", "hipcc"):
    candidate = Path(os.environ["ROCM_BIND_PATH"]) / "bin" / name
    if not candidate.exists():
        candidate = Path(os.environ["ROCM_BIND_PATH"]) / "lib" / "llvm" / "bin" / name
    if not candidate.exists():
        continue
    try:
        line = subprocess.check_output(
            [str(candidate), "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        line = f"<failed to query version: {exc}>"
    manifest[f"{name}_path"] = str(candidate)
    manifest[f"{name}_version"] = line

Path("/overlay/overlay-sidecar-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("+ Wrote /overlay/overlay-sidecar-manifest.json")
PY

python -m pip install --no-user -U \
  pip \
  "setuptools>=77,<80" \
  wheel \
  ninja \
  "${CMAKE_VERSION_SPEC}" \
  packaging \
  jinja2 \
  "setuptools-scm>=8"

python -m pip install --no-user -U --index-url "${PYTORCH_WHEEL_INDEX}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  "triton==${TRITON_VERSION}" \
  "triton-rocm==${TRITON_VERSION}"

rm -rf /overlay/build/amd_smi
mkdir -p /overlay/build
cp -a "${ROCM_BIND_PATH}/share/amd_smi" /overlay/build/amd_smi
python -m pip install --no-user -U \
  /overlay/build/amd_smi \
  "transformers==${TRANSFORMERS_VERSION}"

if [[ "$INSTALL_FLUENT_ALIGNMENT_DA" == "1" ]]; then
  python -m pip install --no-user -U /env-src/fluent-alignment-da
fi

python -m pip install --no-user -U \
  "beartype>=0.21.0" \
  "datasets>=4.0.0" \
  "jaxtyping>=0.3.2" \
  "liger-kernel>=0.5.10" \
  "loguru>=0.7.3" \
  "numpy>=2.2.6" \
  "openai>=1.106.1" \
  "prime>=0.5.37" \
  "pyarrow>=21.0.0" \
  "pydantic>=1.10.13" \
  "ring-flash-attn>=0.1.8" \
  "rich>=14.0.0" \
  "tilelang>=0.1.8" \
  "tomli>=2.2.1" \
  "tomli-w>=1.2.0" \
  "torchdata>=0.11.0" \
  "uvloop>=0.21.0" \
  "wandb>=0.24.2" \
  "aiolimiter>=1.2.1" \
  "pyzmq>=27.1.0" \
  "tenacity>=8.2.0"

python -m pip install --no-user -U \
  "pydantic-config @ git+https://github.com/samsja/pydantic_config.git@main" \
  "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@0db45e6" \
  "torchtitan @ git+https://github.com/pytorch/torchtitan.git@a1fdd7e" \
  "dion @ git+https://github.com/samsja/dion.git@d891eeb" \
  "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git"

if [[ ! -d /overlay/src/vllm/.git ]]; then
  git clone "${VLLM_REPO}" /overlay/src/vllm
fi
cd /overlay/src/vllm
git fetch --tags origin
if git rev-parse --verify "origin/${VLLM_REF}" >/dev/null 2>&1; then
  TARGET_REF="origin/${VLLM_REF}"
else
  TARGET_REF="${VLLM_REF}"
fi
git reset --hard
git clean -fdx
git checkout --detach "${TARGET_REF}"

python -m pip install --no-user -U -r requirements/rocm.txt

export MAX_JOBS="${VLLM_BUILD_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${VLLM_BUILD_JOBS}"
python -m pip install --no-user --no-build-isolation --no-deps -e .

python - <<'PY'
import importlib.metadata as md
import os
import sys
from pathlib import Path

import amdsmi
import torch
if os.environ["INSTALL_FLUENT_ALIGNMENT_DA"] == "1":
    import fluent_alignment_da
import pydantic_config
import tomli_w
import verifiers
import triton
import transformers
import vllm
from transformers import AutoConfig
from vllm.transformers_utils.config import get_config

paths = {
    "torch": Path(torch.__file__).resolve(),
    "triton": Path(triton.__file__).resolve(),
    "transformers": Path(transformers.__file__).resolve(),
    "vllm": Path(vllm.__file__).resolve(),
    "amdsmi": Path(amdsmi.__file__).resolve(),
    "pydantic_config": Path(pydantic_config.__file__).resolve(),
    "tomli_w": Path(tomli_w.__file__).resolve(),
    "verifiers": Path(verifiers.__file__).resolve(),
}

summary = {
    "python": sys.executable,
    "sys_path": sys.path,
    "paths": {name: str(path) for name, path in paths.items()},
    "versions": {},
}
failures = []

for dist_name in (
    "vllm",
    "torch",
    "triton",
    "triton-rocm",
    "amdsmi",
    "transformers",
    "pydantic-config",
    "tomli-w",
    "verifiers",
    "prime",
):
    try:
        summary["versions"][dist_name] = md.version(dist_name)
    except md.PackageNotFoundError:
        summary["versions"][dist_name] = None
        failures.append(f"missing distribution metadata for {dist_name}")

for name, path in paths.items():
    if str(path).startswith("/opt/venv/"):
        failures.append(f"{name} resolved from base container: {path}")

for entry in sys.path:
    if entry.startswith("/opt/venv/lib/python3.12/site-packages"):
        failures.append(f"/opt/venv still present on sys.path: {entry}")

model_id = os.environ["CONFIG_SMOKE_MODEL"]
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
v_cfg = get_config(model_id, trust_remote_code=False)
summary["autoconfig_model_type"] = cfg.model_type
summary["vllm_get_config_model_type"] = v_cfg.model_type
summary["torch_hip_version"] = torch.version.hip

Path("/overlay/overlay-runtime-check.json").write_text(
    __import__("json").dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("torch:", torch.__version__, torch.version.hip)
for dist_name, version in summary["versions"].items():
    print(f"{dist_name}:", version if version is not None else "<missing>")
for name, path in summary["paths"].items():
    print(f"{name} path:", path)
print("AutoConfig model_type:", summary["autoconfig_model_type"])
print("vllm.get_config model_type:", summary["vllm_get_config_model_type"])
print("+ Wrote /overlay/overlay-runtime-check.json")

if failures:
    for failure in failures:
        print("SANITY CHECK FAILED:", failure)
    raise SystemExit(1)
PY

python -m pip freeze > /overlay/overlay-pip-freeze.txt
echo "+ Wrote /overlay/overlay-pip-freeze.txt"
INSIDE

cat > "$OVERLAY_DIR/overlay-runtime.binds" <<EOF
$ROCM_USERLAND_DIR:$ROCM_BIND_PATH
EOF

cat > "$OVERLAY_DIR/overlay-runtime.env" <<EOF
# Generated by scripts/build_overlay_minimal.sh
OVERLAY_SITE=/overlay/venv/vllm-min/lib/python3.12/site-packages
OVERLAY_RUNTIME_SITE=/overlay/runtime
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export OVERLAY_PREFERRED_SITE="\${OVERLAY_SITE}"
export OVERLAY_STRIP_SITE_PREFIXES="/opt/venv/lib/python3.12/site-packages"
export PATH="${ROCM_BIND_PATH}/bin:${ROCM_BIND_PATH}/lib/llvm/bin:\$PATH"
export LD_LIBRARY_PATH="${ROCM_BIND_PATH}/lib:${ROCM_BIND_PATH}/lib64:${ROCM_BIND_PATH}/lib/llvm/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export ROCM_PATH=${ROCM_BIND_PATH}
export ROCM_HOME=${ROCM_BIND_PATH}
export HIP_PATH=${ROCM_BIND_PATH}
export HSA_PATH=${ROCM_BIND_PATH}
export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
export VLLM_TARGET_DEVICE=rocm
export PYTHONPATH="\${OVERLAY_RUNTIME_SITE}:\${OVERLAY_SITE}\${PYTHONPATH:+:\$PYTHONPATH}"
EOF

echo "+ Wrote $OVERLAY_DIR/overlay-runtime.binds"
echo "+ Wrote $OVERLAY_DIR/overlay-runtime.env"
echo "+ Wrote $OVERLAY_DIR/overlay-sidecar-manifest.json"
echo "+ Wrote $OVERLAY_DIR/overlay-runtime-check.json"
echo "+ Build complete."
echo "+ Use OVERLAY_DIR=$OVERLAY_DIR with scripts/lumi_submit_multi_node_rl.sh"
