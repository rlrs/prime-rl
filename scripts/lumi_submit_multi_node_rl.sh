#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

FLASH_PROJECT_ROOT=${FLASH_PROJECT_ROOT:-/flash/project_465002183}
HF_HOME=${HF_HOME:-$FLASH_PROJECT_ROOT/.cache/huggingface}
UV_CACHE_DIR=${UV_CACHE_DIR:-$FLASH_PROJECT_ROOT/.cache/uv}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$FLASH_PROJECT_ROOT/.cache}
WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$XDG_CACHE_HOME/wandb}
WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$FLASH_PROJECT_ROOT/.config/wandb}

CONFIG=${CONFIG:-$REPO_DIR/configs/lumi/rl_multi_node.toml}
OVERLAY_DIR=${OVERLAY_DIR:-/scratch/project_465002183/andreas/overlay_vllm_rocm72_main}
OVERLAY_ENV_NAME=${OVERLAY_ENV_NAME:-vllm-min}

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

export FLASH_PROJECT_ROOT
export HF_HOME
export UV_CACHE_DIR
export XDG_CACHE_HOME
export WANDB_CACHE_DIR
export WANDB_CONFIG_DIR
export OVERLAY_DIR
export OVERLAY_ENV_NAME

cd "$REPO_DIR"

uv run rl @ "$CONFIG" "$@"
