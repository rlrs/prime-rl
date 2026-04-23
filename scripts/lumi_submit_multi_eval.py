"""Submit parallel per-model generation jobs + one dependent judge job.

Each model gets its own Slurm allocation (fresh node, one vLLM lifecycle) to avoid
the errno-108 / post-shutdown state problems we hit when swapping vLLM models
serially inside a single allocation. The judge job chains on `afterok:<all ids>`
and scores every model's outputs, then renders the pairwise comparison markdowns.
"""

import argparse
import copy
import subprocess
import tomllib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def format_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise TypeError(f"Unsupported TOML value: {value!r}")


def render_toml_section(lines: list[str], prefix: list[str], data: dict) -> None:
    scalars = [(k, v) for k, v in data.items() if not isinstance(v, dict)]
    nesteds = [(k, v) for k, v in data.items() if isinstance(v, dict)]
    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")
    for key, value in scalars:
        lines.append(f"{key} = {format_toml_value(value)}")
    if scalars and nesteds:
        lines.append("")
    for index, (key, value) in enumerate(nesteds):
        render_toml_section(lines, [*prefix, key], value)
        if index != len(nesteds) - 1:
            lines.append("")


def render_toml(data: dict) -> str:
    lines: list[str] = []
    render_toml_section(lines, [], data)
    return "\n".join(lines)


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_toml(data) + "\n")


def quote_shell(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_model_inference_config(
    *,
    model_entry: dict,
    model_template: dict,
    inference: dict,
    gpus_per_node: int,
    output_dir: Path,
) -> dict:
    model_section = deep_merge(model_template, {"name": model_entry["name"]})
    return {
        "output_dir": str(output_dir / "inference_artifacts" / model_entry["label"]),
        "gpu_memory_utilization": inference.get("gpu_memory_utilization", 0.85),
        "model": model_section,
        "parallel": inference.get("parallel", {"tp": 8}),
        "deployment": {"gpus_per_node": gpus_per_node},
    }


def build_judge_inference_config(
    *,
    judge: dict,
    gpus_per_node: int,
    output_dir: Path,
) -> dict:
    inference = judge.get("inference", {})
    return {
        "output_dir": str(output_dir / "inference_artifacts" / "judge"),
        "gpu_memory_utilization": inference.get("gpu_memory_utilization", 0.9),
        "model": judge["model"],
        "parallel": inference.get("parallel", {"tp": 8}),
        "deployment": {"gpus_per_node": gpus_per_node},
    }


SINGULARITY_PRELUDE = """\
BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs
PROJECT_SCRATCH=/pfs/lustrep4/scratch/project_465002183

SIF=${SIF:-$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}
OVERLAY_DIR=${OVERLAY_DIR:-/scratch/project_465002183/andreas/overlay_vllm_rocm72_main}
OVERLAY_ENV_NAME=${OVERLAY_ENV_NAME:-vllm-min}
FLASH_PROJECT_ROOT=${FLASH_PROJECT_ROOT:-/flash/project_465002183}
FLASH_CACHE_ROOT=${FLASH_CACHE_ROOT:-$FLASH_PROJECT_ROOT/andreas/.cache}
HF_HOME=${HF_HOME:-$FLASH_CACHE_ROOT/huggingface}
UV_CACHE_DIR=${UV_CACHE_DIR:-$FLASH_CACHE_ROOT/uv}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$FLASH_CACHE_ROOT}
WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$FLASH_CACHE_ROOT/wandb}
WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$FLASH_PROJECT_ROOT/.config/wandb}

if [[ ! -f "$SIF" ]]; then
  echo "FATAL: SIF not found: $SIF" >&2
  exit 1
fi

if [[ ! -d "$OVERLAY_DIR/venv/$OVERLAY_ENV_NAME" ]]; then
  echo "FATAL: overlay env not found: $OVERLAY_DIR/venv/$OVERLAY_ENV_NAME" >&2
  exit 1
fi

export OVERLAY_ENV_NAME
export HF_HOME
export UV_CACHE_DIR
export XDG_CACHE_HOME
export WANDB_CACHE_DIR
export WANDB_CONFIG_DIR

SINGULARITY_BIND_ARGS=(
  -B "$BASE_DIR:$LAIFS_APPL_DIR"
  -B "$PROJECT_SCRATCH:/scratch/project_465002183"
  -B "$OVERLAY_DIR:/overlay"
  -B "$PROJECT_DIR:/workdir"
)

if [[ -f "$OVERLAY_DIR/overlay-runtime.binds" ]]; then
  while IFS= read -r bind; do
    [[ -n "$bind" ]] || continue
    SINGULARITY_BIND_ARGS+=(-B "$bind")
  done < "$OVERLAY_DIR/overlay-runtime.binds"
fi

if [[ -d /flash ]]; then
  SINGULARITY_BIND_ARGS+=(-B /flash:/flash)
fi
if [[ -e "$FLASH_PROJECT_ROOT" ]]; then
  FLASH_PROJECT_TARGET="$(readlink -f "$FLASH_PROJECT_ROOT" || true)"
  if [[ -n "$FLASH_PROJECT_TARGET" && -d "$FLASH_PROJECT_TARGET" ]]; then
    SINGULARITY_BIND_ARGS+=(-B "$FLASH_PROJECT_TARGET:$FLASH_PROJECT_TARGET")
  elif [[ -d "$FLASH_PROJECT_ROOT" ]]; then
    SINGULARITY_BIND_ARGS+=(-B "$FLASH_PROJECT_ROOT:$FLASH_PROJECT_ROOT")
  fi
fi
if [[ -d /scratch ]]; then
  SINGULARITY_BIND_ARGS+=(-B /scratch:/scratch)
fi
"""


CONTAINER_ENV_SETUP = """\
set -euo pipefail

source /overlay/venv/"$OVERLAY_ENV_NAME"/bin/activate
OVERLAY_SITE=/overlay/venv/"$OVERLAY_ENV_NAME"/lib/python3.12/site-packages
export PYTHONPATH=/workdir/src:"${OVERLAY_SITE}"${PYTHONPATH:+:${PYTHONPATH}}
export UV_NO_SYNC=1

if [[ -f /overlay/overlay-runtime.env ]]; then
  source /overlay/overlay-runtime.env
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi
if [[ -n "${HIP_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  export ROCR_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
fi
if [[ -n "${SSL_CERT_FILE:-}" && ! -f "${SSL_CERT_FILE}" ]]; then
  unset SSL_CERT_FILE
fi
if [[ -n "${SSL_CERT_DIR:-}" && ! -d "${SSL_CERT_DIR}" ]]; then
  unset SSL_CERT_DIR
fi

export HF_HOME
export UV_CACHE_DIR
export XDG_CACHE_HOME
export WANDB_CACHE_DIR
export WANDB_CONFIG_DIR
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /workdir
[ -f .env ] && source .env
"""


def build_generation_sbatch(
    *,
    config: dict,
    output_dir: Path,
    model: dict,
    infer_config_path: Path,
    output_path: Path,
) -> str:
    label = model["label"]
    input_path = Path(config["input_path"])
    server_port = config.get("server", {}).get("port", 8000)
    slurm = config["slurm"]
    deployment = config.get("deployment", {})
    generation = config.get("generation", {})
    project_dir = Path(slurm["project_dir"])

    partition = slurm.get("partition", "standard-g")
    account = slurm.get("account")
    time_limit = slurm.get("time", "02:00:00")
    job_name = f"{slurm.get('job_name', 'eval-multi')}-gen-{label}"
    gpus_per_node = deployment.get("gpus_per_node", 8)

    slurm_logs = output_dir / "slurm_logs"
    inference_stdout = output_dir / "inference_artifacts" / f"{label}.stdout.log"

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --partition={partition}
{f'#SBATCH --account={account}' if account else ''}
#SBATCH --time={time_limit}
#SBATCH --mem=0
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output={slurm_logs}/gen_{label}_%j.log
#SBATCH --error={slurm_logs}/gen_{label}_%j.err

set -euo pipefail

PROJECT_DIR={project_dir}
INFER_CONFIG_PATH={infer_config_path}
INPUT_PATH={input_path}
OUTPUT_PATH={output_path}
PORT={server_port}

mkdir -p "{output_dir}" "{output_path.parent}" "{inference_stdout.parent}" "{slurm_logs}"

{SINGULARITY_PRELUDE}
export PROJECT_DIR
export INFER_CONFIG_PATH
export INPUT_PATH
export OUTPUT_PATH
export PORT

singularity exec --rocm "${{SINGULARITY_BIND_ARGS[@]}}" "$SIF" bash -c '
{CONTAINER_ENV_SETUP}
python -m prime_rl.inference.server \\
  @ "$INFER_CONFIG_PATH" \\
  --server.host 0.0.0.0 \\
  --server.port "$PORT" \\
  > {quote_shell(str(inference_stdout))} 2>&1 &
INFER_PID=$!

cleanup() {{
  kill "$INFER_PID" 2>/dev/null || true
  wait "$INFER_PID" 2>/dev/null || true
}}
trap cleanup EXIT

for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$INFER_PID" 2>/dev/null; then
    echo "FATAL: vLLM died during startup" >&2
    tail -n 100 {quote_shell(str(inference_stdout))} >&2 || true
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null

python scripts/generate_eval_outputs.py \\
  --input-path "$INPUT_PATH" \\
  --output-path "$OUTPUT_PATH" \\
  --endpoint "127.0.0.1:$PORT" \\
  --model {quote_shell(model["name"])} \\
  --api-key {quote_shell(generation.get("api_key", "EMPTY"))} \\
  --temperature {generation.get("temperature", 0.7)} \\
  --top-p {generation.get("top_p", 1.0)} \\
  --max-tokens {generation.get("max_tokens", 4096)} \\
  --overwrite
'
"""


def build_judge_sbatch(
    *,
    config: dict,
    output_dir: Path,
    judge_infer_config_path: Path,
) -> str:
    server_port = config.get("server", {}).get("port", 8000)
    slurm = config["slurm"]
    deployment = config.get("deployment", {})
    judge = config["judge"]
    judge_model_name = judge["model"]["name"]
    scoring = judge.get("scoring", {})
    project_dir = Path(slurm["project_dir"])

    partition = slurm.get("partition", "standard-g")
    account = slurm.get("account")
    time_limit = slurm.get("time", "02:00:00")
    job_name = f"{slurm.get('job_name', 'eval-multi')}-judge"
    gpus_per_node = deployment.get("gpus_per_node", 8)

    slurm_logs = output_dir / "slurm_logs"
    inference_stdout = output_dir / "inference_artifacts" / "judge.stdout.log"
    outputs_dir = output_dir / "outputs"
    scored_dir = output_dir / "scored"
    comparisons_dir = output_dir / "comparisons"

    scoring_block = ""
    for model in config["models"]:
        label = model["label"]
        scoring_block += (
            f'\necho "=== Scoring: {label} ==="\n'
            f"python scripts/score_eval_outputs.py \\\n"
            f'  --endpoint "127.0.0.1:$PORT" \\\n'
            f"  --model {quote_shell(judge_model_name)} \\\n"
            f"  --input-path {quote_shell(str(outputs_dir / f'{label}.jsonl'))} \\\n"
            f"  --output-path {quote_shell(str(scored_dir / f'{label}.jsonl'))} \\\n"
            f"  --temperature {scoring.get('temperature', 0.2)} \\\n"
            f"  --max-tokens {scoring.get('max_tokens', 1024)} \\\n"
            f"  --parse-fail-score {scoring.get('parse_fail_score', 3.0)} \\\n"
            f"  --api-key {quote_shell(scoring.get('api_key', 'EMPTY'))} \\\n"
            f"  --overwrite\n"
        )

    rendering_block = ""
    for pair in config.get("comparisons", {}).get("pairs", []):
        a_label, b_label = pair
        rendering_block += (
            f'\necho "=== Rendering: {a_label} vs {b_label} ==="\n'
            f"python scripts/render_eval_comparison.py \\\n"
            f"  --baseline-path {quote_shell(str(scored_dir / f'{a_label}.jsonl'))} \\\n"
            f"  --candidate-path {quote_shell(str(scored_dir / f'{b_label}.jsonl'))} \\\n"
            f"  --output-path {quote_shell(str(comparisons_dir / f'{a_label}_vs_{b_label}.md'))} \\\n"
            f"  --baseline-label {quote_shell(a_label)} \\\n"
            f"  --candidate-label {quote_shell(b_label)}\n"
        )

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --partition={partition}
{f'#SBATCH --account={account}' if account else ''}
#SBATCH --time={time_limit}
#SBATCH --mem=0
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output={slurm_logs}/judge_%j.log
#SBATCH --error={slurm_logs}/judge_%j.err

set -euo pipefail

PROJECT_DIR={project_dir}
INFER_CONFIG_PATH={judge_infer_config_path}
PORT={server_port}

mkdir -p "{output_dir}" "{scored_dir}" "{comparisons_dir}" "{inference_stdout.parent}" "{slurm_logs}"

{SINGULARITY_PRELUDE}
export PROJECT_DIR
export INFER_CONFIG_PATH
export PORT

singularity exec --rocm "${{SINGULARITY_BIND_ARGS[@]}}" "$SIF" bash -c '
{CONTAINER_ENV_SETUP}
python -m prime_rl.inference.server \\
  @ "$INFER_CONFIG_PATH" \\
  --server.host 0.0.0.0 \\
  --server.port "$PORT" \\
  > {quote_shell(str(inference_stdout))} 2>&1 &
INFER_PID=$!

cleanup() {{
  kill "$INFER_PID" 2>/dev/null || true
  wait "$INFER_PID" 2>/dev/null || true
}}
trap cleanup EXIT

for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$INFER_PID" 2>/dev/null; then
    echo "FATAL: judge vLLM died during startup" >&2
    tail -n 100 {quote_shell(str(inference_stdout))} >&2 || true
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null
{scoring_block}{rendering_block}
echo "=== Done ==="
'
"""


def submit_sbatch(path: Path, dependency: str | None = None) -> int:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.extend(["--dependency", dependency])
    cmd.append(str(path))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return int(result.stdout.strip().split(";")[0])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config["output_dir"])
    inference_configs_dir = output_dir / "inference_configs"
    sbatch_dir = output_dir / "sbatch"
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    model_template = config.get("model_template", {})
    inference = config.get("inference", {})
    gpus_per_node = config.get("deployment", {}).get("gpus_per_node", 8)

    gen_sbatch_paths: list[tuple[str, Path]] = []
    skipped_labels: list[str] = []
    for model in config["models"]:
        label = model["label"]
        if model.get("skip_generation", False):
            skipped_labels.append(label)
            continue

        infer_config = build_model_inference_config(
            model_entry=model,
            model_template=model_template,
            inference=inference,
            gpus_per_node=gpus_per_node,
            output_dir=output_dir,
        )
        infer_path = inference_configs_dir / f"{label}.toml"
        write_toml(infer_path, infer_config)

        sbatch_path = sbatch_dir / f"generate_{label}.sbatch"
        sbatch_path.write_text(
            build_generation_sbatch(
                config=config,
                output_dir=output_dir,
                model=model,
                infer_config_path=infer_path,
                output_path=output_dir / "outputs" / f"{label}.jsonl",
            )
        )
        gen_sbatch_paths.append((label, sbatch_path))

    judge_infer_config = build_judge_inference_config(
        judge=config["judge"],
        gpus_per_node=gpus_per_node,
        output_dir=output_dir,
    )
    judge_infer_path = inference_configs_dir / "judge.toml"
    write_toml(judge_infer_path, judge_infer_config)

    judge_sbatch_path = sbatch_dir / "judge.sbatch"
    judge_sbatch_path.write_text(
        build_judge_sbatch(
            config=config,
            output_dir=output_dir,
            judge_infer_config_path=judge_infer_path,
        )
    )

    print(f"Wrote {len(gen_sbatch_paths)} generation sbatch files + 1 judge sbatch to {sbatch_dir}")
    for label in skipped_labels:
        print(f"  skipping generation for {label} (skip_generation=true; expecting existing outputs/{label}.jsonl)")

    if args.dry_run:
        for label, path in gen_sbatch_paths:
            print(f"  generate_{label}: {path}")
        print(f"  judge: {judge_sbatch_path}")
        return

    gen_job_ids: list[int] = []
    for label, path in gen_sbatch_paths:
        job_id = submit_sbatch(path)
        print(f"Generation {label}: submitted job {job_id}")
        gen_job_ids.append(job_id)

    if gen_job_ids:
        dependency = "afterok:" + ":".join(str(j) for j in gen_job_ids)
        judge_job_id = submit_sbatch(judge_sbatch_path, dependency=dependency)
        print(f"Judge: submitted job {judge_job_id} (dependency={dependency})")
    else:
        judge_job_id = submit_sbatch(judge_sbatch_path)
        print(f"Judge: submitted job {judge_job_id} (no gen jobs this run — all skipped)")


if __name__ == "__main__":
    main()
