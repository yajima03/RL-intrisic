#!/bin/bash


#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_config_sweep.sh \
    --sweep {env|algo|intrinsic} \
    [--env-config PATH] \
    [--algo-config PATH] \
    [--intrinsic-config PATH] \
    [--env-dir DIR] \
    [--algo-dir DIR] \
    [--intrinsic-dir DIR] \
    [--output-dir DIR] \
    [--extra-args "..."] \
    [--dry-run]

Examples:
  # Sweep all intrinsic configs while fixing env/algo
  bash run_config_sweep.sh \
    --sweep intrinsic \
    --env-config configs/env/sp_base.yaml \
    --algo-config configs/algo/dqn.yaml

  # Sweep all env configs while fixing algo/intrinsic
  bash run_config_sweep.sh \
    --sweep env \
    --algo-config configs/algo/dqn.yaml \
    --intrinsic-config configs/intrinsic/pglp_local.yaml

  # Sweep all algo configs and pass extra args to train.py
  bash run_config_sweep.sh \
    --sweep algo \
    --env-config configs/env/sp_base.yaml \
    --intrinsic-config configs/intrinsic/count.yaml \
    --extra-args "--total-timesteps 50000"

Options:
  --sweep            Which config type to sweep: env, algo, intrinsic
  --env-config       Fixed env config path (required unless --sweep env)
  --algo-config      Fixed algo config path (required unless --sweep algo)
  --intrinsic-config Fixed intrinsic config path (required unless --sweep intrinsic)
  --env-dir          Directory containing env configs (default: configs/env)
  --algo-dir         Directory containing algo configs (default: configs/algo)
  --intrinsic-dir    Directory containing intrinsic configs (default: configs/intrinsic)
  --output-dir       Base output directory passed to train.py (default: outputs)
  --extra-args       Extra CLI args appended to train.py command
  --dry-run          Print commands only, do not execute
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] File not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Directory not found: $path" >&2
    exit 1
  fi
}

list_yaml_files() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) | sort
}

basename_no_ext() {
  local path="$1"
  local base
  base="$(basename "$path")"
  base="${base%.yaml}"
  base="${base%.yml}"
  echo "$base"
}

SWEEP=""
ENV_CONFIG=""
ALGO_CONFIG=""
INTRINSIC_CONFIG=""
ENV_DIR="configs/env"
ALGO_DIR="configs/algo"
INTRINSIC_DIR="configs/intrinsic"
OUTPUT_DIR="outputs"
EXTRA_ARGS=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep)
      SWEEP="$2"
      shift 2
      ;;
    --env-config)
      ENV_CONFIG="$2"
      shift 2
      ;;
    --algo-config)
      ALGO_CONFIG="$2"
      shift 2
      ;;
    --intrinsic-config)
      INTRINSIC_CONFIG="$2"
      shift 2
      ;;
    --env-dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --algo-dir)
      ALGO_DIR="$2"
      shift 2
      ;;
    --intrinsic-dir)
      INTRINSIC_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --extra-args)
      EXTRA_ARGS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SWEEP" ]]; then
  echo "[ERROR] --sweep is required." >&2
  usage
  exit 1
fi

case "$SWEEP" in
  env|algo|intrinsic) ;;
  *)
    echo "[ERROR] --sweep must be one of: env, algo, intrinsic" >&2
    exit 1
    ;;
esac

require_dir "$ENV_DIR"
require_dir "$ALGO_DIR"
require_dir "$INTRINSIC_DIR"
mkdir -p "$OUTPUT_DIR"

if [[ "$SWEEP" != "env" ]]; then
  if [[ -z "$ENV_CONFIG" ]]; then
    echo "[ERROR] --env-config is required unless --sweep env" >&2
    exit 1
  fi
  require_file "$ENV_CONFIG"
fi

if [[ "$SWEEP" != "algo" ]]; then
  if [[ -z "$ALGO_CONFIG" ]]; then
    echo "[ERROR] --algo-config is required unless --sweep algo" >&2
    exit 1
  fi
  require_file "$ALGO_CONFIG"
fi

if [[ "$SWEEP" != "intrinsic" ]]; then
  if [[ -z "$INTRINSIC_CONFIG" ]]; then
    echo "[ERROR] --intrinsic-config is required unless --sweep intrinsic" >&2
    exit 1
  fi
  require_file "$INTRINSIC_CONFIG"
fi

case "$SWEEP" in
  env)
    SWEEP_DIR="$ENV_DIR"
    ;;
  algo)
    SWEEP_DIR="$ALGO_DIR"
    ;;
  intrinsic)
    SWEEP_DIR="$INTRINSIC_DIR"
    ;;
esac

SWEEP_FILES=()
while IFS= read -r file; do
  SWEEP_FILES+=("$file")
done < <(list_yaml_files "$SWEEP_DIR")

if [[ ${#SWEEP_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No YAML files found in $SWEEP_DIR" >&2
  exit 1
fi

echo "[INFO] Sweep type      : $SWEEP"
echo "[INFO] Sweep directory : $SWEEP_DIR"
echo "[INFO] Number of files : ${#SWEEP_FILES[@]}"
echo "[INFO] Output dir      : $OUTPUT_DIR"
[[ -n "$ENV_CONFIG" ]] && echo "[INFO] Fixed env config       : $ENV_CONFIG"
[[ -n "$ALGO_CONFIG" ]] && echo "[INFO] Fixed algo config      : $ALGO_CONFIG"
[[ -n "$INTRINSIC_CONFIG" ]] && echo "[INFO] Fixed intrinsic config : $INTRINSIC_CONFIG"
[[ -n "$EXTRA_ARGS" ]] && echo "[INFO] Extra args             : $EXTRA_ARGS"
[[ "$DRY_RUN" -eq 1 ]] && echo "[INFO] Dry run enabled"

for cfg in "${SWEEP_FILES[@]}"; do
  CUR_ENV="$ENV_CONFIG"
  CUR_ALGO="$ALGO_CONFIG"
  CUR_INTRINSIC="$INTRINSIC_CONFIG"

  case "$SWEEP" in
    env)
      CUR_ENV="$cfg"
      ;;
    algo)
      CUR_ALGO="$cfg"
      ;;
    intrinsic)
      CUR_INTRINSIC="$cfg"
      ;;
  esac

  ENV_NAME="$(basename_no_ext "$CUR_ENV")"
  ALGO_NAME="$(basename_no_ext "$CUR_ALGO")"
  INTR_NAME="$(basename_no_ext "$CUR_INTRINSIC")"
  RUN_NAME="${SWEEP}_${ENV_NAME}_${ALGO_NAME}_${INTR_NAME}"

  CMD=(
    uv run python -m src.training.train
    --env-config "$CUR_ENV"
    --algo-config "$CUR_ALGO"
    --intrinsic-config "$CUR_INTRINSIC"
    --output-dir "$OUTPUT_DIR"
    --run-name "$RUN_NAME"
  )

  echo
  echo "============================================================"
  echo "[INFO] Running sweep target: $cfg"
  echo "[INFO] Run name           : $RUN_NAME"
  echo "[INFO] Env config         : $CUR_ENV"
  echo "[INFO] Algo config        : $CUR_ALGO"
  echo "[INFO] Intrinsic config   : $CUR_INTRINSIC"
  echo "[INFO] Command            : ${CMD[*]} ${EXTRA_ARGS}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2086
    eval "${CMD[*]} ${EXTRA_ARGS}"
  else
    "${CMD[@]}"
  fi
done

echo
echo "[INFO] Sweep finished."
