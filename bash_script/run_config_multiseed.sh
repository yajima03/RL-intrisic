#!/usr/bin/env bash

# bash bash_script/run_config_multiseed.sh \
#   --env-config configs/env/sp_base.yaml \
#   --algo-config configs/algo/dqn.yaml \
#   --intrinsic-config configs/intrinsic/none.yaml \
#   --num-sims 10 \
#   --base-seed 100

set -euo pipefail

usage() {
  cat <<'EOH'
Usage:
  bash run_config_multiseed.sh \
    --env-config PATH \
    --algo-config PATH \
    --intrinsic-config PATH \
    [--num-sims N] \
    [--base-seed N] \
    [--seed-step N] \
    [--output-dir DIR] \
    [--group-name NAME] \
    [--sweep {none|env|algo|intrinsic}] \
    [--env-dir DIR] \
    [--algo-dir DIR] \
    [--intrinsic-dir DIR] \
    [--dry-run] \
    [-- EXTRA ARGS FOR train.py ...]

Examples:
  # Fixed env/algo/intrinsic, run 10 simulations with seeds 0..9
  bash run_config_multiseed.sh \
    --env-config configs/env/sp_base.yaml \
    --algo-config configs/algo/dqn.yaml \
    --intrinsic-config configs/intrinsic/none.yaml \
    --num-sims 10

  # Start seeds at 100 and increment by 5
  bash run_config_multiseed.sh \
    --env-config configs/env/sp_base.yaml \
    --algo-config configs/algo/dqn.yaml \
    --intrinsic-config configs/intrinsic/pglp_local.yaml \
    --num-sims 10 \
    --base-seed 100 \
    --seed-step 5

  # Sweep all intrinsic configs, and for each config run 3 seeds
  bash run_config_multiseed.sh \
    --sweep intrinsic \
    --env-config configs/env/sp_base.yaml \
    --algo-config configs/algo/dqn.yaml \
    --num-sims 3

  # Pass extra arguments to train.py after "--"
  bash run_config_multiseed.sh \
    --env-config configs/env/sp_base.yaml \
    --algo-config configs/algo/dqn.yaml \
    --intrinsic-config configs/intrinsic/none.yaml \
    --num-sims 5 \
    -- --total-timesteps 50000

Options:
  --env-config       Fixed env config path (required unless --sweep env)
  --algo-config      Fixed algo config path (required unless --sweep algo)
  --intrinsic-config Fixed intrinsic config path (required unless --sweep intrinsic)

  --num-sims         Number of simulations to run with different seeds (default: 1)
  --base-seed        First seed value (default: 0)
  --seed-step        Increment added for each next simulation (default: 1)

  --output-dir       Base output directory (default: outputs)
  --group-name       Name of the parent folder created under output-dir
                     If omitted, it is auto-generated.

  --sweep            Which config type to sweep: none, env, algo, intrinsic (default: none)
  --env-dir          Directory containing env configs (default: configs/env)
  --algo-dir         Directory containing algo configs (default: configs/algo)
  --intrinsic-dir    Directory containing intrinsic configs (default: configs/intrinsic)

  --dry-run          Print commands only, do not execute

  --                 Everything after this is passed through to train.py
EOH
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "File not found: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Directory not found: $path"
}

basename_no_ext() {
  local path="$1"
  local base
  base="$(basename "$path")"
  base="${base%.yaml}"
  base="${base%.yml}"
  echo "$base"
}

timestamp_now() {
  date +"%Y%m%d-%H%M%S"
}

list_yaml_files() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) | sort
}

safe_name() {
  local s="$1"
  s="${s// /_}"
  s="${s//\//_}"
  s="${s//:/_}"
  echo "$s"
}

build_group_name() {
  local env_name="$1"
  local algo_name="$2"
  local intr_name="$3"
  local sweep_name="$4"
  local n_sims="$5"
  local ts="$6"

  if [[ "$sweep_name" == "none" ]]; then
    echo "multiseed_${env_name}_${algo_name}_${intr_name}_${n_sims}sim_${ts}"
  else
    echo "sweep_${sweep_name}_${env_name}_${algo_name}_${intr_name}_${n_sims}sim_${ts}"
  fi
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
  fi
}

ENV_CONFIG=""
ALGO_CONFIG=""
INTRINSIC_CONFIG=""
NUM_SIMS=1
BASE_SEED=0
SEED_STEP=1
OUTPUT_DIR="outputs"
GROUP_NAME=""
SWEEP="none"
ENV_DIR="configs/env"
ALGO_DIR="configs/algo"
INTRINSIC_DIR="configs/intrinsic"
DRY_RUN=0
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --num-sims)
      NUM_SIMS="$2"
      shift 2
      ;;
    --base-seed)
      BASE_SEED="$2"
      shift 2
      ;;
    --seed-step)
      SEED_STEP="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --group-name)
      GROUP_NAME="$2"
      shift 2
      ;;
    --sweep)
      SWEEP="$2"
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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        PASSTHROUGH_ARGS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

case "$SWEEP" in
  none|env|algo|intrinsic) ;;
  *) die "--sweep must be one of: none, env, algo, intrinsic" ;;
esac

[[ "$NUM_SIMS" =~ ^[0-9]+$ ]] || die "--num-sims must be a non-negative integer"
[[ "$BASE_SEED" =~ ^-?[0-9]+$ ]] || die "--base-seed must be an integer"
[[ "$SEED_STEP" =~ ^-?[0-9]+$ ]] || die "--seed-step must be an integer"
(( NUM_SIMS >= 1 )) || die "--num-sims must be >= 1"

mkdir -p "$OUTPUT_DIR"

if [[ "$SWEEP" != "env" ]]; then
  [[ -n "$ENV_CONFIG" ]] || die "--env-config is required unless --sweep env"
  require_file "$ENV_CONFIG"
fi
if [[ "$SWEEP" != "algo" ]]; then
  [[ -n "$ALGO_CONFIG" ]] || die "--algo-config is required unless --sweep algo"
  require_file "$ALGO_CONFIG"
fi
if [[ "$SWEEP" != "intrinsic" ]]; then
  [[ -n "$INTRINSIC_CONFIG" ]] || die "--intrinsic-config is required unless --sweep intrinsic"
  require_file "$INTRINSIC_CONFIG"
fi

if [[ "$SWEEP" == "env" || "$SWEEP" == "algo" || "$SWEEP" == "intrinsic" ]]; then
  require_dir "$ENV_DIR"
  require_dir "$ALGO_DIR"
  require_dir "$INTRINSIC_DIR"
fi

ENV_LIST=()
ALGO_LIST=()
INTR_LIST=()

if [[ "$SWEEP" == "env" ]]; then
  while IFS= read -r f; do ENV_LIST+=("$f"); done < <(list_yaml_files "$ENV_DIR")
  (( ${#ENV_LIST[@]} > 0 )) || die "No YAML files found in $ENV_DIR"
else
  ENV_LIST=("$ENV_CONFIG")
fi

if [[ "$SWEEP" == "algo" ]]; then
  while IFS= read -r f; do ALGO_LIST+=("$f"); done < <(list_yaml_files "$ALGO_DIR")
  (( ${#ALGO_LIST[@]} > 0 )) || die "No YAML files found in $ALGO_DIR"
else
  ALGO_LIST=("$ALGO_CONFIG")
fi

if [[ "$SWEEP" == "intrinsic" ]]; then
  while IFS= read -r f; do INTR_LIST+=("$f"); done < <(list_yaml_files "$INTRINSIC_DIR")
  (( ${#INTR_LIST[@]} > 0 )) || die "No YAML files found in $INTRINSIC_DIR"
else
  INTR_LIST=("$INTRINSIC_CONFIG")
fi

FIXED_ENV_NAME="$(basename_no_ext "${ENV_LIST[0]}")"
FIXED_ALGO_NAME="$(basename_no_ext "${ALGO_LIST[0]}")"
FIXED_INTR_NAME="$(basename_no_ext "${INTR_LIST[0]}")"

if [[ -z "$GROUP_NAME" ]]; then
  GROUP_NAME="$(build_group_name \
    "$(safe_name "$FIXED_ENV_NAME")" \
    "$(safe_name "$FIXED_ALGO_NAME")" \
    "$(safe_name "$FIXED_INTR_NAME")" \
    "$SWEEP" \
    "$NUM_SIMS" \
    "$(timestamp_now)")"
fi

GROUP_DIR="${OUTPUT_DIR%/}/$GROUP_NAME"
mkdir -p "$GROUP_DIR"

{
  echo "group_name=$GROUP_NAME"
  echo "sweep=$SWEEP"
  echo "num_sims=$NUM_SIMS"
  echo "base_seed=$BASE_SEED"
  echo "seed_step=$SEED_STEP"
  echo "created_at=$(date +"%Y-%m-%d %H:%M:%S")"
  if [[ -n "$ENV_CONFIG" ]]; then echo "fixed_env_config=$ENV_CONFIG"; fi
  if [[ -n "$ALGO_CONFIG" ]]; then echo "fixed_algo_config=$ALGO_CONFIG"; fi
  if [[ -n "$INTRINSIC_CONFIG" ]]; then echo "fixed_intrinsic_config=$INTRINSIC_CONFIG"; fi
  if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    printf 'passthrough_args='
    printf '%q ' "${PASSTHROUGH_ARGS[@]}"
    printf '\n'
  fi
} > "$GROUP_DIR/group_info.txt"

copy_if_exists "${ENV_LIST[0]}" "$GROUP_DIR/_fixed_or_first_env_config.yaml"
copy_if_exists "${ALGO_LIST[0]}" "$GROUP_DIR/_fixed_or_first_algo_config.yaml"
copy_if_exists "${INTR_LIST[0]}" "$GROUP_DIR/_fixed_or_first_intrinsic_config.yaml"

MANIFEST_CSV="$GROUP_DIR/run_manifest.csv"
echo "config_group,sim_index,seed,env_config,algo_config,intrinsic_config,run_name,run_dir" > "$MANIFEST_CSV"

echo "[INFO] Group directory : $GROUP_DIR"
echo "[INFO] Sweep          : $SWEEP"
echo "[INFO] Num sims       : $NUM_SIMS"
echo "[INFO] Base seed      : $BASE_SEED"
echo "[INFO] Seed step      : $SEED_STEP"
if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
  echo "[INFO] Extra train.py args: ${PASSTHROUGH_ARGS[*]}"
fi
[[ "$DRY_RUN" -eq 1 ]] && echo "[INFO] Dry run enabled"

run_one() {
  local cfg_group_name="$1"
  local env_cfg="$2"
  local algo_cfg="$3"
  local intr_cfg="$4"
  local sim_index="$5"
  local seed="$6"

  local cfg_dir="$GROUP_DIR/$cfg_group_name"
  mkdir -p "$cfg_dir"

  local run_name
  run_name="$(printf "sim_%03d_seed_%d" "$sim_index" "$seed")"

  local cmd=(
    uv run python -m src.training.train
    --env-config "$env_cfg"
    --algo-config "$algo_cfg"
    --intrinsic-config "$intr_cfg"
    --output-dir "$cfg_dir"
    --run-name "$run_name"
    --seed "$seed"
  )

  if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    cmd+=("${PASSTHROUGH_ARGS[@]}")
  fi

  local expected_run_dir="$cfg_dir/$run_name"

  echo
  echo "============================================================"
  echo "[INFO] Config group   : $cfg_group_name"
  echo "[INFO] Sim index      : $sim_index / $NUM_SIMS"
  echo "[INFO] Seed           : $seed"
  echo "[INFO] Env config     : $env_cfg"
  echo "[INFO] Algo config    : $algo_cfg"
  echo "[INFO] Intrinsic cfg  : $intr_cfg"
  echo "[INFO] Run dir        : $expected_run_dir"
  echo "[INFO] Command        : ${cmd[*]}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$cfg_group_name" \
    "$sim_index" \
    "$seed" \
    "$env_cfg" \
    "$algo_cfg" \
    "$intr_cfg" \
    "$run_name" \
    "$expected_run_dir" >> "$MANIFEST_CSV"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  "${cmd[@]}"
}

if [[ "$SWEEP" == "none" ]]; then
  CFG_GROUP_NAME="$(safe_name "$(basename_no_ext "$ENV_CONFIG")")__$(safe_name "$(basename_no_ext "$ALGO_CONFIG")")__$(safe_name "$(basename_no_ext "$INTRINSIC_CONFIG")")"
  sim=1
  while [[ "$sim" -le "$NUM_SIMS" ]]; do
    seed=$(( BASE_SEED + (sim - 1) * SEED_STEP ))
    run_one "$CFG_GROUP_NAME" "$ENV_CONFIG" "$ALGO_CONFIG" "$INTRINSIC_CONFIG" "$sim" "$seed"
    sim=$((sim + 1))
  done
else
  for env_cfg in "${ENV_LIST[@]}"; do
    for algo_cfg in "${ALGO_LIST[@]}"; do
      for intr_cfg in "${INTR_LIST[@]}"; do
        CFG_GROUP_NAME="$(safe_name "$(basename_no_ext "$env_cfg")")__$(safe_name "$(basename_no_ext "$algo_cfg")")__$(safe_name "$(basename_no_ext "$intr_cfg")")"
        sim=1
        while [[ "$sim" -le "$NUM_SIMS" ]]; do
          seed=$(( BASE_SEED + (sim - 1) * SEED_STEP ))
          run_one "$CFG_GROUP_NAME" "$env_cfg" "$algo_cfg" "$intr_cfg" "$sim" "$seed"
          sim=$((sim + 1))
        done
      done
    done
  done
fi

echo
echo "[INFO] Finished."
echo "[INFO] Group directory: $GROUP_DIR"
echo "[INFO] Manifest CSV   : $MANIFEST_CSV"
