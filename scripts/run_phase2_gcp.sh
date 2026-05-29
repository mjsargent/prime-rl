#!/usr/bin/env bash
set -euo pipefail

NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_GROUPS="${GPU_GROUPS:-}"
RUN_LABEL="${RUN_LABEL:-}"
LOG_DIR="${LOG_DIR:-runs/gcp_logs}"
export PATH="${HOME}/.local/bin:${PATH}"
mkdir -p "$LOG_DIR"

phase2_configs=(
  "configs/controllability/experiments/phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32.yaml"
  "configs/controllability/experiments/phase2_swe_grep_koopman_dmd_c_behavioral_v3_qwen_tooluse_float32.yaml"
  "configs/controllability/experiments/phase2_swe_grep_pca_proxy_behavioral_v3_qwen_tooluse_float32.yaml"
  "configs/controllability/experiments/phase2_math500_parametric_b_behavioral_v3_qwen_float32.yaml"
)

stage5_configs=(
  "configs/controllability/experiments/stage5_phase2_swe_grep_comparators_behavioral_v3.yaml"
  "configs/controllability/experiments/stage5_phase2_math500_parametric_b_behavioral_v3.yaml"
)

stage6_configs=(
  "configs/controllability/experiments/stage6_phase2_swe_grep_parametric_b_behavioral_v3_comparator.yaml"
  "configs/controllability/experiments/stage6_phase2_swe_grep_average_controllability_a_behavioral_v3.yaml"
  "configs/controllability/experiments/stage6_phase2_swe_grep_koopman_dmd_c_behavioral_v3.yaml"
  "configs/controllability/experiments/stage6_phase2_swe_grep_pca_proxy_behavioral_v3.yaml"
  "configs/controllability/experiments/stage6_phase2_math500_parametric_b_behavioral_v3.yaml"
)

run_id_for_config() {
  uv run --locked python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["run_id"])' "$1"
}

run_phase2_config() {
  local config="$1"
  local gpu_groups=()
  if [[ -n "$GPU_GROUPS" ]]; then
    IFS=';' read -r -a gpu_groups <<< "$GPU_GROUPS"
    NUM_SHARDS="${#gpu_groups[@]}"
  else
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
      gpu_groups+=("$shard")
    done
  fi
  local run_label="${RUN_LABEL:-gcp${NUM_SHARDS}}"
  local run_id base_run_id output_run_id
  run_id="$(run_id_for_config "$config")"
  base_run_id="${run_id}_${run_label}"
  output_run_id="${run_id}_merged"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase2 start ${run_id}"

  local pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_padded
    shard_padded="$(printf "%02d" "$shard")"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_groups[$shard]}"
      uv run --locked python -m controllability.experiments.phase1_5_verifier_smoke \
        --config "$config" \
        --job-shard-index "$shard" \
        --num-job-shards "$NUM_SHARDS" \
        --run-id-suffix "_${run_label}_shard${shard_padded}"
    ) >"${LOG_DIR}/${base_run_id}_shard${shard_padded}.out" 2>"${LOG_DIR}/${base_run_id}_shard${shard_padded}.err" &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase2 failed ${run_id}" >&2
    return "$status"
  fi

  uv run --locked python -m controllability.reports.merge_phase2_shards \
    --base-run-id "$base_run_id" \
    --output-run-id "$output_run_id" \
    --num-shards "$NUM_SHARDS"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] phase2 merged ${output_run_id}"
}

for config in "${phase2_configs[@]}"; do
  run_phase2_config "$config"
done

for config in "${stage5_configs[@]}"; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage5 start ${config}"
  uv run --locked python -m controllability.experiments.stage5_identifiability --config "$config"
done

for config in "${stage6_configs[@]}"; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage6 start ${config}"
  uv run --locked python -m controllability.experiments.stage6_geometric_behavioral_correlation --config "$config"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Phase 2 comparator/replication batch complete"
