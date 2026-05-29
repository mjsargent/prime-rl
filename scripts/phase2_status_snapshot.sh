#!/usr/bin/env bash
set -euo pipefail

RUN_LABEL="${RUN_LABEL:-mega5x3}"
LOG_DIR="${LOG_DIR:-runs/gcp_logs}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/phase2_${RUN_LABEL}_STATUS.md}"

phase2_prefixes=(
  "phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32"
  "phase2_swe_grep_koopman_dmd_c_behavioral_v3_qwen_tooluse_float32"
  "phase2_swe_grep_pca_proxy_behavioral_v3_qwen_tooluse_float32"
  "phase2_math500_parametric_b_behavioral_v3_qwen_float32"
)

mkdir -p "$LOG_DIR"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf "# Phase 2 %s Status\n\n" "$RUN_LABEL" > "$STATUS_FILE"
fi

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_head="$(git rev-parse --short HEAD 2>/dev/null || printf "unknown")"
master_pid_file="${LOG_DIR}/run_phase2_${RUN_LABEL}_master.pid"
master_state="not found"
if [[ -f "$master_pid_file" ]]; then
  master_pid="$(cat "$master_pid_file")"
  master_state="$(ps -p "$master_pid" -o pid=,etime=,cmd= 2>/dev/null || printf "not running")"
fi

total=0
failed=0
shard_tmp="$(mktemp)"
trap 'rm -f "$shard_tmp"' EXIT

shopt -s nullglob
for prefix in "${phase2_prefixes[@]}"; do
  subtotal=0
  seen=0
  for run_dir in runs/${prefix}_${RUN_LABEL}_shard*; do
    [[ -d "$run_dir" ]] || continue
    seen=1
    trajectories=0
    failures=0
    if [[ -f "${run_dir}/trajectories.jsonl" ]]; then
      trajectories="$(wc -l < "${run_dir}/trajectories.jsonl")"
    fi
    if [[ -f "${run_dir}/failed_jobs.jsonl" ]]; then
      failures="$(wc -l < "${run_dir}/failed_jobs.jsonl")"
    fi
    subtotal=$((subtotal + trajectories))
    failed=$((failed + failures))
    printf "%s\n" "- $(basename "$run_dir"): trajectories=${trajectories}, failed=${failures}" >> "$shard_tmp"
  done
  if [[ "$seen" -eq 1 ]]; then
    total=$((total + subtotal))
    printf "%s\n" "- ${prefix}: subtotal=${subtotal}" >> "$shard_tmp"
  fi
done
shopt -u nullglob

error_tail="$(
  grep -R "OutOfMemory\\|Traceback\\|ERROR\\|Expected all tensors\\|ModelError" "${LOG_DIR}"/*"${RUN_LABEL}"*.err 2>/dev/null | tail -5 || true
)"
health="Healthy"
if [[ "$failed" -ne 0 ]] || [[ -n "$error_tail" ]]; then
  health="Degraded"
fi

{
  printf "## %s\n\n" "$timestamp"
  printf "**Health**: %s\n" "$health"
  printf "**HEAD**: %s\n" "$git_head"
  printf "**Progress**: %s total trajectories across Phase 2 %s shards\n" "$total" "$RUN_LABEL"
  printf "**Failures**: %s failed jobs\n" "$failed"
  printf "**Master**: %s\n\n" "$master_state"
  printf "**Shards**:\n"
  if [[ -s "$shard_tmp" ]]; then
    cat "$shard_tmp"
  else
    printf "No matching shard directories found.\n"
  fi
  printf "\n**Errors**:\n"
  if [[ -n "$error_tail" ]]; then
    printf "%s\n" "$error_tail"
  else
    printf "No OOM/traceback/tensor-device/ModelError matches.\n"
  fi
  printf "\n"
} >> "$STATUS_FILE"

tail -60 "$STATUS_FILE"
