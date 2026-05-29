#!/usr/bin/env bash
set -euo pipefail

RUN_LABEL="${RUN_LABEL:-mega5x3}"
LOG_DIR="${LOG_DIR:-runs/gcp_logs}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/phase2_${RUN_LABEL}_STATUS.md}"
PHASE2_A_TARGET="${PHASE2_A_TARGET:-9600}"

phase2_prefixes=(
  "phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32"
  "phase2_swe_grep_koopman_dmd_c_behavioral_v3_qwen_tooluse_float32"
  "phase2_swe_grep_pca_proxy_behavioral_v3_qwen_tooluse_float32"
  "phase2_math500_parametric_b_behavioral_v3_qwen_float32"
)

phase2_a_prefix="${phase2_prefixes[0]}"

process_names=(
  "A"
  "C"
  "PCA"
  "MATH"
)

process_patterns=(
  "controllability.experiments.phase1_5_verifier_smoke.*phase2_swe_grep_average_controllability_a"
  "controllability.experiments.phase1_5_verifier_smoke.*phase2_swe_grep_koopman_dmd_c"
  "controllability.experiments.phase1_5_verifier_smoke.*phase2_swe_grep_pca_proxy"
  "controllability.experiments.phase1_5_verifier_smoke.*phase2_math500_parametric_b"
)

line_count() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path" | tr -d " "
  else
    printf "0\n"
  fi
}

mtime_epoch() {
  local path="$1"
  local mtime
  if [[ ! -f "$path" ]]; then
    printf "0\n"
    return
  fi
  if mtime="$(stat -c %Y "$path" 2>/dev/null)"; then
    printf "%s\n" "$mtime"
  else
    stat -f %m "$path"
  fi
}

process_count() {
  local pattern="$1"
  local matches
  matches="$(pgrep -af "$pattern" 2>/dev/null || true)"
  if [[ -z "$matches" ]]; then
    printf "0\n"
    return
  fi
  printf "%s\n" "$matches" | awk '!/pgrep -af/ { count++ } END { print count + 0 }'
}

mkdir -p "$LOG_DIR"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf "# Phase 2 %s Status\n\n" "$RUN_LABEL" > "$STATUS_FILE"
fi

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
now_epoch="$(date -u +%s)"
git_head="$(git rev-parse --short HEAD 2>/dev/null || printf "unknown")"
master_pid_file="${LOG_DIR}/run_phase2_${RUN_LABEL}_master.pid"
master_state="not found"
master_elapsed_s=""
if [[ -f "$master_pid_file" ]]; then
  master_pid="$(cat "$master_pid_file")"
  master_state="$(ps -p "$master_pid" -o pid=,etime=,cmd= 2>/dev/null || printf "not running")"
  master_elapsed_s="$(ps -p "$master_pid" -o etimes= 2>/dev/null | tr -d " " || true)"
  if [[ ! "$master_elapsed_s" =~ ^[0-9]+$ ]]; then
    master_elapsed_s=""
  fi
fi

total=0
failed=0
stage_tmp="$(mktemp)"
shard_tmp="$(mktemp)"
worker_tmp="$(mktemp)"
merged_tmp="$(mktemp)"
trap 'rm -f "$stage_tmp" "$shard_tmp" "$worker_tmp" "$merged_tmp"' EXIT

for idx in "${!process_names[@]}"; do
  printf "%s=%s\n" "${process_names[$idx]}" "$(process_count "${process_patterns[$idx]}")" >> "$worker_tmp"
done

shopt -s nullglob
for prefix in "${phase2_prefixes[@]}"; do
  subtotal=0
  failed_subtotal=0
  seen=0
  newest=0
  oldest=0
  min=999999
  max=0
  for run_dir in runs/${prefix}_${RUN_LABEL}_shard*; do
    [[ -d "$run_dir" ]] || continue
    seen=1
    trajectories="$(line_count "${run_dir}/trajectories.jsonl")"
    failures="$(line_count "${run_dir}/failed_jobs.jsonl")"
    mtime="$(mtime_epoch "${run_dir}/trajectories.jsonl")"
    if [[ "$mtime" -gt "$newest" ]]; then
      newest="$mtime"
    fi
    if [[ "$mtime" -gt 0 ]] && { [[ "$oldest" -eq 0 ]] || [[ "$mtime" -lt "$oldest" ]]; }; then
      oldest="$mtime"
    fi
    last_write_age_s="na"
    if [[ "$mtime" -gt 0 ]]; then
      last_write_age_s=$((now_epoch - mtime))
    fi
    subtotal=$((subtotal + trajectories))
    failed_subtotal=$((failed_subtotal + failures))
    if [[ "$trajectories" -lt "$min" ]]; then
      min="$trajectories"
    fi
    if [[ "$trajectories" -gt "$max" ]]; then
      max="$trajectories"
    fi
    printf "%s\n" "- $(basename "$run_dir"): trajectories=${trajectories}, failed=${failures}, last_write_age_s=${last_write_age_s}" >> "$shard_tmp"
  done
  if [[ "$seen" -eq 1 ]]; then
    total=$((total + subtotal))
    failed=$((failed + failed_subtotal))
    last_write_age_s="na"
    oldest_shard_write_age_s="na"
    shard_spread="na"
    if [[ "$newest" -gt 0 ]]; then
      last_write_age_s=$((now_epoch - newest))
    fi
    if [[ "$oldest" -gt 0 ]]; then
      oldest_shard_write_age_s=$((now_epoch - oldest))
    fi
    if [[ "$max" -gt 0 ]]; then
      shard_spread=$((max - min))
    fi
    progress_extra=""
    if [[ "$prefix" == "$phase2_a_prefix" ]]; then
      remaining=$((PHASE2_A_TARGET - subtotal))
      if [[ "$remaining" -lt 0 ]]; then
        remaining=0
      fi
      progress_extra=", remaining=${remaining}"
      if [[ -n "$master_elapsed_s" ]] && [[ "$subtotal" -gt 0 ]]; then
        rate="$(awk -v generated="$subtotal" -v elapsed="$master_elapsed_s" 'BEGIN { if (elapsed > 0) printf "%.4f", generated / elapsed; else print "0" }')"
        eta_s="$(awk -v remaining="$remaining" -v rate="$rate" 'BEGIN { if (rate > 0) printf "%.0f", remaining / rate; else print "0" }')"
        progress_extra="${progress_extra}, rate_traj_per_s=${rate}, eta_s=${eta_s}"
      fi
    fi
    printf "%s\n" "- ${prefix}: subtotal=${subtotal}, failed=${failed_subtotal}, last_write_age_s=${last_write_age_s}, oldest_shard_write_age_s=${oldest_shard_write_age_s}, shard_spread=${shard_spread}${progress_extra}" >> "$stage_tmp"
  fi

  merged_dir="runs/${prefix}_merged"
  if [[ -d "$merged_dir" ]]; then
    trajectories="$(line_count "${merged_dir}/trajectories.jsonl")"
    printf "%s\n" "- ${merged_dir}: trajectories=${trajectories}" >> "$merged_tmp"
  fi
done
shopt -u nullglob

error_tail="$(
  grep -R "OutOfMemory\\|Traceback\\|ERROR\\|Expected all tensors\\|ModelError\\|No space left" "${LOG_DIR}"/*"${RUN_LABEL}"*.err 2>/dev/null | tail -5 || true
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
  printf "**Workers**: "
  awk 'BEGIN { sep = "" } { printf "%s%s", sep, $0; sep = ", " } END { printf "\n" }' "$worker_tmp"
  printf "**Master**: %s\n\n" "$master_state"

  printf "**Stages**:\n"
  if [[ -s "$stage_tmp" ]]; then
    cat "$stage_tmp"
  else
    printf "No matching stage shard directories found.\n"
  fi

  printf "\n**Merged Outputs**:\n"
  if [[ -s "$merged_tmp" ]]; then
    cat "$merged_tmp"
  else
    printf "No merged Phase 2 outputs found.\n"
  fi

  printf "\n**Shards**:\n"
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
  printf "\n**Disk**:\n"
  df -h / | tail -1
  du -sh runs 2>/dev/null || true
  printf "\n"
} >> "$STATUS_FILE"

tail -60 "$STATUS_FILE"
