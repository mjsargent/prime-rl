# Phase 2 GCP Repair Log

## Failure Observed

The first GCP Phase 2 comparator batch stopped on
`phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32`.
The initial launch used four 2-GPU A100-40GB workers:

```bash
GPU_GROUPS='0,1;2,3;4,5;6,7' RUN_LABEL=gcp4x2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/run_phase2_gcp.sh
```

Each shard wrote 1200 trajectories, but most were empty verifier-error
rollouts. The shard logs contained repeated CUDA OOMs once multi-turn prompts
grew:

```text
OutOfMemoryError: CUDA out of memory ... GPU ... 39.49 GiB ... process has
39.4x GiB memory in use
```

The run also exposed a driver bug: `verifiers` returns an error-shaped rollout
for `ModelError` instead of raising. The Phase 2 driver was recording those as
zero-token trajectories rather than failed jobs.

## Repairs

- Treat verifier outputs with top-level `error` as failed jobs after configured
  rollout retries.
- Add `max_rollout_retries`, `empty_cache_between_jobs`, and
  `generate_use_cache` controls to Phase 2 configs.
- Clear Python and CUDA allocator state between jobs.
- Run generation under `torch.inference_mode()`.
- Clear captured autograd tensors after VJP computation.
- Make Phase 2 gates require completeness, no failed jobs, tool-call presence,
  encoder activity, and linearization. Reward and trajectory length remain
  reported outcomes, not stop-the-batch smoke gates.
- Fix multi-GPU chart projection for 4-GPU A100 device maps by placing chart
  tensors on the readout residual device.
- Bound Phase 2 linearization diagnostics with `linearization_diagnostic_max_jobs: 16`
  so each shard records per-turn linearization fidelity on a sample instead of
  running two extra patched forwards for every trajectory.
- Harden `scripts/run_phase2_gcp.sh` by prefixing `~/.local/bin` onto `PATH`.
  The GCP VM found `uv` in interactive shells, but a non-login `nohup` launch
  failed with `uv: command not found`.

## Validation

Remote GCP validation on `controllability-phase2-a2`:

```text
uv run ruff check ...: passed
uv run pytest tests/controllability -q: 7 passed, 1 skipped
```

A 4-GPU probe was launched with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python -m controllability.experiments.phase1_5_verifier_smoke \
  --config configs/controllability/experiments/phase2_probe_swe_grep_average_controllability_a_gcp_4gpu_memory_v2.yaml
```

The probe completed all 16 configured A-comparator trajectories with no failed
jobs and no OOM. It used one prompt, all 8 coordinates, and both signs. The
summary gate failed only because one prompt is too small for the encoder
active-fraction gate:

```text
num_trajectories: 16
num_failed: 0
tool_call_count: 112
mean_generated_tokens: 213.0
median_turn_linearization_cosine: 0.999984
median_turn_linearization_r2: 0.999964
encoder_active_feature_fraction: 0.0101
```

The valid restart configuration for A100-40GB is therefore one node with two
4-GPU workers:

```bash
GPU_GROUPS='0,1,2,3;4,5,6,7' RUN_LABEL=gcp2x4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash -lc 'bash scripts/run_phase2_gcp.sh'
```

## Capacity Notes

The preferred throughput path is one 8xH100 node. Attempts to start existing
8xH100 instances in `us-central1-a` and `europe-west4-b` failed with
`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` stockout. Attempts to create fresh
8xH100 nodes in alternate regions either hit zero regional H100
`GPUS_PER_GPU_FAMILY` quota or unsupported/stockout configurations.

After preserving the partial A100 output, the repaired A100 fallback was
restarted as `gcp2x4f` on `controllability-phase2-a2` with two 4-GPU workers.
At 2026-05-29T06:34:48Z it had produced 8 trajectories per shard, 0 failed jobs,
and no OOM/device-mismatch errors. Throughput is much slower than H100 but the
implementation repair is holding.

To improve throughput while keeping the user-requested one-node limit, a single
`a2-megagpu-16g` 16xA100 node was created in `us-central1-c` after GCP reported
capacity there. The smaller 8xA100 node was stopped first. The new run label is
`mega4x4`, using four 4-GPU workers:

```bash
GPU_GROUPS='0,1,2,3;4,5,6,7;8,9,10,11;12,13,14,15' \
RUN_LABEL=mega4x4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/run_phase2_gcp.sh
```

At 2026-05-29T07:07:32Z the 16xA100 run had produced 2 trajectories per shard
across 4 shards, 0 failed jobs, and no OOM/device-mismatch errors.

Because the 4-GPU layout was still too slow, a bounded 2-GPU worker probe was
run on the same 16xA100 node after the OOM fixes:

```text
run_id: phase2_probe_swe_grep_average_a_2gpu_after_oom_fixes
num_trajectories: 16
num_failed: 0
tool_call_count: 112
mean_generated_tokens: 213
median_turn_linearization_cosine: 0.9999966621398926
median_turn_linearization_r2: 0.9999886155128479
```

The probe failed only the one-prompt encoder active-fraction gate, which is
expected for such a small smoke and not an OOM/memory failure. The full Phase 2
batch was therefore relaunched as `mega8x2`, using eight 2-GPU workers on the
single 16xA100 node:

```bash
GPU_GROUPS='0,1;2,3;4,5;6,7;8,9;10,11;12,13;14,15' \
RUN_LABEL=mega8x2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/run_phase2_gcp.sh
```

At 2026-05-29T07:37:42Z, `mega8x2` had produced 17 trajectories across 8
shards, 0 failed jobs, and no OOM/device-mismatch errors.

The full `mega8x2` run was not safe despite the one-prompt 2-GPU probe. After
the rollout distribution reached longer prompts, every shard began producing OOM
failed jobs. The run was stopped at:

```text
run_label: mega8x2
trajectories_per_shard: 7-8
failed_jobs_per_shard: 2
dominant_error: ModelError -> OutOfMemoryError, 1.22 GiB allocation with ~0.9 GiB free
```

A stricter 3-GPU probe used two prompts, all 8 A coordinates, both signs, and
one seed. This catches the longer-context failure missed by the 2-GPU probe:

```text
run_id: phase2_probe_swe_grep_average_a_3gpu_after_oom_fixes
num_trajectories: 32
num_failed: 0
tool_call_count: 227
mean_generated_tokens: 222.1875
median_turn_linearization_cosine: 0.9999966621398926
median_turn_linearization_r2: 0.9999886155128479
```

It failed only the two-prompt encoder active-fraction gate. The full batch was
restarted as `mega5x3`, using five 3-GPU workers on the single 16xA100 node:

```bash
GPU_GROUPS='0,1,2;3,4,5;6,7,8;9,10,11;12,13,14' \
RUN_LABEL=mega5x3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/run_phase2_gcp.sh
```

At 2026-05-29T08:56:03Z, `mega5x3` had produced 144 trajectories across 5
shards, 0 failed jobs, and no OOM/device-mismatch errors. This passed the
longer-context point where `mega8x2` failed. Post-diagnostic throughput was
roughly 44-50 seconds per trajectory per worker, with reward means around
0.64-0.76 across shards.

Phase 2 shard resume support was added after this launch. For configs with
`resume_existing: true`, rerunning the same shard skips jobs already present in
`trajectories.jsonl` and archives old `failed_jobs.jsonl` before retrying failed
cells. This does not affect already-running worker processes, but it protects
future restarts of the long Phase 2 comparator batch from duplicating completed
cells.

At 2026-05-29T09:00:31Z, the active `mega5x3` A-comparator run had produced
158 trajectories across 5 shards, 0 failed jobs, and no OOM/device-mismatch
errors.

At 2026-05-29T09:32:32Z, `mega5x3` had produced 343 A-comparator trajectories
across 5 shards, 0 failed jobs, and no OOM/device-mismatch errors. Mean reward
across the observed trajectories was 0.951, and post-diagnostic throughput was
about 52-54 seconds per trajectory per worker.

At 2026-05-29T10:05:32Z, `mega5x3` had produced 468 A-comparator trajectories
across 5 shards, 0 failed jobs, and no OOM/device-mismatch errors. Observed
throughput from the shard trajectory timestamps was about 269 trajectories/hour
across the five workers. At that rate the full A-comparator target of 9600
trajectories has roughly 34 hours remaining.

At 2026-05-29T10:11:04Z, the GCP VM branch pointer was aligned to pushed commit
`479de9db5` while the active A-comparator workers continued running. This fixed
the hot-patch reproducibility issue where the VM had been running staged
patched files on an older `HEAD`. The running A shards were already started from
the hot-patched worktree; future C/PCA/math500 and Stage 5/6 processes will
record the pushed commit hash. The VM still had an unstaged `uv.lock` rewrite
from `uv run` and two untracked probe configs; those are not used by the active
Phase 2 comparator configs.

At 2026-05-29T10:18:23Z, `mega5x3` had produced 521 A-comparator trajectories
with 0 failed jobs and no OOM/device-mismatch errors. The VM `uv.lock` drift
from `uv run` was backed up under `/tmp` and restored to the pushed branch
version so future `uv run` invocations for C/PCA/math500 and Stage 5/6 use the
checked-in lockfile. The active A workers continued running.
