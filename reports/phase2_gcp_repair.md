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
