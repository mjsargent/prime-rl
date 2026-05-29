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

At 2026-05-29T10:21:45Z, a CPU-only downstream config wiring check confirmed
that the Stage 5/6 config files exist and point to the expected pending merged
outputs. The check also reproduced the `uv.lock` rewrite from plain `uv run`;
the VM lockfile was restored again. `scripts/run_phase2_gcp.sh` and the monitor
runbook were patched to use `uv run --locked` for future restarts and monitoring
commands. The currently active A process was already launched from the previous
script contents and was not interrupted.

At 2026-05-29T10:33:25Z, `mega5x3` had produced 590 A-comparator trajectories
with 0 failed jobs and no OOM/device-mismatch errors. The VM remained at
`f9c39f276` because the active bash runner has already opened
`scripts/run_phase2_gcp.sh`; replacing that script under a running bash process
is not worth the handoff risk. If the active run fails and must be restarted,
restart from the newer pushed branch tip (`4728b8444` or later), which uses
`uv run --locked`.

At 2026-05-29T11:48:32Z, the active A-comparator run crossed the first archive
checkpoint and continued running afterward:

```text
run_id prefix: phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32_mega5x3
trajectories: 1008 / 9600
failed jobs: 0
OOM/traceback/tensor-device/ModelError signatures: 0
snapshot: runs/gcp_snapshots/phase2_mega5x3_a_snapshot_20260529T1148Z.tgz
```

At 2026-05-29T11:53:06Z, the same run was still live at 1024 trajectories, with
0 failed jobs and no OOM/device-mismatch errors.

A downstream handoff check found that the VM was missing the Stage 2
parametric-B random baseline artifacts that Stage 5 loads from
`stage2_parametric_b`. The artifacts existed locally and were copied to the VM:

```text
runs/stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b/random_baseline.json
runs/stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b/random_baseline.json
```

The post-copy handoff check passed: the preexisting swe-grep B merged run was
available and passing (`10800` trajectories, `0` failed jobs, encoder active
fraction `0.8586`), both Stage 2 random baselines were present, and Stage 3
aggregated geometry contained the expected formulations for swe-grep and
math500 (`parametric_b`, `average_controllability_a`, `koopman_dmd_c`, and
`original_graph`).

A follow-up remote input-path check passed for the remaining queued Phase 2
configs. The VM has the Stage 3 aggregated geometry and locked baseline YAMLs
needed for swe-grep A, swe-grep C, swe-grep PCA-proxy, and math500 B. At that
point only the active A shards existed (`5` shard directories); C, PCA-proxy,
and math500 B correctly had `0` shard directories because they had not started
yet.

The active VM checkout still cannot run `uv run --locked` because the old
checkout reports that `uv.lock` needs updating against the current resolver
metadata. This does not affect the already-running process, which was launched
with plain `uv run`. For any restart, use the newer pushed branch tip
(`6a1cb16c1` or later), where `scripts/run_phase2_gcp.sh` uses
`uv run --locked`.

At 2026-05-29T12:57:41Z, the active A-comparator run had crossed the second
archive checkpoint and remained healthy:

```text
run_id prefix: phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32_mega5x3
trajectories: 1272 / 9600
failed jobs: 0
OOM/traceback/tensor-device/ModelError/no-space signatures: 0
last shard write age: 26 seconds
oldest shard write age: 125 seconds
snapshot: runs/gcp_snapshots/phase2_mega5x3_a_snapshot_20260529T1257Z.tgz
```

At 2026-05-29T13:12:35Z, `mega5x3` was still live at 1368 A-comparator
trajectories with 0 failed jobs, no OOM/device-mismatch/ModelError/no-space
signatures, and recent shard writes. The observed elapsed-time rate was about
0.0776 trajectories/second across the five 3-GPU workers, leaving roughly
29.5 hours for A at the current average. No merged A artifact existed yet, and
C, PCA-proxy, and math500 B had not started. The local branch was clean at
`ad88b5388`; the active VM checkout remained at `f9c39f276` for the already
running process.

At 2026-05-29T13:34:21Z, `mega5x3` had produced 1437 A-comparator
trajectories with 0 failed jobs and no OOM/traceback/tensor-device/ModelError
signatures. Active worker counts were still A=10, C=0, PCA=0, and MATH=0. The
latest shard write was 24 seconds old, the oldest shard write was 82 seconds
old, and shard counts remained balanced at 289, 288, 284, 288, and 288. The
observed elapsed-time rate was about 0.0759 trajectories/second, leaving 8163
A trajectories. No merged A artifact existed yet, and C, PCA-proxy, and
math500 B had not started. The local branch was clean at `99a72cfb3`; the
active VM checkout remained at `f9c39f276` for the already-running process.

At 2026-05-29T13:50:02Z, `mega5x3` crossed a 1500-trajectory archive
checkpoint and remained healthy:

```text
run_id prefix: phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32_mega5x3
trajectories: 1504 / 9600
failed jobs: 0
OOM/traceback/tensor-device/ModelError/no-space signatures: 0
last shard write age: 6 seconds
oldest shard write age: 56 seconds
shard counts: 303, 302, 296, 302, 301
snapshot: runs/gcp_snapshots/phase2_mega5x3_a_snapshot_20260529T1350Z.tgz
```

The archive contains the five active A shard directories and
`runs/gcp_logs/phase2_mega5x3_STATUS.md`. No merged A artifact existed yet, and
C, PCA-proxy, and math500 B had not started. The local branch was clean at
`283d971a9`; the active VM checkout remained at `f9c39f276`.

At 2026-05-29T14:10:22Z, `mega5x3` crossed 1600 A-comparator trajectories and
remained healthy:

```text
run_id prefix: phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32_mega5x3
trajectories: 1602 / 9600
failed jobs: 0
OOM/traceback/tensor-device/ModelError/no-space signatures: 0
workers: A=10, C=0, PCA=0, MATH=0
last shard write age: 0 seconds
oldest shard write age: 37 seconds
shard counts: 323, 322, 315, 321, 321
disk: /dev/root 969G total, 58G used, 912G free
```

No merged A artifact existed yet, and C, PCA-proxy, and math500 B had not
started. The local branch was clean at `11efaa102`; the active VM checkout
remained at `f9c39f276` for the already-running process.

At 2026-05-29T14:34:41Z, `mega5x3` crossed 1700 A-comparator trajectories and
remained healthy:

```text
run_id prefix: phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32_mega5x3
trajectories: 1704 / 9600
failed jobs: 0
OOM/traceback/tensor-device/ModelError/no-space signatures: 0
workers: A=10, C=0, PCA=0, MATH=0
last shard write age: 50 seconds
oldest shard write age: 93 seconds
shard counts: 342, 342, 338, 341, 341
disk: /dev/root 969G total, 58G used, 912G free
```

No merged A artifact existed yet, and C, PCA-proxy, and math500 B had not
started. The local branch was clean at `b6f54c874`; the active VM checkout
remained at `f9c39f276` for the already-running process.

At 2026-05-29T15:03:00Z, `mega5x3` crossed 1800 A-comparator trajectories and
remained healthy:

```text
run_id prefix: phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32_mega5x3
trajectories: 1803 / 9600
failed jobs: 0
OOM/traceback/tensor-device/ModelError/no-space signatures: 0
workers: A=10, C=0, PCA=0, MATH=0
last shard write age: 0 seconds
oldest shard write age: 73 seconds
shard counts: 362, 362, 356, 362, 361
disk: /dev/root 969G total, 58G used, 912G free
```

No merged A artifact existed yet, and C, PCA-proxy, and math500 B had not
started. The local branch was clean at `a8c84ccbe`; the active VM checkout
remained at `f9c39f276` for the already-running process.

At 2026-05-29T15:13:30Z, a narrowed read-only VM handoff preflight passed for
the files the queued code actually opens before or during downstream stages:

```text
OK runs/phase2_swe_grep_parametric_b_behavioral_v3_qwen_tooluse_float32_merged/summary.json
OK runs/stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b/random_baseline.json
OK runs/stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b/random_baseline.json
OK runs/stage3_v2_Qwen_Qwen3_8B_prime_swe_grep/aggregated_per_coordinate.json
OK runs/stage3_v2_Qwen_Qwen3_8B_primeintellect_math500/aggregated_per_coordinate.json
```

A broader exploratory check also found missing A/C/PCA locked-baseline
`random_baseline.json` paths and a missing math500 steering-contract summary,
but those paths are not opened by the queued Phase 2 generator or Stage 5
handoff path. Phase 2 generator inputs are the Stage 3 aggregated geometry and
the existing parametric-B reference, while Stage 5 reads only
`stage2_parametric_b/random_baseline.json` for each environment.

At 2026-05-29T15:24:23Z, `mega5x3` crossed 1900 A-comparator trajectories and
remained healthy:

```text
run_id prefix: phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32_mega5x3
trajectories: 1919 / 9600
failed jobs: 0
OOM/traceback/tensor-device/ModelError/no-space signatures: 0
workers: A=10, C=0, PCA=0, MATH=0
last shard write age: 3 seconds
oldest shard write age: 36 seconds
shard counts: 386, 386, 376, 385, 386
disk: /dev/root 969G total, 58G used, 912G free
```

No merged A artifact existed yet, and C, PCA-proxy, and math500 B had not
started. The local branch was clean at `698d2a368`; the active VM checkout
remained at `f9c39f276` for the already-running process.

At 2026-05-29T15:45:42Z, `mega5x3` crossed a 2000-trajectory archive
checkpoint and remained healthy:

```text
run_id prefix: phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32_mega5x3
trajectories: 2040 / 9600
failed jobs: 0
OOM/traceback/tensor-device/ModelError/no-space signatures: 0
workers: A=10, C=0, PCA=0, MATH=0
last shard write age: 2 seconds
oldest shard write age: 92 seconds
shard counts: 409, 409, 405, 408, 409
disk: /dev/root 969G total, 58G used, 912G free
snapshot: runs/gcp_snapshots/phase2_mega5x3_a_snapshot_20260529T1545Z.tgz
```

The archive contains the five active A shard `trajectories.jsonl` and
`raw_rollouts.jsonl` files plus `runs/gcp_logs/phase2_mega5x3_STATUS.md`. No
merged A artifact existed yet, and C, PCA-proxy, and math500 B had not started.
The local branch was clean at `4aa9fe2d4`; the active VM checkout remained at
`f9c39f276` for the already-running process.

At 2026-05-29T16:19:24Z, `mega5x3` remained healthy:

```text
run_id prefix: phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32_mega5x3
trajectories: 2246 / 9600
failed jobs: 0
OOM/traceback/tensor-device/ModelError/no-space signatures: 0
workers: A=10, C=0, PCA=0, MATH=0
last shard write age: 8 seconds
oldest shard write age: 22 seconds
shard counts: 453, 451, 440, 450, 452
disk: /dev/root 969G total, 58G used, 912G free
```

The local `matthew@vmax.ai` gcloud credential required interactive
reauthentication at this checkpoint, but the VM was still reachable
non-interactively by using `research-ops@vmax-rl.iam.gserviceaccount.com` for
the IAP ProxyCommand and the existing `matthew_vmax_ai_com` SSH key for the VM
user. No merged A artifact existed yet, and C, PCA-proxy, and math500 B had not
started. The local branch was clean at `e7abf42ff`; the active VM checkout
remained at `f9c39f276` for the already-running process.

At 2026-05-29T17:11:02Z, `mega5x3` crossed 2500 A-comparator trajectories and
remained healthy:

```text
run_id prefix: phase2_swe_grep_average_controllability_a_behavioral_v3_qwen_tooluse_float32_mega5x3
trajectories: 2540 / 9600
failed jobs: 0
OOM/traceback/tensor-device/ModelError/no-space signatures: 0
workers: A=10, C=0, PCA=0, MATH=0
last shard write age: 6 seconds
oldest shard write age: 62 seconds
shard counts: 512, 510, 497, 510, 511
disk: /dev/root 969G total, 58G used, 912G free
```

No merged A artifact existed yet, and C, PCA-proxy, and math500 B had not
started. The active VM checkout remained at `f9c39f276` for the already-running
process.
