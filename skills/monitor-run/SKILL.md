---
name: monitor-run
description: How to monitor ongoing training runs — find output directories, check logs, diagnose performance, and inspect SLURM jobs. Use when asked to check on a run, debug training issues, or investigate performance.
---

# Monitor RL Run

## Runbook

### On launch

Immediately gather context and write a summary of the run into `{output_dir}/STATUS.md`:

1. Identify the output directory and read the resolved configs to understand the experiment (model, envs, hyperparameters, deployment details).
2. Make sure that the run started successfully and that all processes are alive.
3. Read the logs and note the current training step and health.

### Recurring check-ins

After the initial overview, schedule recurring check-ins. By default, check in every **1 hour** (the researcher can override this).

At each check-in:

1. Check that all processes are alive.
2. Read the logs — look for errors, warnings, hangs, or degraded performance
3. Note the current training step, key metrics, and checkpoint progress.
4. **Append an entry to `{output_dir}/STATUS.md`**:

```markdown
## YYYY-MM-DD HH:MM UTC

**Step**: {current_step} / {max_steps}
**Health**: {Healthy | Degraded | Down}

**Progress**: reward/mean, seq_len, truncation rate, eval scores (if available), notable env-specific metrics.
**Stability**: entropy, mismatch KL, grad norm — flag any spikes or concerning trends.
**Performance**: trainer and orchestrator step times, who is waiting on whom, env server lag, inference pressure.

**Notes**: (anything unusual — errors, restarts, hangs, etc. Omit if nothing notable.)
```

Always append — never overwrite previous entries.

### Controllability Phase 2 on GCP

For `scripts/run_phase2_gcp.sh`, monitor the master log and shard output under `runs/gcp_logs/`.

```bash
tail -n 80 runs/gcp_logs/run_phase2_gcp4x2_master.out
tail -n 120 runs/gcp_logs/run_phase2_gcp4x2_master.err
for d in runs/phase2_*_gcp4x2_shard*; do
  [ -d "$d" ] && printf "%s " "$(basename "$d")" && wc -l "$d/trajectories.jsonl" "$d/failed_jobs.jsonl" 2>/dev/null | tail -1
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

For Qwen3-8B float32 steering on A100-40GB, two GPUs per worker can load
and start rollouts, but long verifier-loop contexts can OOM during VJP-based
closed-loop steering. Use four GPUs per worker for Phase 2 verifier-scale
comparators on A100-40GB:

```bash
GPU_GROUPS='0,1,2,3;4,5,6,7' RUN_LABEL=gcp2x4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash -lc 'bash scripts/run_phase2_gcp.sh'
```

When launching from SSH/IAP with `nohup`, use a login shell or rely on the
script's built-in `~/.local/bin` PATH prefix. A non-login `nohup bash ...`
can fail with `uv: command not found` even though interactive shells find `uv`.

One GPU per worker is not enough for this path: the model loads but rollout
generation OOMs near 39.5 GiB used. Two GPUs per worker can also OOM once
multi-turn prompts grow. H100-80GB can run one worker per GPU.

On a single 16xA100 `a2-megagpu-16g`, prefer five 3-GPU workers for the
float32 Qwen3-8B Phase 2 verifier path:

```bash
GPU_GROUPS='0,1,2;3,4,5;6,7,8;9,10,11;12,13,14' RUN_LABEL=mega5x3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/run_phase2_gcp.sh
```

Eight 2-GPU workers are not safe for full swe-grep Phase 2 even if a tiny
one-prompt probe passes: longer prompts produced `ModelError ->
OutOfMemoryError` failed jobs around the second prompt. A two-prompt 3-GPU probe
completed 32/32 trajectories with 0 failed jobs. In the full `mega5x3`
A-comparator run, a 468-trajectory stability sample had 0 failed jobs and
throughput around 269 trajectories/hour across all five workers, so a 9600-cell
comparator takes roughly 36 hours on this fallback node.

If an existing or fresh H100 node is unavailable, document the exact GCP error
(`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` stockout or zero regional
`GPUS_PER_GPU_FAMILY` quota) and continue on the repaired one-node A100 fallback
rather than leaving the phase idle.

If a Phase 2 comparator produces low reward or short outputs, treat those as
reported comparator outcomes unless `failed_jobs.jsonl` is non-empty or the
summary's `phase2_required_checks` fail. Phase 2 no longer uses the Phase 1.5
quality/length smoke gates to stop the batch.

Phase 2 configs should set `linearization_diagnostic_max_jobs: 16` to keep a
bounded sample of per-turn linearization probes. Full per-trajectory probes are
too slow on A100-40GB because every probe does extra patched forwards through
long multi-turn contexts.

Use `uv run --locked ...` for Phase 2 monitoring and relaunch commands unless
you intentionally need to update the lockfile. Plain `uv run` can rewrite
`uv.lock` on the GCP VM because of project-level exclude-newer settings, leaving
the long-running worktree dirty even when no source files changed.

Prefer explicit one-shot status snapshots over ad hoc background monitor loops
for the GCP Phase 2 fallback. A failed loop can leave duplicate sleeping shell
processes that add noise without improving recovery. If you do generate markdown
shard lines in shell, avoid `printf "- ..."` because bash can parse the leading
dash as an option; use `printf "%s\n" "- shard: ..."` instead.

The repo provides a one-shot helper for this:

```bash
RUN_LABEL=mega5x3 bash scripts/phase2_status_snapshot.sh
```

It appends a markdown snapshot under `runs/gcp_logs/` and prints the tail. The
snapshot includes active A/C/PCA/MATH process counts, stage subtotals, shard
write ages, A-comparator ETA, merged-output presence, error signatures, and disk
usage. From a local workstation, prefer a simple remote helper invocation over a
long nested quoted diagnostic:

```bash
gcloud compute ssh controllability-phase2-a2-16-uscentral1c \
  --zone=us-central1-c \
  --tunnel-through-iap \
  --command='cd ~/prime-rl-phase2 && RUN_LABEL=mega5x3 bash scripts/phase2_status_snapshot.sh'
```

For Phase 2 handoff preflights, check files the queued code actually opens:
the existing swe-grep parametric-B merged summary, the swe-grep and math500
`stage2_parametric_b/random_baseline.json` files consumed by Stage 5, and the
Stage 3 aggregated geometry files consumed by the queued Phase 2 configs.
Do not treat the A/C/PCA locked-baseline `random_baseline.json` paths as
handoff blockers; the Phase 2 generator does not open them, and Stage 5 reads
only `stage2_parametric_b`.

For unusually complex one-off SSH diagnostics, pass the remote shell through
stdin with `--command='bash -s' <<'REMOTE'` to avoid local quoting failures.

If the active user account needs interactive `gcloud` reauthentication, but the
`research-ops@vmax-rl.iam.gserviceaccount.com` account can still access the
Compute/IAP APIs, use a raw SSH command through an IAP ProxyCommand so OS Login
does not force the service-account VM username:

```bash
ssh -i ~/.ssh/google_compute_engine \
  -o 'ProxyCommand=gcloud compute start-iap-tunnel controllability-phase2-a2-16-uscentral1c 22 --listen-on-stdin --zone=us-central1-c --project=vmax-rl --account=research-ops@vmax-rl.iam.gserviceaccount.com' \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR \
  matthew_vmax_ai_com@10.128.0.85 \
  'cd ~/prime-rl-phase2 && RUN_LABEL=mega5x3 bash -s' < scripts/phase2_status_snapshot.sh
```

### Restarting a run

**IMPORTANT**: Never restart a run unless you were explicitly instructed by the researcher. If you were given permission, make sure to ask the researcher for the exact command to resume a run and under what conditions a restart is necessary.

**IMPORTANT**: Never run kill or launch commands directly from your shell. Instead, send them to the tmux **Launcher** window so the researcher can see exactly what was executed.

Use `tmux send-keys` to dispatch commands to the Launcher pane:

```bash
# Get your current tmux session name
SESSION=$(tmux display-message -p '#S')

# Send a command to the Launcher window
tmux send-keys -t "$SESSION:Launcher" 'your command here' Enter
```

After a restart, verify that all processes are back up and healthy before resuming periodic check-ins. Check the process tree and tail the logs to confirm the run is making progress again.

---

## Reference

### Output directory and tmux session

The output directory and tmux session name are typically provided by the researcher in the appended system prompt (see `scripts/tmux.sh` — the Claude window is launched with this context). If not provided, **ask the researcher** which output directory to monitor and which tmux session the run is in.

The tmux session contains the **Launcher** window where the researcher runs launch commands — this is where you should send any restart commands (see [Restarting a run](#restarting-a-run)).

Once you have the output directory, the resolved configs are at `{output_dir}/configs/`.

### Configs

The launcher writes resolved configs as TOML files to `{output_dir}/configs/`. Read `rl.toml` to get the full picture of the experiment (model, envs, hyperparameters, wandb, deployment).

### Logs

Logs are usually the most informative place to monitor a run.

```
{output_dir}/logs/
├── trainer.log                  # trainer stdout (rank 0)
├── orchestrator.log             # orchestrator stdout
├── inference.log                # vLLM inference server stdout
├── trainer/
│   ├── node_*.log               # per-node logs (multi-node only)
│   └── torchrun/                # per-rank stdout/stderr (all ranks)
├── inference/
│   ├── node_*.log               # per-node logs (multi-node only)
│   └── router_0.log             # vllm-router per replica (multi-node only)
└── envs/
    ├── train/{env_name}/
    │   ├── env_server.log
    │   └── env_worker_{id}.log
    └── eval/{env_name}/
        └── ...
```

Usually it's sufficient to tail `trainer.log`, `orchestrator.log`, and `inference.log`. For debugging, it may be necessary to check the per-node logs (`node_*.log`) or per-rank trainer logs under `torchrun/`.

```bash
tail {output_dir}/logs/trainer.log                     # training progress — kl mismatch, entropy, grad norm, trainer step time
tail {output_dir}/logs/orchestrator.log                # orchestrator — reward, rollouts, env execution, inference step time
tail {output_dir}/logs/inference.log                   # inference — completed HTTP requests, engine stats, OOM errors
tail {output_dir}/logs/envs/train/*/env_server.log     # env server — aggregated stats across its workers (lag, task distribution)
tail {output_dir}/logs/envs/train/*/env_worker_*.log   # env workers — individual env logs
```

All logs use loguru with the format `HH:mm:ss  LEVEL message`. Log levels: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`. To scan for problems:

```bash
grep -E "WARNING|ERROR" {output_dir}/logs/trainer.log
grep -E "WARNING|ERROR" {output_dir}/logs/orchestrator.log
grep -E "WARNING|ERROR" {output_dir}/logs/inference.log
grep -E "WARNING|ERROR" {output_dir}/logs/envs/train/*/env_server.log
grep -E "WARNING|ERROR" {output_dir}/logs/envs/train/*/env_worker_*.log
```

### Metrics

All metrics below are logged to the console. The source column indicates which log file to check.

#### Progress

These tell you whether the model is learning and how the run is progressing.

| Metric | Source | Description |
|--------|--------|-------------|
| `reward/{all,env}/mean` | orchestrator | mean training reward |
| `seq_len/{all,env}/mean` | orchestrator | average sequence length in tokens |
| `num_turns/{all,env}/mean` | orchestrator | average turns per rollout (ignore for single-turn envs) |
| `is_truncated/{all,env}/mean` | orchestrator | fraction of truncated rollouts |
| `empty_rollouts/{all,env}` | orchestrator | fraction of empty rollouts |
| `errored_rollouts/{all,env}` | orchestrator | fraction of errored rollouts |
| `metrics/{env}/{metric}` | orchestrator | env-specific metrics (e.g. pass rate for unit tests) |
| `eval/{env}/{avg@k,pass@k}` | orchestrator | eval scores (if eval is configured) |

#### Stability

These tell you whether training is healthy or diverging.

| Metric | Source | Description |
|--------|--------|-------------|
| `mismatch_kl/mean` | trainer | KL divergence between trainer and (old) inference policy  |
| `entropy/mean` | trainer | policy entropy |
| `optim/grad_norm` | trainer | gradient norm — spikes may precede divergence |

#### Performance

These tell you how fast the run is and where the bottlenecks are. Trainer and orchestrator step independently — compare their step times to identify who is waiting on whom.

**Trainer** (in `trainer.log`):

| Metric | Description |
|--------|-------------|
| `time/step` | total trainer step time |
| `time/wait_for_batch` | time waiting for orchestrator to deliver a batch — **high = orchestrator is the bottleneck** |
| `time/forward_backward` | forward/backward pass time |
| `time/broadcast_weights` | time broadcasting weights to inference |
| `time/save_ckpt` | checkpoint save time |
| `perf/throughput` | tokens/s throughput |
| `perf/mfu` | model FLOPs utilization % |

**Orchestrator** (in `orchestrator.log`):

| Metric | Description |
|--------|-------------|
| `time/step` | total orchestrator step time |
| `time/generate_completions` | rollout generation time |
| `time/wait_for_ckpt` | time waiting for trainer checkpoint — **high = trainer is the bottleneck** |
| `time/update_weights` | weight update time |
| `scheduler/async_level` | current async level |
| `scheduler/inflight_rollouts` | number of in-flight rollouts |

**Env servers** (in `envs/train/{env_name}/env_server.log`):

| Metric | Description |
|--------|-------------|
| event loop lag (min/mean/p90/p99/max) | server and worker lag stats, logged periodically |
| active task distribution | per-worker and per-env task counts  |

**Inference** (in `inference.log`):

vLLM logs completed HTTP requests and occasionally engine stats. For live inference metrics, query the vLLM metrics endpoint directly:

```bash
# Get vLLM Prometheus metrics (num running/queued reqs, KV cache usage, etc.)
curl -s http://localhost:8000/metrics | grep -E "num_requests|gpu_cache_usage"
```

Key vLLM metrics to watch:
- `vllm:num_requests_running` — requests currently being processed
- `vllm:num_requests_waiting` — requests queued waiting for KV cache space
- `vllm:gpu_cache_usage_perc` — KV cache pressure (approaching 1.0 = requests will queue)

### Rollouts

Plain-text rollouts (`verifiers` format) are saved every step alongside the binary training batch inside the run directory. For single-run (local) runs this is typically `{output_dir}/run_default`.

```
{output_dir}/{run_dir}/rollouts/
└── step_{N}/
    ├── train_rollouts.jsonl   # all train rollouts
    ├── eval_rollouts.jsonl    # all eval rollouts (only present when eval ran)
    └── train_rollouts.bin     # binary-encoded training batch (consumed by the trainer)
```

Each line in the `.jsonl` files is a JSON-serialized `vf.RolloutOutput` dict with fields like `example_id`, `task`, `prompt`, `completion`, `reward`, `trajectory`, `metrics`, etc. To inspect rollouts for a given step:

```bash
# Count rollouts at a step
wc -l {output_dir}/{run_dir}/rollouts/step_42/train_rollouts.jsonl

# Preview first rollout (pretty-printed)
head -1 {output_dir}/{run_dir}/rollouts/step_42/train_rollouts.jsonl | python -m json.tool

# Extract rewards
jq '.reward' {output_dir}/{run_dir}/rollouts/step_42/train_rollouts.jsonl
```

### Controllability encoder checks

For controllability verifier-loop smokes, do not treat a low sentence-encoder active fraction as a construction failure until the encoder input has been audited. On long tool-use prompts, chronological trajectory text can be prompt-prefix dominated: E5-style encoders with `max_length=512` may truncate before assistant/tool behavior begins. A known failure signature is:

- same-prompt/different-coordinate sentence-embedding cosine near `1.0`;
- zero active sentence dimensions across thresholds down to `1e-12`;
- coordinate discriminator on sentence embeddings at chance;
- behavior starts after the encoder window in a tokenizer audit.

When this appears, fix the encoder input by embedding behavior-bearing text only (assistant messages, tool calls, tool outputs, final answer, compact metadata), or by chunking/pooling the trajectory so tool-use behavior cannot be truncated away.

### Controllability steering contract

Before running scaled verifier-loop steering experiments, run the steering contract on
the target model/env/formulation. The contract must pass zero-edit greedy decode
fidelity, finite-difference linearity, edit-norm sweep, and closed-loop cumulative
budget semantics.

```bash
uv run python -m controllability.reports.steering_contract_validation \
  --config configs/controllability/experiments/steering_contract_math500_parametric_b_float32.yaml
```

Do not scale Stage 4/Phase 2 from a bfloat16 steering backend unless its own contract
passes. In prior validation, bfloat16 preserved zero-edit decode fidelity but broke
finite-difference linearization, while float32 passed the same contract.

For Qwen3 tool-use verifier-loop runs, avoid greedy/truncated HF decoding. Use
Qwen3 no-thinking tool-call settings (`enable_thinking=false`, `temperature=0.7`,
`top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`) and keep enough prompt context
for the full tool-use system prompt plus tool results. A 512-token prefix cap
truncated swe-grep prompts that were over 1k tokens in successful base rollouts.

### Errors and warnings

As part of every check-in, grep all logs for `WARNING` and `ERROR` level messages. Pay special attention to env server and env worker logs — these are the most common source of issues since they run user-provided code.

Common things to look for:
- **Env workers**: exceptions in environment execution, timeouts, sandbox errors, OOM kills
- **Orchestrator**: empty/errored rollout spikes, weight broadcast failures, checkpoint errors
- **Trainer**: NCCL/CUDA errors, OOM, NaN loss or gradients
- **Inference**: NCCL/CUDA errors, OOM, request timeouts

A small number of warnings is normal (e.g. occasional env timeouts). Escalate to the researcher if you see errors that are persistent, increasing, or affect a large fraction of rollouts.

### Processes

All processes have custom process names, making them easy to identify in `ps`, `htop`, and `pstree`. Use this in case you need to debug a process that is not responding or behaving unexpectedly.

```
PRIME-RL::Launcher
├── PRIME-RL::Inference          (vLLM server, GPU 0)
├── PRIME-RL::Orchestrator       (CPU-only, data/scheduling)
│   ├── Verifiers::EnvServer     (ZMQ env server per environment)
│   │   ├── Verifiers::EnvWorker0
│   │   ├── Verifiers::EnvWorker1
│   │   └── ...
│   └── ...
├── torchrun
│   └── PRIME-RL::Trainer        (RL trainer, GPU 1+)
└── tail trainer.log
```

For multi-node runs, trainer and inference processes are distributed across separate nodes. Use `srun` or `ssh` to inspect processes on other nodes directly.
