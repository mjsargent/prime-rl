# Phase 1.5 Float32 Verifier-Loop Smoke Report

Date: 2026-05-14 UTC

## Run

- Slurm job: `6689`
- Slurm allocation: one node, one task, one GPU
- Node: `primeintellect-freyr-pool2-gpu-001`
- Script: `runs/slurm/phase1_5_smoke_float32.sbatch`
- Swe-grep config: `configs/controllability/experiments/phase1_5_swe_grep_verifier_smoke_parametric_b_float32.yaml`
- Swe-grep output: `runs/phase1_5_swe_grep_verifier_smoke_parametric_b_float32/summary.json`

The Slurm script was constrained to `--nodes=1`, `--ntasks=1`, and `--gres=gpu:1`.
The job stopped after swe-grep because the Phase 1.5 gate failed; math500 was not
run in this job.

## Precision Blocker Documentation

The bfloat16 steering backend is invalid for this pipeline. The steering contract
run `steering_contract_validation_math500_parametric_b` passed zero-edit decode
fidelity and closed-loop budget semantics, but failed finite-difference linearity:

- median small-edit cosine: `0.3053908050060272`
- p10 small-edit cosine: `0.1573934257030487`
- median small-edit R2: `-2.2026100158691406`
- eta sweep selected multiplier: `0.0`

The float32 steering contract run `steering_contract_validation_math500_parametric_b_float32`
passed the same contract:

- median small-edit cosine: `0.9997302889823914`
- p10 small-edit cosine: `0.9995665550231934`
- median small-edit R2: `0.9993909597396851`
- eta sweep selected multiplier: `1.0`

Interpretation: bfloat16 quantization destroys the residual finite-difference
signal used by steering diagnostics. The earlier Phase 1.5 bfloat16 failure is
superseded for linearization purposes and should not be interpreted as evidence
against the construction.

## Swe-Grep Float32 Result

Gate: `fail`

The float32 rerun fixed the linearization failure but did not pass the full
verifier-loop smoke.

Key metrics:

- expected jobs: `2160`
- trajectories produced: `2160`
- failed jobs: `0`
- tool calls: `7563`
- reward mean: `0.0`
- nonzero reward count: `0`
- mean generated tokens: `111.77361111111111`
- encoder active feature fraction: `0.0028846153846153848`
- turn diagnostics: `8208`
- median turn linearization cosine: `0.999897837638855`
- p10 turn linearization cosine: `0.999833345413208`
- median turn linearization R2: `0.9997844696044922`
- p10 turn linearization R2: `0.9996509552001953`

Passed checks:

- all configured controller/sign/coordinate cells present
- completeness negative-control test exited nonzero after removing one cell
- trajectories were produced for every job
- tool calls were produced
- turn-level linearization passed decisively

Failed checks:

- no trajectory received nonzero verifier reward
- generated trajectories remained below the agent-length floor
- encoder active feature fraction was far below the 0.5 floor

## Interpretation

This run separates the precision problem from the remaining verifier-loop problem.
Float32 steering is numerically valid: the turn-level linearization is excellent
across 8208 measured steering points. The remaining failure is not a residual
linearization failure.

The remaining blockers are behavioral/integration-side:

1. The tool-use loop executes tool calls but does not produce rewarded swe-grep
   trajectories under steering.
2. The generated trajectories are still short relative to the full-agent floor.
3. The sentence-encoder active fraction remains near zero within prompt groups,
   so the encoder still sees almost no coordinate-conditioned behavioral variation.

Phase 2 remains blocked. The next fix should target the verifier-loop behavior and
encoder input, not the steering finite-difference math.
