# Steering Contract Combined Report

Date: 2026-05-13

## Runs

- `runs/steering_contract_validation_math500_parametric_b/summary.json`
  - Slurm job: `6658`
  - Backend dtype: `bfloat16`
  - Gate: `fail`
- `runs/steering_contract_validation_math500_parametric_b_float32/summary.json`
  - Slurm job: `6659`
  - Backend dtype: `float32`
  - Gate: `pass`

Both runs used `primeintellect/math500`, `Qwen/Qwen3-8B`, formulation `parametric_b`,
patch layer `12`, readout layer `14`, and checked 12 prompt/coordinate/sign cells.

## Fix Status

1. Zero-edit greedy decode fidelity: fixed.
   - `bfloat16`: pass on 3 prompts.
   - `float32`: pass on 3 prompts.

2. Small-edit finite-difference linearity: fixed only under `float32`.
   - `bfloat16`: median cosine `0.3054`, p10 cosine `0.1574`, median R2 `-2.2026`; fail.
   - `float32`: median cosine `0.9997`, p10 cosine `0.9996`, median R2 `0.9994`; pass.

3. Eta sweep and valid operating point: fixed only under `float32`.
   - `bfloat16`: no eta multiplier passed; selected multiplier `0.0`.
   - `float32`: all tested multipliers passed through `1.0`; selected multiplier `1.0`.
   - Selected float32 eta values:
     - coordinate `0`: `0.05729645914623891`
     - coordinate `1`: `0.0688925258913129`
     - coordinate `2`: `0.07426123809318612`

4. Closed-loop cumulative edit semantics: fixed.
   - Budget mode is `l2_per_turn`.
   - For `max_completion_tokens=512` and chunk size `32`, worst-case cumulative L2 equals eta instead of compounding per chunk.
   - Gate passed in both `bfloat16` and `float32`.

## Interpretation

The steering hook and closed-loop semantics are now structurally correct: zero edits
do not perturb greedy decode, and closed-loop edits no longer repeatedly apply a full
eta budget per chunk.

The remaining failure is dtype-specific. In `bfloat16`, the finite-difference signal
is either quantized away at small norms or becomes poorly aligned once the edit is
large enough to move the residual stream. This reproduces the Phase 1.5 failure mode
and explains why the earlier verifier-loop smoke had near-random linearization.

In `float32`, the same contract passes cleanly. The first-order theory is valid at
the checked steering points, and the original Stage 3 eta values are acceptable for
the checked coordinates under float32 patched forward/decoding.

## Decision

Do not run Phase 2 scale-up with the bfloat16 steering backend.

Phase 2 may proceed only with a float32 HF steering backend, or with a separate
mixed-precision patching implementation that passes the same steering contract before
rollout generation. The current evidence supports `float32` as the valid backend for
the next verifier-loop smoke.

## Phase 1.5 Follow-Up

The follow-up verifier-loop smoke must use explicit float32 configs:

- `configs/controllability/experiments/phase1_5_swe_grep_verifier_smoke_parametric_b_float32.yaml`
- `configs/controllability/experiments/phase1_5_math500_sentence_encoder_smoke_parametric_b_float32.yaml`

The corresponding Slurm script is single-node only:

- `runs/slurm/phase1_5_smoke_float32.sbatch`
  - `--nodes=1`
  - `--ntasks=1`
  - `--gres=gpu:1`

The original bfloat16 Phase 1.5 smoke remains superseded by this precision-gated
rerun. Its failure should not be interpreted as evidence against the construction
because the steering contract established that bfloat16 invalidates the residual
finite-difference linearization.

## Phase 1.5 Float32 Outcome

The one-node Slurm smoke was run as job `6689` using the float32 swe-grep config.
It completed all `2160` swe-grep trajectories and then failed the Phase 1.5 gate,
so the script stopped before math500.

The precision fix worked:

- median turn linearization cosine: `0.999897837638855`
- median turn linearization R2: `0.9997844696044922`
- p10 turn linearization cosine: `0.999833345413208`
- turn diagnostics: `8208`

The remaining failures are not finite-difference failures:

- reward mean: `0.0`
- nonzero reward count: `0`
- mean generated tokens: `111.77361111111111`, below the `128` swe-grep floor
- encoder active feature fraction: `0.0028846153846153848`, below the `0.5` floor

Detailed report: `reports/phase1_5_float32_smoke_report.md`.
