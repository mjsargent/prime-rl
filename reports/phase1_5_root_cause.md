# Phase 1.5 Root-Cause Diagnostics

Date: 2026-05-13

## Status

Do not run Stage 3 v2 or Phase 2 yet. The Phase 1.5 failure is not explained by a single overly loose multi-turn trust-region budget.

Three diagnostics were run through Slurm:

- `runs/diagnostic_steering_application/summary.json`
- `runs/diagnostic_encoder_audit/summary.json`
- `runs/diagnostic_singleturn_baseline_math500_parametric_b/summary.json`

## Diagnostic 1: Steering Application

Gate: `fail`

Findings:

- Closed-loop verifier steering recomputes and applies a residual edit inside every generation chunk.
- It is not once per trajectory.
- For `one_shot` and `open_loop`, steering is applied per assistant turn.
- For `closed_loop`, steering is applied per internal generation chunk within a turn.
- The default diagnostic records only `chunk_index=0`, so Phase 1.5 under-reported within-turn repeated edits.
- Patch position is consistently last-token (`-1`) relative to the current prompt, matching the single-completion implementation syntactically.
- There is no cumulative trust-region cap across chunks or turns. Trust-region violation is checked only as `edit_norm > eta_star` per edit.

Interpretation:

The closed-loop controller currently violates the single-application assumption behind the original trust-region budget. This is a real implementation-level confound for verifier-loop steering.

## Diagnostic 2: Encoder Audit

Gate: `fail`

Key metrics:

- Feature dimension: `1040`
- Global active fraction: `0.9913`
- Within-prompt active fraction: `0.0048`
- Prompt groups: `20`
- First-10 pairwise distance mean: `0.000106`
- First-10 pairwise distance max: `0.000528`
- First-10 trajectory text character count: exactly `4807` for all 10 sampled rows
- First-10 mean tool calls: `1.0`

Interpretation:

The sentence encoder is not globally constant. Across prompts, feature dimensions vary. The failure is specifically within-prompt: for the same task prompt, different coordinates/signs/controllers produce near-identical trajectory text and near-identical sentence embeddings.

This means the Phase 1.5 encoder active-fraction failure is mostly a behavioral/steering failure, not a frozen-encoder integration failure.

## Diagnostic 3: Single-Turn Math500 Baseline

Gate: `fail`

Setup:

- `50` single-turn steered math500 trajectories.
- `5` prompts.
- `5` coordinates.
- `2` signs.
- `one_shot` only.
- Same HF client, same sentence encoder, same locked Stage 3 budget.

Key metrics:

- Trajectories: `50 / 50`
- Failed jobs: `0`
- Mean reward: `0.6`
- Nonzero-reward trajectories: `30`
- Mean generated tokens: `411.76`
- Encoder active fraction: `0.9846`
- Median linearization cosine: `0.1295`
- Median linearization R2: `-474.75`
- P10 linearization cosine: `-0.2952`
- P10 linearization R2: `-2082.75`

Interpretation:

The new pipeline is not simply failing because swe-grep is multi-turn. On math500, quality, length, and encoder variation recover, but the measured linearization still fails badly.

This isolates the main linearization problem to the steering/diagnostic contract itself or to the locked edit budget under actual finite-difference measurement. The original Stage 3 linearization values were analytic surrogates, not this verifier-loop finite-difference measurement, so the original budget cannot be treated as empirically validated for the current HF client.

## Root Cause Assessment

The dominant blocker is: actual finite-difference linearization at the locked edit budget is invalid in the current HF steering pipeline, even in single-turn settings.

Secondary blockers:

- Closed-loop verifier steering compounds edits inside generation chunks and has no cumulative cap.
- The swe-grep encoder failure reflects near-identical prompt-conditioned steered behavior, not a globally constant sentence encoder.
- swe-grep reward remains zero despite tool calls, likely because the steered behaviors are not retrieving useful files and are too invariant/low-content within each prompt.

## Required Fix Before Any Rerun

Do not start Stage 3 v2 as a pure recalibration pass yet. First add and pass an end-to-end steering contract:

1. Zero-edit fidelity: greedy base decode and greedy patched decode with `delta=0` must match byte-for-byte under the same prompt/tool template.
2. Small-edit linearization: measured finite-difference displacement and predicted displacement must have cosine `> 0.9` on a 10-trajectory sample at a small probe norm.
3. Locked-budget validation: after small-edit passes, sweep edit norms up to the locked `eta_star` and choose a budget only where finite-difference cosine remains acceptable.
4. Closed-loop semantics: either apply steering once per assistant turn, or track and cap cumulative edit exposure across chunks/turns.

Only after those contract checks pass should Stage 3 v2 recalibrate verifier-loop trust regions.

## Stop Decision

Phase 2 remains blocked. Stage 5 and Stage 6 Phase 1 entries remain unsuperseded.
