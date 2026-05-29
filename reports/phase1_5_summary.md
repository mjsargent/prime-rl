# Phase 1.5 Slurm Smoke Summary

Date: 2026-05-13

## Execution

- Submitted Slurm smoke job `6585` on `primeintellect-freyr-pool2-gpu-004`.
- Job `6585` completed all `prime/swe-grep` verifier-loop trajectories but failed during encoding because verifier-loop metadata stores token counts under `usage.completion_tokens`, while `_write_encoded` expected `generated_tokens`.
- Patched the encoding fallback and submitted Slurm recovery job `6639`, which encoded and summarized the Slurm-produced trajectories without rerunning rollouts.
- Official evidence: `runs/phase1_5_swe_grep_verifier_smoke_parametric_b/summary.json`.

## Completeness Gate

The completeness gate is now a hard fail. The self-test removed:

- `controller=one_shot`
- `coordinate=0`
- `sign=+`

The negative completeness-only run exited with status `1`, and its summary recorded `gate=fail` with missing cell `one_shot|k=0|sign=+`.

The real swe-grep config had the full required grid:

- `3` controllers: `one_shot`, `open_loop`, `closed_loop`
- `6` coordinates
- `2` signs
- `20` prompts
- `3` seeds
- `2160` expected jobs

Configured grid completeness passed.

## swe-grep Results

Gate: `fail`.

Key metrics:

- Trajectories: `2160 / 2160`
- Failed jobs: `0`
- Tool calls: `7668`
- Mean reward: `0.0`
- Nonzero-reward trajectories: `0`
- Mean generated tokens: `110.52`
- Encoder active-feature fraction: `0.0048`
- Turn-level linearization diagnostics: `8302`
- Median turn-level linearization cosine: `0.1681`
- Median turn-level linearization R2: `-1731.75`
- P10 turn-level linearization cosine: `-0.1018`
- P10 turn-level linearization R2: `-7295.23`

Passed checks:

- Trajectories were produced.
- Tool calls were produced.
- Completeness self-test passed.
- Configured steering grid was complete.

Failed checks:

- `has_nonzero_quality`: all rewards were zero.
- `agent_length_tokens`: mean generated tokens `110.52` is below the `128` floor.
- `encoder_active_fraction_pass`: `0.0048` is far below the `0.5` threshold.
- `turn_linearization_cosine_pass`: median cosine `0.1681` is far below the `0.7` threshold.

## Stop Decision

Phase 1.5 failed on swe-grep, so Phase 2 scale-up is blocked. Stage 5 and Stage 6 Phase 1 entries were not marked as superseded, because the superseding condition requires Phase 1.5 to pass all gates.

I did not start Phase 2.

Math500 was not run after the swe-grep hard failure. The both-environment encoder active-fraction gate cannot pass once swe-grep is `0.0048 < 0.5`.

## Interpretation

The tool-use loop integration is working in the narrow sense that full verifier-loop trajectories are generated and tool calls execute. The failure is not "no tool use"; it is that the resulting steered tool-loop trajectories do not yet satisfy the quality, active-encoder, or multi-turn linearization requirements.

The multi-turn linearization result is the most important technical blocker. The median turn-level cosine dropped to `0.1681`, so the Stage 3 trust-region budget should not be reused for multi-turn verifier-loop steering. This should trigger a Stage 3 v2 trust-region/linearization study measured at each tool-loop turn before any Phase 2 trajectory scale-up.
