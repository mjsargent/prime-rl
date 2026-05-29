# Stage 4 v2 Pilot Results

Stage 4 v2 ran as a reduced HF patched-decoding pilot for the `parametric_b` formulation on the two Tier A environments. This is not yet the full verifier/tool-interactive Stage 4 sweep.

## Passing runs

| env | run_id | jobs | trajectories | failed | coherence | mean generated tokens | trust violation mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| swe-grep | `stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b_pilot_rerun3` | 74 | 74 | 0 | 1.000 | 24.0 | 0.333 |
| math500 | `stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b_pilot_rerun3` | 74 | 74 | 0 | 1.000 | 24.0 | 0.333 |

The 0.333 trust-region violation mean is expected for this diagnostic pilot: each coordinate swept `0.5x`, `1.0x`, and `1.5x eta_star`, so the `1.5x` rows exceed the trust region by construction.

Corrected realized-vs-predicted displacement correlations:

| env | diagnostics | correlation |
| --- | ---: | ---: |
| swe-grep | 72 | 0.985 |
| math500 | 72 | 0.981 |

## Diagnostic artifacts

Each passing run writes:

- `trust_region_violation_rate_by_edit_norm.json`
- `realized_vs_predicted_displacement.parquet`
- `trajectory_length_distribution_by_controller.json`
- `coherence_rate_by_edit_norm.json`
- `encoded.parquet`
- `failed_jobs.jsonl`

Requested figures were generated with `manifest.json`, `data.parquet`, `plot.py`, `figure.svg`, and `figure.pdf`:

- `paper/figures/stage4_trust_region_violation_rate/`
- `paper/figures/stage4_realized_vs_predicted_displacement/`
- `paper/figures/stage4_trajectory_length_by_controller/`
- `paper/figures/stage4_coherence_rate_by_edit_norm/`

## Implementation fixes surfaced by Stage 4

- The first pilot runs were corrected from `pass` to `fail` in `runs/audit_table.md` and their `summary.json` files because the gate incorrectly accepted zero-trajectory runs when all cells were accounted for as failures.
- `controllability/experiments/stage4_trajectory_generation.py` now handles dict-like tokenizer outputs from Qwen chat templates.
- The Stage 4 gate now requires every job cell to produce a trajectory for this pilot.
- `controllability/models/frozen_model.py` now moves VJP residual deltas to the actual patch-layer device, which fixes `device_map=auto` multi-GPU placement failures.
- The displacement diagnostic now logs the first-order scalar prediction `delta dot J^T grad`, not merely the edit norm.

## Limitations

This pilot uses HF patched decoding from environment prompts. It does not execute full prime-rl tool-use rollouts or verifier-scored environment interaction, so rewards and success are unset for generated steered trajectories. Coherence is a text-level completion sanity signal. The pilot covers two prompts, one seed, two coordinates, three controllers, two signs, and three edit norms per environment.
