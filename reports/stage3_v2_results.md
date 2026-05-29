# Stage 3 v2 Results

Date: 2026-05-07

Stage 3 v2 completed for both Tier A environments. This is a descriptive geometric-profile stage downstream of the passed Stage 1 v2 and Stage 2 v2 gates.

## Gate Result

Both Stage 3 v2 runs passed.

| env_id | run_id | primary K | primary mean rho | primary mean gamma | primary min eta_star |
| --- | --- | ---: | ---: | ---: | ---: |
| `prime/swe-grep` | `stage3_v2_Qwen_Qwen3_8B_prime_swe_grep` | 6 | 0.848 | 114.265 | 0.101 |
| `primeintellect/math500` | `stage3_v2_Qwen_Qwen3_8B_primeintellect_math500` | 7 | 0.798 | 9.618 | 0.057 |

## Evidence

- `runs/stage3_v2_Qwen_Qwen3_8B_prime_swe_grep/summary.json`
- `runs/stage3_v2_Qwen_Qwen3_8B_prime_swe_grep/per_coordinate_geometry.parquet`
- `runs/stage3_v2_Qwen_Qwen3_8B_prime_swe_grep/aggregated_per_coordinate.json`
- `runs/stage3_v2_Qwen_Qwen3_8B_prime_swe_grep/linearization_sweep.parquet`
- `runs/stage3_v2_Qwen_Qwen3_8B_primeintellect_math500/summary.json`
- `runs/stage3_v2_Qwen_Qwen3_8B_primeintellect_math500/per_coordinate_geometry.parquet`
- `runs/stage3_v2_Qwen_Qwen3_8B_primeintellect_math500/aggregated_per_coordinate.json`
- `runs/stage3_v2_Qwen_Qwen3_8B_primeintellect_math500/linearization_sweep.parquet`

## Interpretation

The B-primary graph-free coordinates remain writable under the Stage 3 gain profile. The profile also carries A, C, and the original graph baseline so downstream figures can compare per-coordinate gain across formulations.

Limitations are explicit in each summary: Stage 3 v2 uses saved Stage 1 spectra and aggregate train/heldout energy profiles, not steered rollouts. The `rho` field is a normalized gain/reachability proxy, and the linearization sweep is an analytic trust-region surrogate. Finite-difference steering validation remains downstream.
