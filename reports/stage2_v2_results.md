# Stage 2 v2 Results

Date: 2026-05-07

Stage 2 v2 ran the shared baseline-encoder sanity driver once per environment and formulation. The branch is downstream of the passed Stage 1 v2 graph-free gate only.

## Gate Result

All eight Stage 2 v2 runs passed.

| env_id | formulation | K | gate | encoder coverage | null Wasserstein | random accuracy | chance |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `prime/swe-grep` | `parametric_b` | 6 | pass | 1.000 | 0.000 | 0.1663 | 0.1667 |
| `prime/swe-grep` | `average_controllability_a` | 8 | pass | 1.000 | 0.000 | 0.1259 | 0.1250 |
| `prime/swe-grep` | `koopman_dmd_c` | 8 | pass | 1.000 | 0.000 | 0.1259 | 0.1250 |
| `prime/swe-grep` | `original_graph` | 10 | pass | 1.000 | 0.000 | 0.0993 | 0.1000 |
| `primeintellect/math500` | `parametric_b` | 7 | pass | 1.000 | 0.000 | 0.1435 | 0.1429 |
| `primeintellect/math500` | `average_controllability_a` | 8 | pass | 1.000 | 0.000 | 0.1256 | 0.1250 |
| `primeintellect/math500` | `koopman_dmd_c` | 8 | pass | 1.000 | 0.000 | 0.1256 | 0.1250 |
| `primeintellect/math500` | `original_graph` | 10 | pass | 1.000 | 0.000 | 0.1005 | 0.1000 |

## Evidence

- `runs/stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_average_controllability_a/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_koopman_dmd_c/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_original_graph/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_average_controllability_a/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_koopman_dmd_c/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_original_graph/summary.json`

Locked baseline configs were written under `configs/controllability/preregistration/baselines_locked_stage1_v2_*.yaml`.

## Interpretation

The fixed-feature behavioral encoder covers 100% of the 5,000 base rollouts in each Tier A environment. Re-encoding identical trajectories produces zero distribution shift, and random coordinate labels are indistinguishable from chance under a prompt-held-out nearest-centroid classifier. This clears the Stage 2 v2 encoder sanity gate for B as primary and for A/C/original-graph comparators.

Temperature and SAE baselines are locked as downstream Stage 4 policies rather than evaluated here: no matched-temperature or SAE-steered trajectory sets exist yet. PCA baseline metadata is locked from the encoded base-rollout feature covariance for each run.
