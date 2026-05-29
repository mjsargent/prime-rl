# Stage 1 v2 Results

Date: 2026-05-07

Stage 1 v2 is a fresh graph-free validation branch. It does not overwrite the official v1 result: the original median-kernel graph estimator remains failed and is retained as a baseline/negative control. The v2 primary estimator is Formulation B, parametric neural eigenfunctions. Formulations A and C are locked comparators.

## Gate Result

Both Tier A environments passed Stage 1 v2.

| env_id | run_id | gate | selected method | selected pair | K | B inverse-energy effective rank | A generalized effective rank | C held-out R2 | original graph |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `prime/swe-grep` | `stage1_v2_graph_free_Qwen_Qwen3_8B_prime_swe_grep` | pass | `parametric_b` | `(20,22)` | 6 | 7.71 | 28.47 | 0.986 | fail |
| `primeintellect/math500` | `stage1_v2_graph_free_Qwen_Qwen3_8B_primeintellect_math500` | pass | `parametric_b` | `(12,14)` | 7 | 6.61 | 28.67 | 0.982 | fail |

## Evidence

- `runs/stage1_v2_graph_free_Qwen_Qwen3_8B_prime_swe_grep/summary.json`
- `runs/stage1_v2_graph_free_Qwen_Qwen3_8B_prime_swe_grep/method_comparison.json`
- `runs/stage1_v2_graph_free_Qwen_Qwen3_8B_primeintellect_math500/summary.json`
- `runs/stage1_v2_graph_free_Qwen_Qwen3_8B_primeintellect_math500/method_comparison.json`

Primary B runs:

- `runs/stage1_v2_parametric_b_Qwen_Qwen3_8B_prime_swe_grep/summary.json`
- `runs/stage1_v2_parametric_b_Qwen_Qwen3_8B_primeintellect_math500/summary.json`

Locked next-stage configs:

- `configs/controllability/preregistration/headline_locked_stage1_v2_prime_swe_grep.yaml`
- `configs/controllability/preregistration/headline_locked_stage1_v2_primeintellect_math500.yaml`

## Interpretation

The v2 result supports the claim that the v1 Stage 1 failure was estimator-specific. The original graph estimator still fails on both Tier A environments, but the graph-free branch recovers multidimensional controllability structure with B as primary and A/C as independent comparators.

Stage 2 is now authorized for the v2 branch only. Stage 2 should use B as primary, carry A and C as locked comparators, retain the original graph estimator as a negative/discretization baseline, and report chance-corrected identifiability as `accuracy - 1/K` because the selected K differs by environment and may differ by formulation.
