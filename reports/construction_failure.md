# Stage 1 Construction Failure

Date: 2026-05-07

Run ID: `stage1_Qwen_Qwen3_8B_primeintellect_math500_20260507T040948Z`

Evidence: `runs/stage1_Qwen_Qwen3_8B_primeintellect_math500_20260507T040948Z/summary.json`

## Result

Stage 1 construction validation failed for `Qwen/Qwen3-8B` on `primeintellect/math500`.

The run produced the required Stage 1 artifacts, but the preregistered gate did not pass:

| Metric | Value | Threshold | Status |
|---|---:|---:|---|
| Median chart-invariance overlap on leading 10 | 0.7554 | >= 0.7 | pass |
| Graph-vs-parametric discretization overlap on leading 10 | 0.8558 | >= 0.7 | pass |
| Sample-size convergence, 10k vs 50k | 0.8021 | >= 0.85 | fail |
| Effective-rank median | 1.0251 | >= 4 | fail |

## Stop Condition

Per the implementation plan, Stage 1 failure stops the pipeline. Stage 2 and later stages must not run from this result, and no empirical manuscript claim should be made from this failed construction run.

## Inputs

- Rollouts: `runs/rollouts/primeintellect_math500_Qwen_Qwen3_8B/trajectories.jsonl`
- Rollout count: 5,000
- Residual cache: `runs/activations/primeintellect_math500_Qwen_Qwen3_8B/residuals_l18.npy`
- Residual cache shape: 50,000 x 4,096
- Residual cache manifest: `runs/activations/primeintellect_math500_Qwen_Qwen3_8B/residuals_l18_manifest.json`

## Interpretation

The failed effective-rank diagnostic indicates that, under the current Stage 1 construction and `cov_delta_h` residual metric, the candidate controllability basis is effectively one-dimensional rather than meeting the minimum rank of 4. The failed sample-size convergence diagnostic indicates the leading subspace was not stable enough between 10k and 50k samples.

The next valid action is analysis of the construction failure, not continuation to Stage 2.
