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

## Diagnostic Amendment

After this failed single-pair run, a direct suffix-Jacobian layer-pair diagnostic was run because the original run only tested `(12, 18)` and did not satisfy the intended layer-pair heatmap coverage.

Diagnostic run: `stage1_layer_pair_sweep_Qwen_Qwen3_8B_primeintellect_math500_20260507T043447Z`

Evidence: `runs/stage1_layer_pair_sweep_Qwen_Qwen3_8B_primeintellect_math500_20260507T043447Z/summary.json`

That diagnostic computes `M = J C J^T` directly on a bounded sample with `chart_dim=16`, 8 Jacobian states, and the grid `patch_layer in [4, 8, 12, 16, 20]`, `readout_layer in {patch+2, patch+4, L-1}`. It found layer pairs exceeding the rank-4 threshold:

| Patch layer | Readout layer | Median effective rank |
|---:|---:|---:|
| 4 | 6 | 4.3819 |
| 12 | 14 | 4.3539 |

This means the original failed run should be treated as a failed `(12, 18)` attempt, not as definitive evidence that math500 has no viable controllability construction. The next valid action is a corrected Stage 1 implementation/run that includes the planned layer-pair sweep. Stage 2 still must not run until a full Stage 1 gate passes and writes a locked headline config.

## Corrected Stage 1 Run

Corrected run: `stage1_Qwen_Qwen3_8B_primeintellect_math500_20260507T044610Z`

Evidence: `runs/stage1_Qwen_Qwen3_8B_primeintellect_math500_20260507T044610Z/summary.json`

This run used the direct layer-pair sweep artifact as an input, selected the best layer pair by `effective_rank_median`, and gated chart/discretization/sample convergence on cached residuals at the selected readout layer.

Selected layer pair: `(patch_layer=4, readout_layer=6)`

| Metric | Value | Threshold | Status |
|---|---:|---:|---|
| Median chart-invariance overlap on leading 10 | 0.7000 | >= 0.7 | pass |
| Graph-vs-parametric discretization overlap on leading 10 | 0.8736 | >= 0.7 | pass |
| Sample-size convergence, 10k vs 50k | 0.7345 | >= 0.85 | fail |
| Effective-rank median | 4.3819 | >= 4 | pass |

The corrected Stage 1 run still failed the preregistered gate because sample-size convergence did not pass. No headline lock was written at `configs/controllability/preregistration/headline_locked_primeintellect_math500.yaml`.
