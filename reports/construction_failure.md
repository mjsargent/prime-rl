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

## Third-Pass Eigengap and Large-N Diagnostic

Diagnostic run: `stage1_eigengap_convergence_Qwen_Qwen3_8B_primeintellect_math500_20260507T050554Z`

Evidence: `runs/stage1_eigengap_convergence_Qwen_Qwen3_8B_primeintellect_math500_20260507T050554Z/summary.json`

This diagnostic was run before treating Stage 1 as finally failed. It used 200,000-state residual caches at readout layers 6 and 14, fit chart bases at `N in {10k, 50k, 100k, 200k}`, and compared fixed-anchor graph eigenspaces at the natural eigengap and at leading 10.

Result:

| Patch layer | Readout layer | Natural K | Natural-K 100k vs 200k | Leading-10 100k vs 200k |
|---:|---:|---:|---:|---:|
| 4 | 6 | 1 | 1.0000 | 0.9777 |
| 12 | 14 | 1 | 0.9998 | 0.9800 |

This supports the narrower conclusion that the original sample-convergence failure is tied to the preregistered 10k-vs-50k proxy convergence check, not to rank degeneracy or chart/discretization instability.

## Official Math500 Stage 1 Status

The post-hoc eigengap diagnostic found a natural cutoff of `K*=1` on both tested layer pairs. That is a substantive finding, but it does not support the multidimensional DIAYN-style empirical program.

Math500 is therefore treated as failed under the preregistered Stage 1 gate. No math500 headline lock is retained, and Stage 2 must not be run on math500 unless a future preregistered protocol explicitly reopens it.

## Swe-Grep Stage 1 Attempt

Rollout evidence: `runs/rollouts/prime_swe_grep_Qwen_Qwen3_8B/summary.json`

Stage 1 run: `stage1_Qwen_Qwen3_8B_prime_swe_grep_20260507T061706Z`

Evidence: `runs/stage1_Qwen_Qwen3_8B_prime_swe_grep_20260507T061706Z/summary.json`

The Tier A code-retrieval environment was run next, using `prime/swe-grep` through prime-rl's rollout machinery. The rollout set contains 5,000 trajectories and 0 failed cells.

Layer-pair sweep evidence: `runs/stage1_layer_pair_sweep_Qwen_Qwen3_8B_prime_swe_grep_20260507T060653Z/summary.json`

Best local-rank layer pair: `(patch_layer=20, readout_layer=22)`, with effective-rank median `6.9441`.

Strict preregistered Stage 1 gate:

| Metric | Value | Threshold | Status |
|---|---:|---:|---|
| Median chart-invariance overlap on leading 10 | 0.7761 | >= 0.7 | pass |
| Graph-vs-parametric discretization overlap on leading 10 | 0.7620 | >= 0.7 | pass |
| Sample-size convergence, 10k vs 50k | 0.8097 | >= 0.85 | fail |
| Effective-rank median | 6.9441 | >= 4 | pass |

Eigengap diagnostic run: `stage1_eigengap_convergence_Qwen_Qwen3_8B_prime_swe_grep_20260507T061824Z`

Evidence: `runs/stage1_eigengap_convergence_Qwen_Qwen3_8B_prime_swe_grep_20260507T061824Z/summary.json`

| Patch layer | Readout layer | Natural K | Natural-K 100k vs 200k | Leading-10 100k vs 200k |
|---:|---:|---:|---:|---:|
| 20 | 22 | 1 | 0.0000 | 0.8931 |
| 12 | 14 | 2 | 1.0000 | 0.9983 |
| 4 | 6 | 3 | 0.9999 | 0.9963 |

Swe-grep does not show the clean multidimensional `K* >= 4` structure needed for the planned DIAYN-style downstream empirical program. It also fails the original Stage 1 sample-convergence gate. The project is therefore stopped at Stage 1 rather than advanced to Stage 2.
