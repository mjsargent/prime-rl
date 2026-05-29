# Method Validation Results

Date: 2026-05-07

Status: diagnostics after the official Tier A Stage 1 stop. These runs do not reopen Stage 1, do not change preregistered gates, and do not authorize Stage 2. They test whether the observed low natural eigengap dimension is explained by obvious implementation or methodology confounds.

## Evidence

- `runs/stage1_confound_diagnostics_Qwen_Qwen3_8B_prime_swe_grep_20260507T065514Z/summary.json`
- `runs/stage1_confound_diagnostics_Qwen_Qwen3_8B_primeintellect_math500_20260507T065650Z/summary.json`
- `runs/stage1_layer_pair_sweep_Qwen_Qwen3_8B_prime_swe_grep_20260507T065859Z/summary.json`
- `runs/stage1_layer_pair_sweep_Qwen_Qwen3_8B_prime_swe_grep_20260507T065946Z/summary.json`
- `runs/stage1_layer_pair_sweep_Qwen_Qwen3_8B_primeintellect_math500_20260507T070035Z/summary.json`
- `runs/stage1_layer_pair_sweep_Qwen_Qwen3_8B_primeintellect_math500_20260507T070135Z/summary.json`

## Findings

1. Chart dimension ceiling is not supported as the explanation.
   Sweeping chart dimension `d in {32,64,128,256}` did not recover a multidimensional natural eigengap. The maximum natural `K*` seen was 3 on `swe-grep` and 2 on `math500`.

2. Chart collapse is not supported.
   PCA and transition-PCA spectra are not collapsed. `swe-grep` had maximum PCA effective rank 23.89 and transition-PCA effective rank 15.06. `math500` had maximum PCA effective rank 99.72 and transition-PCA effective rank 92.67.

3. Residual metric is a real local-rank confound.
   Local controllability rank depends strongly on `C_l`. On `swe-grep`, best median effective rank was 13.39 for `I`, 8.64 for `Cov(h_l)`, and 6.94 for `Cov(delta h_l)`. On `math500`, best median effective rank was 13.02 for `I`, 7.27 for `Cov(h_l)`, and 4.38 for `Cov(delta h_l)`. This means `Cov(delta h_l)` is constraining local rank, but it does not by itself explain the low global eigengap because the current eigengap diagnostic is still a residual-chart proxy rather than the full anisotropic `G_tau` graph.

4. Median-kernel bandwidth is not supported as the simple explanation.
   Scaling the median bandwidth by `{0.5,1,2,4}` did not recover `K* >= 4` under the kernel graph. Natural `K*` stayed in `1..3` on `swe-grep` and `1..2` on `math500`.

5. k-NN graph construction is a real graph-discretization confound.
   k-NN graphs often produced much larger natural `K*`, and several of those subspaces were stable between 100k and 200k samples. Examples: `swe-grep` `(12,14)` with 15 neighbors gave `K*=6` and natural-K overlap 0.9996; `swe-grep` `(4,6)` with 30 neighbors gave `K*=14` and overlap 0.9990; `math500` `(4,6)` with 60 neighbors gave `K*=12` and overlap 0.9977. This is post-hoc relative to the stopped Stage 1 run, so it is not a pass, but it should be treated as a serious discretization issue in the next preregistered validation branch.

6. Local rank is richer than global kernel eigengap.
   The saved layer-pair sweeps and the new metric sweeps support the interpretation that local writable directions exist but do not aggregate into a high-dimensional global median-kernel basis under the current estimator. The unresolved question is whether this is intrinsic state-dependent rotation or a graph/metric estimator artifact.

## Not Resolved

- Full residual metric sweep in the true anisotropic graph: requires caching or recomputing `M_tau = J_tau C_l J_tau^T` matrices, not just local eigenvalues.
- `lambda` sweep for `G_tau = (M_tau + lambda I)^-1`: requires the true controllability-metric graph.
- Mean-vs-local `E[M_tau]` diagnostic: current saved sweeps contain local eigenvalues but not full `M_tau` matrices.
- Patch-position diagnostic: requires fresh HF Jacobian runs at observation-boundary, response-start, last-token, and multi-position patch sites.
- LayerNorm across-state consistency: requires finite-difference and singular-spectrum checks across states with different residual norms.
- Rollout heterogeneity: requires new rollout distributions with mixed temperatures/system prompts.
- Open-ended environment validation: requires a new non-Tier-A rollout/cache path.

## Current Interpretation

The low natural `K*` under the preregistered median-kernel Stage 1 estimator is not explained by too-small chart dimension, collapsed PCA charts, or a simple bandwidth setting. The two strongest remaining confounds are graph discretization and the residual metric used in the local controllability tensor. The k-NN result is especially important: it shows that a different graph estimator can expose stable higher-dimensional structure, but it was not preregistered and therefore cannot be used to retroactively pass Stage 1.

The Tier A Stage 1 stop remains the official protocol result. The next valid move is a new preregistered validation branch that implements the full anisotropic controllability graph, sweeps `C_l` and `lambda`, and treats kernel, k-NN, landmark Nyström, and parametric eigenfunctions as locked discretization alternatives.

## Graph-Free Follow-Up: Formulations A and C

Additional graph-free diagnostics were run after the observations above:

- `runs/stage1_graph_free_spectra_Qwen_Qwen3_8B_prime_swe_grep_20260507T071952Z/summary.json`
- `runs/stage1_graph_free_spectra_Qwen_Qwen3_8B_primeintellect_math500_20260507T072140Z/summary.json`

Formulation A, average-controllability spectrum, solves `E[M_tau] v = lambda Cov(z) v` directly in the chart space. It avoids graph construction entirely. On `swe-grep`, the best generalized effective rank was 28.47 at `(4,6)` with `C_l = Cov(h_l)`. On `math500`, the best generalized effective rank was 28.67 at `(12,14)` with `C_l = Cov(h_l)`. Across both tasks and all three tested residual metrics (`I`, `Cov(h_l)`, `Cov(delta h_l)`), the average-controllability spectra were high-rank in the 32-dimensional chart. This contradicts the hypothesis that the aggregate controllability operator itself is intrinsically rank-1 or rank-3.

Formulation C, Koopman/DMD-style suffix operator, fits a linear operator from patch-layer chart features to readout-layer chart features using aligned residual pairs. It also avoids graph construction. On `swe-grep`, the best held-out `R^2` was 0.986 at `(4,6)` with 32 features; on `math500`, the best held-out `R^2` was 0.982 at `(12,14)` with 16 features. The DMD operators had high effective ranks and many large-magnitude modes. This indicates that the patch-to-readout suffix map has rich graph-free linear structure in PCA feature space.

These two graph-free diagnostics change the interpretation of the Stage 1 failure. The median-kernel graph estimator appears to be the main bottleneck for global dimensionality, not the absence of high-rank aggregate controllability or suffix dynamics. The result still does not pass Stage 1 because it is post-hoc and not the preregistered estimator, but it strongly motivates a new preregistered graph-free validation branch.

Recommended next branch:

1. Treat average-controllability modes as the linear graph-free baseline.
2. Treat Koopman/DMD modes as the dynamics-first graph-free baseline.
3. Implement parametric neural eigenfunctions as the graph-free replacement for the anisotropic Laplacian if the project still needs nonlinear coordinates.
4. Re-run construction validation with all three graph-free operators preregistered before any steering or identifiability stage.

## Formulation B Diagnostic

Parametric neural eigenfunction diagnostics were run after the A/C graph-free follow-up:

- `runs/stage1_parametric_eigenfunctions_Qwen_Qwen3_8B_prime_swe_grep_20260507T074057Z/summary.json`
- `runs/stage1_parametric_eigenfunctions_Qwen_Qwen3_8B_primeintellect_math500_20260507T074250Z/summary.json`

This implementation trains small MLP eigenfunctions on HF-computed suffix-Jacobian states with the anisotropic Rayleigh energy and output orthogonality penalties. It is diagnostic-scale, not a fresh preregistered Stage 1 v2 gate.

On `swe-grep`, B recovered nontrivial parametric structure. The best pair by held-out mean Rayleigh energy was `(20,22)`, with inverse-energy effective rank 7.71 and energy-gap `K*=6`. The other pairs also produced multi-coordinate spectra: `(4,6)` had inverse-energy effective rank 6.91 and `K*=4`; `(12,14)` had inverse-energy effective rank 6.28 and `K*=6`.

On `math500`, B was mixed but still not rank-collapsed. The best pair by held-out mean Rayleigh energy was `(4,6)`, with inverse-energy effective rank 7.26 and energy-gap `K*=3`. The `(12,14)` pair had inverse-energy effective rank 6.61 and energy-gap `K*=7`, but with higher mean Rayleigh energy. This should be interpreted as evidence that B can represent multiple coordinates on math500, but the stable operating pair and model-selection criterion need to be locked in a v2 preregistration rather than chosen post-hoc.

The overall graph-free picture is now:

- A: high-rank average controllability on both Tier A tasks.
- B: nontrivial parametric eigenfunctions, clearly `K>=4` on `swe-grep` and mixed-but-not-collapsed on `math500`.
- C: high-R2, high-rank suffix dynamics on both Tier A tasks.
- Original graph estimator: failed under the preregistered median-kernel Stage 1 gate.

This strengthens the conclusion that the original failure was estimator-specific. The next official protocol should be a fresh `stage1_v2_graph_free` branch with B as primary, A and C as locked comparators, and the original graph estimator retained as a negative/discretization baseline.

## Stage 2 Across Formulations

Stage 2 should be shared in implementation but run separately per formulation. The invariant pieces are encoder end-to-end coverage, null intervention calibration, random-direction chance calibration, temperature-as-skill, and SAE baselines where available.

The formulation-specific pieces are:

- A: constant gradients from linear generalized eigenvectors. PCA is the closest baseline because both are linear; Stage 2 must show controllability weighting is not merely PCA rotation.
- B: gradients come from autodiff through the trained eigenfunction network. PCA is a clean linear baseline against nonlinear controllability eigenfunctions. B should be primary if v2 passes.
- C: gradients come from the Koopman feature dictionary. PCA is a variance baseline, and A/B are controllability baselines; C tests whether suffix dynamics alone organize behavior.
- Original graph: retain as a baseline/negative control, not as primary, unless a new preregistered graph variant passes.

Each formulation should lock its own `K`, coordinate set, and trust-region `eta_star`. Downstream cross-formulation comparisons should use chance-corrected identifiability, for example `accuracy - 1/K`, because natural `K` can differ by formulation.
