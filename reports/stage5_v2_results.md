# Stage 5 v2 Identifiability Results

Timestamp: 2026-05-07T09:49:22Z

## Evidence

- Stage 5 run: `runs/stage5_v2_Qwen_Qwen3_8B_tier_a_stage5/summary.json`
- Inputs: eight Stage 4 expansion runs, covering Tier A environments (`prime/swe-grep`, `primeintellect/math500`) across `parametric_b`, `average_controllability_a`, `koopman_dmd_c`, and `original_graph`.
- Gate: `conditional`.

## Result

The strong Stage 5 headline gate did not pass. The gate required coherent `parametric_b` identifiability to exceed the PCA proxy with conservative non-overlap on both Tier A environments.

- `swe_grep`: `parametric_b_coherent_minus_pca_proxy = 0.0417`, conservative CI `[0.0000, 0.0833]`; non-overlap condition failed because the lower bound is not strictly positive.
- `math500`: `parametric_b_coherent_minus_pca_proxy = 0.0000`, conservative CI `[0.0000, 0.0000]`.

The supported claim is therefore descriptive/conditional: at this reduced Stage 4 scale, the Stage 5 discriminator does not recover a strong coherent identifiability advantage for the primary parametric-B spectral coordinates.

## Cross-Formulation Notes

On the reduced swe-grep run, `average_controllability_a` showed the only positive chance-corrected identifiability signal among the evaluated formulations: `I = 0.0417`, CI `[0.0000, 0.0833]`. `parametric_b` and `koopman_dmd_c` were at chance, and the original-graph/PCA proxy was below chance. On math500, all evaluated formulations were at chance.

The original graph method remains a negative/discretization baseline. Its Stage 4 expansion uses a PCA-proxy off-sample basis because the failed graph estimator did not produce reusable off-sample eigenfunctions.

## Limitations

- Stage 5 v2 uses reduced Stage 4 HF patched-decoding trajectories, not full verifier-scored tool-interactive rollouts.
- Temperature-as-skill and SAE-feature baselines are logged as `not_evaluated`; their Stage 4 rollouts/artifacts were not available.
- Quality-conditioning sweeps are limited because reduced Stage 4 trajectories have unset reward/quality values.
- The held-out split has only two test prompts per environment, so the confidence intervals are coarse.

## Figures

- `paper/figures/stage5_identifiability_bars/figure.pdf`
- `paper/figures/stage5_per_coordinate_identifiability/figure.pdf`
- `paper/figures/stage5_cross_formulation_identifiability/figure.pdf`
- `paper/figures/stage5_confusion_matrix_headline/figure.pdf`
- `paper/figures/stage5_quality_conditioning_sensitivity/figure.pdf`
