# Stage 6 v2 Geometric-Behavioral Correlation Results

Timestamp: 2026-05-11T23:32:02Z

## Evidence

- Stage 6 run: `runs/stage6_v2_Qwen_Qwen3_8B_tier_a_reduced/summary.json`
- Pooled correlations: `runs/stage6_v2_Qwen_Qwen3_8B_tier_a_reduced/correlation_pooled.json`
- Per-environment correlations: `runs/stage6_v2_Qwen_Qwen3_8B_tier_a_reduced/correlation_per_env.json`
- Frequency-band correlations: `runs/stage6_v2_Qwen_Qwen3_8B_tier_a_reduced/correlation_per_freq_band.json`
- Scatter data: `runs/stage6_v2_Qwen_Qwen3_8B_tier_a_reduced/scatter_data.parquet`

## Gate Result

Gate: `pass`.

Stage 6 is descriptive: weak or null correlations are reportable rather than failure conditions.

## Result

The reduced Stage 4/5 data does not show a meaningful pooled geometric-behavioral correlation for the primary `parametric_b` coordinates.

Pooled over the Tier A per-coordinate rows:

- `n = 13`
- Controllability-score vs identifiability Pearson `r = -0.0173`, `p = 0.9554`
- Controllability-score vs identifiability Spearman `rho = -0.3964`, `p = 0.1800`
- Log-gamma vs identifiability Pearson `r = 0.0002`, `p = 0.9996`
- Rho vs identifiability Pearson `r = 0.0315`, `p = 0.9187`

Per environment:

- `swe_grep`: controllability-score Pearson `r = -0.4572`, `p = 0.3620`; Spearman `rho = -0.3381`, `p = 0.5122`.
- `math500`: controllability-score Pearson `r = 0.3214`, `p = 0.4821`; Spearman `rho = 0.1581`, `p = 0.7349`.

## Interpretation

This supports a null/weak Stage 6 finding on the reduced trajectory set: Stage 3 geometric writability proxies do not explain the Stage 5 per-coordinate identifiability values at this scale.

This does not overturn the Stage 1-3 construction diagnostics. It says the current reduced HF patched-decoding Stage 4/5 data is too weak or too small to show the expected geometry-to-behavior relationship.

## Limitations

- Stage 6 uses reduced Stage 4/5 v2 trajectories, not a full verifier-scored Stage 4 sweep.
- Per-coordinate identifiability is available only for the primary `parametric_b` formulation in the current Stage 5 artifact.
- The pooled dataset has only 13 per-coordinate rows, so correlation estimates are coarse.
