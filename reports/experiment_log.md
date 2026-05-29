# Controllability Spectroscopy Experiment Log

Date: 2026-05-08

Branch: `feat/controllability-spectroscopy-port`

Code/evidence commit before this log: `1eb1d6c feat: port controllability spectroscopy work`

## Reproducibility Status

The branch contains the controllability package, configs, paper figure sources, reports, audit table, and committed run evidence needed to inspect the completed reduced-scale experiments.

The full node `runs/` artifact tree was also copied locally from `/root/prime-rl/runs/` into this checkout:

- Local `runs/`: 8.9G, 1932 files.
- Node `runs/`: 8.9G, 1872 files.
- `rsync --dry-run` after transfer reported 0 files remaining to transfer from the node.
- The local tree is a superset because smaller evidence files had already been ported before the full node pull.

The pushed Git branch is not, by itself, a complete large-artifact mirror. The large activation arrays, rollout shards, and local databases are present in this local checkout under `runs/`, but the repository keeps `runs/` mostly ignored and should not push multi-GB artifacts to ordinary Git history. A fully reproducible remote clone should either preserve this local checkout or publish `runs/` to object storage, Git LFS, DVC, or an equivalent artifact store.

Re-running experiments still requires the external runtime dependencies from the original plan: model checkpoint access, Prime Hub/verifiers environments, valid API keys where applicable, GPU hardware, and a compatible Linux/uv environment.

Cluster reestablishment on 2026-05-11:

- New cluster head node: `matthew@176.56.204.42`.
- Working tree restored at `/data/matthew/controllability-spectroscopy`.
- `/data` had about 8 TB free when checked.
- `uv sync --all-extras` completed successfully in the remote checkout.
- Slurm tools are available with `/data/slurm/bin` on `PATH`.
- GitHub SSH auth was not available for `matthew`, so the repo was restored by `rsync` from the local branch.
- `.env`, `.venv`, and `.claude` were intentionally not copied. Large activation arrays under `runs/activations*` were also not fully copied during the selective sync; the Stage 6 reduced-data run does not require them.

## Completed Stages

### Preflight

Status: pass, with the user-approved storage waiver.

Evidence:

- `runs/preflight/preflight_complete.json`
- `runs/preflight/repo_state.log`
- `runs/preflight/tooling.log`
- `runs/preflight/prime_rl_inventory.md`
- `runs/audit_table.md`

Notes:

- Hub environments were installed and smoke-verified.
- Qwen3-8B inference and model-loading checks were completed after API/key reauth issues were resolved.
- The original 2 TB free-space requirement was waived by the user before continuing.

### Stage 1 v1: Original Graph Construction Validation

Status: failed/diagnostic.

Evidence:

- `runs/stage1_Qwen_Qwen3_8B_primeintellect_math500_20260507T040948Z/summary.json`
- `runs/stage1_Qwen_Qwen3_8B_prime_swe_grep_20260507T061706Z/summary.json`
- `runs/stage1_eigengap_convergence_Qwen_Qwen3_8B_primeintellect_math500_20260507T050554Z/summary.json`
- `runs/stage1_eigengap_convergence_Qwen_Qwen3_8B_prime_swe_grep_20260507T061824Z/summary.json`
- `reports/construction_failure.md`

Result:

- The preregistered median-kernel graph estimator did not pass the Stage 1 gate.
- Follow-up layer-pair and eigengap diagnostics showed the original failure was not simply local rank collapse; it was mainly global graph/discretization instability and low natural graph eigengap on the tested Tier A tasks.

### Stage 1 Diagnostics: Confounds and Graph-Free Formulations

Status: completed as post-v1 diagnostics.

Evidence:

- `runs/stage1_confound_diagnostics_Qwen_Qwen3_8B_prime_swe_grep_20260507T065037Z/summary.json`
- `runs/stage1_confound_diagnostics_Qwen_Qwen3_8B_primeintellect_math500_20260507T065305Z/summary.json`
- `runs/stage1_graph_free_spectra_Qwen_Qwen3_8B_prime_swe_grep_20260507T071952Z/summary.json`
- `runs/stage1_graph_free_spectra_Qwen_Qwen3_8B_primeintellect_math500_20260507T072140Z/summary.json`
- `reports/method_validation_results.md`
- `reports/method_validation_and_benchmark_backlog.md`

Result:

- Formulation A, linear average-controllability modes, recovered rich global structure.
- Formulation C, Koopman/DMD modes, recovered rich dynamical structure with high held-out fit.
- These alternatives were logged as comparators and benchmarks, not as silent replacements for the failed graph estimator.

### Stage 1 v2: Graph-Free Construction Validation

Status: pass on both Tier A environments.

Primary method: Formulation B, parametric neural eigenfunctions.

Comparators: Formulation A, Formulation C, and the original graph estimator as a negative/discretization baseline.

Evidence:

- `runs/stage1_v2_graph_free_Qwen_Qwen3_8B_prime_swe_grep/summary.json`
- `runs/stage1_v2_graph_free_Qwen_Qwen3_8B_primeintellect_math500/summary.json`
- `runs/stage1_v2_parametric_b_Qwen_Qwen3_8B_prime_swe_grep/summary.json`
- `runs/stage1_v2_parametric_b_Qwen_Qwen3_8B_primeintellect_math500/summary.json`
- `reports/stage1_v2_results.md`

Result:

- `prime/swe-grep`: selected pair `(20,22)`, `K=6`, B inverse-energy effective rank 7.71.
- `primeintellect/math500`: selected pair `(12,14)`, `K=7`, B inverse-energy effective rank 6.61.
- Original graph remains failed and is retained as a negative control.

### Stage 2 v2: Baseline Encoder Sanity

Status: pass on all eight environment/formulation runs.

Evidence:

- `runs/stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_average_controllability_a/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_koopman_dmd_c/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_prime_swe_grep_original_graph/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_average_controllability_a/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_koopman_dmd_c/summary.json`
- `runs/stage2_v2_Qwen_Qwen3_8B_primeintellect_math500_original_graph/summary.json`
- `reports/stage2_v2_results.md`

Result:

- Encoder coverage was 100% for both Tier A environments.
- Null-intervention Wasserstein distance was 0.000 in all runs.
- Random-label accuracy matched chance for each formulation-specific `K`.
- Temperature and SAE baselines were not evaluated here because matched steered trajectory sets were not yet available.

### Stage 3 v2: Per-Coordinate Geometric Profile

Status: pass/descriptive on both Tier A environments.

Evidence:

- `runs/stage3_v2_Qwen_Qwen3_8B_prime_swe_grep/summary.json`
- `runs/stage3_v2_Qwen_Qwen3_8B_prime_swe_grep/aggregated_per_coordinate.json`
- `runs/stage3_v2_Qwen_Qwen3_8B_prime_swe_grep/linearization_sweep.parquet`
- `runs/stage3_v2_Qwen_Qwen3_8B_primeintellect_math500/summary.json`
- `runs/stage3_v2_Qwen_Qwen3_8B_primeintellect_math500/aggregated_per_coordinate.json`
- `runs/stage3_v2_Qwen_Qwen3_8B_primeintellect_math500/linearization_sweep.parquet`
- `reports/stage3_v2_results.md`

Result:

- `prime/swe-grep`: primary `K=6`, mean reachability proxy 0.848, mean gain 114.265, minimum `eta_star=0.101`.
- `primeintellect/math500`: primary `K=7`, mean reachability proxy 0.798, mean gain 9.618, minimum `eta_star=0.057`.
- Stage 3 used saved spectra and analytic trust-region surrogates, not full steered verifier rollouts.

### Stage 4 v2: Reduced Trajectory Generation

Status: pass for the reduced HF patched-decoding runs; not the full prime-rl tool-interactive Stage 4 sweep.

Pilot evidence:

- `runs/stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b_pilot_rerun3/summary.json`
- `runs/stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b_pilot_rerun3/summary.json`
- `reports/stage4_v2_results.md`

Stage 5 expansion evidence:

- `runs/stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_parametric_b_stage5/summary.json`
- `runs/stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_average_controllability_a_stage5/summary.json`
- `runs/stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_koopman_dmd_c_stage5/summary.json`
- `runs/stage4_v2_Qwen_Qwen3_8B_prime_swe_grep_original_graph_stage5/summary.json`
- `runs/stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_parametric_b_stage5/summary.json`
- `runs/stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_average_controllability_a_stage5/summary.json`
- `runs/stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_koopman_dmd_c_stage5/summary.json`
- `runs/stage4_v2_Qwen_Qwen3_8B_primeintellect_math500_original_graph_stage5/summary.json`

Result:

- Passing B pilot rerun generated 74/74 trajectories per Tier A environment with zero failed jobs.
- Stage 5 expansion generated reduced trajectory sets for B, A, C, and original graph/PCA-proxy across both Tier A environments.
- Trust-region violation, realized-vs-predicted displacement, trajectory length, and coherence-by-edit-norm diagnostics were logged and plotted.
- Rewards and success are unset for these generated trajectories because this run used HF patched decoding from environment prompts, not full verifier-scored tool interaction.

### Stage 5 v2: Identifiability Measurement

Status: conditional; strong headline gate did not pass.

Evidence:

- `runs/stage5_v2_Qwen_Qwen3_8B_tier_a_stage5/summary.json`
- `reports/stage5_v2_results.md`

Result:

- `prime/swe-grep`: primary B coherent advantage over the PCA proxy was 0.0417 with conservative CI `[0.0000, 0.0833]`; non-overlap failed because the lower bound was not strictly positive.
- `primeintellect/math500`: primary B coherent advantage over the PCA proxy was 0.0000 with CI `[0.0000, 0.0000]`.
- On reduced swe-grep, average-controllability A showed the only positive chance-corrected signal, `I=0.0417` with CI `[0.0000, 0.0833]`.
- On math500, all evaluated formulations were at chance.

Supported claim:

- The current evidence supports a diagnostic/methodological claim: graph-free construction, encoder sanity, geometric profiling, and reduced-scale steering infrastructure work, but reduced Stage 5 does not support a strong coherent skill-identifiability advantage for primary B.

Unsupported claim:

- Do not claim that spectral coordinates beat PCA/temperature/SAE baselines under coherence conditioning. That strong headline is not supported by the current audit table.

### Stage 6 v2: Geometric-Behavioral Correlation

Status: pass/descriptive on the reduced Tier A data.

Evidence:

- `runs/stage6_v2_Qwen_Qwen3_8B_tier_a_reduced/summary.json`
- `runs/stage6_v2_Qwen_Qwen3_8B_tier_a_reduced/correlation_pooled.json`
- `runs/stage6_v2_Qwen_Qwen3_8B_tier_a_reduced/correlation_per_env.json`
- `runs/stage6_v2_Qwen_Qwen3_8B_tier_a_reduced/correlation_per_freq_band.json`
- `runs/stage6_v2_Qwen_Qwen3_8B_tier_a_reduced/scatter_data.parquet`
- `reports/stage6_v2_results.md`

Result:

- Pooled `parametric_b` controllability-score vs identifiability Pearson `r=-0.0173`, `p=0.9554`.
- Pooled Spearman `rho=-0.3964`, `p=0.1800`.
- Per-environment correlations were also non-significant.

Supported claim:

- On the reduced Stage 4/5 v2 data, Stage 3 geometric writability proxies do not explain Stage 5 per-coordinate identifiability. This is a reportable null/weak Stage 6 finding, not a gate failure.

## Figures Prepared

Stage 1:

- `paper/figures/stage1_chart_invariance/figure.pdf`
- `paper/figures/stage1_discretization_invariance/figure.pdf`
- `paper/figures/stage1_sample_size_convergence/figure.pdf`
- `paper/figures/stage1_layer_pair_heatmap/figure.pdf`
- `paper/figures/stage1_effective_rank_distribution/figure.pdf`
- `paper/figures/stage1_cumulative_gain_curve/figure.pdf`
- `paper/figures/stage1_cross_formulation_comparison/figure.pdf`

Stage 2:

- `paper/figures/stage2_null_intervention_distribution/figure.pdf`
- `paper/figures/stage2_random_baseline_calibration/figure.pdf`
- `paper/figures/stage2_baseline_identifiability_profile/figure.pdf`

Stage 3:

- `paper/figures/stage3_per_coordinate_geometry/figure.pdf`
- `paper/figures/stage3_linearization_budget/figure.pdf`
- `paper/figures/stage3_reachability_distribution/figure.pdf`
- `paper/figures/stage3_controllability_spectrum/figure.pdf`
- `paper/figures/stage3_cross_formulation_per_coordinate/figure.pdf`

Stage 4:

- `paper/figures/stage4_trust_region_violation_rate/figure.pdf`
- `paper/figures/stage4_realized_vs_predicted_displacement/figure.pdf`
- `paper/figures/stage4_trajectory_length_by_controller/figure.pdf`
- `paper/figures/stage4_coherence_rate_by_edit_norm/figure.pdf`

Stage 5:

- `paper/figures/stage5_identifiability_bars/figure.pdf`
- `paper/figures/stage5_per_coordinate_identifiability/figure.pdf`
- `paper/figures/stage5_cross_formulation_identifiability/figure.pdf`
- `paper/figures/stage5_confusion_matrix_headline/figure.pdf`
- `paper/figures/stage5_quality_conditioning_sensitivity/figure.pdf`

## Stages Still Needed

### Stage 4 Full Scale

Run the full trajectory-generation sweep with verifier/tool-interactive rollouts where required by the original plan. The completed Stage 4 v2 work is a reduced HF patched-decoding pilot/expansion, not the full environment-interactive run.

Needed additions:

- Larger prompt and seed counts.
- Full cell accounting through `runs/jobs.db`.
- Verifier reward/success fields populated where environments provide them.
- Matched temperature-as-skill rollouts.
- SAE-feature rollouts if SAE artifacts are available.

### Stage 5 Full Identifiability

Re-run after full Stage 4 data exists.

Needed additions:

- More held-out prompts and seeds for tighter bootstrap CIs.
- Actual temperature and SAE baselines.
- Quality/coherence conditioning based on verifier-scored or judge-scored trajectories, not only reduced completion heuristics.

### Stage 7

Not run.

Goal:

- Coverage, coherence, off-manifold diagnostics, and Pareto curves.

### Stage 8

Not run.

Goal:

- Cross-task Tier B replication.

### Stage 9

Not run.

Goal:

- Cross-model replication across the preregistered model list.

### Manuscript

Partially scaffolded, not yet NeurIPS-submittable.

Needed:

- Update `paper/main.tex` so every empirical claim references a `runs/<run_id>/summary.json` row through `runs/audit_table.md`.
- Make the headline claim match the audit table. Current strongest supported claim is diagnostic/conditional, not a positive Stage 5 skill-identifiability headline.
- Include only figures with valid `manifest.json` and source run IDs.

## Current Decision

The project should not scale Stage 7-9 or assert the original DIAYN-style headline until full Stage 4 and full Stage 5 have been rerun with richer, verifier-scored trajectory data and matched baselines. The reduced Stage 6 run is complete and reports a weak/null geometry-behavior correlation on the reduced data.

The present branch preserves a reproducible reduced-scale validation path and enough evidence to continue without node access, provided the local `runs/` artifact tree is retained or published to artifact storage.
