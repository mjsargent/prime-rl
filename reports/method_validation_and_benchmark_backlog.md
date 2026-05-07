# Method Validation and Benchmark Backlog

Date: 2026-05-07

Status: logged after the Tier A Stage 1 stop. This file does not reopen Stage 1, does not change the preregistered gate, and does not authorize Stage 2. It records follow-up work needed before treating the low natural eigengap results as a final verdict on the method, and it records alternative methods as later benchmark candidates.

## Current Method Validation Checks

These checks test whether the observed low natural `K*` is a property of the model/task distribution or an artifact of the implementation and construction choices.

1. Chart dimension sweep.
   Re-run eigengap diagnostics with chart dimensions `d in {32, 64, 128, 256}`. If `K*` scales with `d`, the chart is limiting the construction. If `K*` saturates, the low-dimensional result is more likely intrinsic to the rollout geometry.

2. Chart collapse diagnostic.
   Record the PCA explained-variance and stable-rank curves at each readout layer. Compare PCA, transition-PCA, contrastive, and flow charts to determine whether absolute residual variance is hiding controllability-relevant directions.

3. Residual metric sweep.
   Compare `C_l = I`, `C_l = Cov(h_l)`, and `C_l = Cov(delta h_l)`. Low rank in `Cov(delta h_l)` can constrain `M_tau = J_tau C_l J_tau^T` even when `J_tau` itself is richer.

4. Regularization sweep.
   Sweep `lambda` relative to the leading scale of `M_tau`, for example `{1e-5, 1e-3, 1e-1} * sigma_1(M_tau)`. The natural eigengap should not be highly sensitive to this choice.

5. Graph bandwidth and graph type sweep.
   Compare median-bandwidth kernels against scaled bandwidths `{0.5, 1, 2, 4} * median` and a k-NN graph. This checks whether the graph is averaging away within-cluster writable structure.

6. Patch-position diagnostic.
   Compare last-token patching with patching at observation-boundary tokens, response-start tokens, and multiple positions. This is especially relevant for multi-turn agentic environments.

7. LayerNorm and Jacobian consistency checks.
   Extend the finite-difference VJP tests with across-state diagnostics: compare singular-value ratios for states with different residual norms and ensure LayerNorm scaling is not creating inconsistent aggregation.

8. Mean-vs-local controllability diagnostic.
   Compare eigenvalues of typical `M_tau` against eigenvalues of `E_tau[M_tau]`. If typical states have rank 5-10 but the mean has rank 1-3, the issue is state-dependent rotation of writable directions rather than lack of local controllability.

9. Rollout-distribution heterogeneity.
   Run construction diagnostics on mixed rollout distributions, for example temperatures `{0.3, 0.7, 1.0, 1.3}` and multiple system-prompt variants. The construction can only recover variation present in the sampled rollout manifold.

10. More open-ended environments.
    Before treating the method as globally low-dimensional, run the same validation on an open-ended generation environment. `math500` and `swe-grep` are both narrow, verifiable tasks with constrained solution spaces.

## Alternative Methods to Benchmark Later

These are not replacements for the current method until the checks above are complete. They should be treated as benchmark or ablation families in a later preregistered iteration.

1. Koopman or DMD-style operator methods.
   Estimate spectral structure of the suffix dynamics rather than a static global graph. This may recover dynamic structure when local controllability directions rotate across states.

2. Matrix dictionary learning on suffix Jacobians.
   Treat `{J_tau}` or `{M_tau}` as a dataset of matrices and learn state-conditional controllability atoms. This directly targets locally rich but globally inconsistent controllability.

3. Contrastive controllability tensors.
   Use behaviorally distinct state pairs and decompose tensors such as `E[(J_a - J_b)(J_a - J_b)^T]`. This benchmarks whether state-conditional differences are stronger than global common directions.

4. Per-prompt local control.
   Fit a local controllability basis per prompt or prompt cluster instead of requiring one global basis across the task distribution. This is a smaller claim than global controllability spectroscopy but preserves the suffix-Jacobian actuator.

## Audit Interpretation

The Tier A result remains a Stage 1 stop under the current preregistered protocol. The items above are logged as future validation and benchmarking work, not as post-hoc changes to the stopped run.
