# Phase 1 Free Ablations Summary

Generated: 2026-05-12T05:20:48Z

## Scope

Phase 1 used only existing reduced Stage 4 v2 encoded trajectories. No new rollouts, model loading, or verifier calls were run.

## Results

- Phase 1.1 per-controller: evaluated only for closed_loop; one_shot/open_loop absent. Reduced data contains only `closed_loop` steered trajectories, so one-shot and open-loop comparisons are not estimable.
- Phase 1.2 sign-as-class: K-class coordinate discriminator evaluated where possible; 2K signed discriminator absent. Reduced data contains only `+` steered signs, so 2K signed discrimination is not estimable.
- Phase 1.3 per-coordinate audit: gate `pass`. top two coordinates account for more than 70% of positive identifiability signal in every evaluated cell
- Phase 1.4 top-coordinate geometry: gate `inconclusive`. top identifiable coordinates do not show a clear geometric pattern at reduced scale
- Phase 1.5 quality-conditioned identifiability: gate `inconclusive`. quality conditioning cannot be evaluated on reduced Stage 4 data because quality is unset or filters remove required classes
- Phase 1.6 encoder audit: gate `fail`. encoder has too few feature dimensions varying across coordinates
- Phase 1.7 graph/PCA proxy collision: gate `pass`. graph condition documented as dropped for downstream Phase 2 rather than treated as an independent method

## Recommendation For Phase 2

Do not start Phase 2 at full scope. Phase 1.6 failed: too few behavioral encoder feature dimensions vary across coordinates, so the reduced Stage 5/6 null may be an encoder-power failure rather than a geometry failure. Fix or replace the encoder first, then rerun Phase 1.6 and the cheap discriminator ablations before scaling. When scaling resumes, prioritize B + A, include both signs, include controller variants if available, and drop the graph condition unless a true Nyström extension is implemented.

## Evidence

- `phase1_per_coordinate_audit`: `runs/phase1_per_coordinate_audit/summary.json`
- `phase1_top_coordinate_pattern`: `runs/phase1_top_coordinate_pattern/summary.json`
- `phase1_quality_conditioned`: `runs/phase1_quality_conditioned/summary.json`
- `phase1_encoder_audit`: `runs/phase1_encoder_audit/summary.json`
- `phase1_graph_proxy_collision`: `runs/phase1_graph_proxy_collision/summary.json`
