# Phase 1.5 Encoder V3 Diagnosis

## Status

`behavioral_encoder_v3` replaces the sentence encoder for future Phase 1.5/Phase 2 work. The replacement was logged as a `PREREG_CHANGE_encoder_replacement` audit-table row with this rationale:

> sentence encoder was trained for semantic-similarity invariance, which actively works against measuring the fine-grained behavioral variation the construction produces. Task-aware features measure what we actually want.

## Encoder Design

For `prime/swe-grep`, v3 is a 99-dimensional task-behavior feature vector:

- tool-call counts by type: `grep`, `glob_files`, `read_file`, `list_dir`, `unknown`
- ordered first-8 tool sequence, one-hot by tool type
- file/directory clusters touched from tool args and tool outputs
- hashed query-pattern features from grep/glob/path arguments
- retrieval depth, turn count, number of tool outputs, generated-token counts, reward, success, and wall-clock features

For `primeintellect/math500`, v3 is a 62-dimensional task-behavior feature vector:

- reasoning-marker counts: `let`, `therefore`, `because`, `suppose`, `case`, `hence`, `thus`, `so`
- solution length, line count, punctuation, digits, and math-operator density
- intermediate-equation and LaTeX structure counts
- hashed canonical final-answer feature
- casework indicators
- method-category indicators for calculus, integration, algebra, trig, geometry, probability, number theory, and inequalities
- hashed lexical/style features from the assistant solution
- reward, success, turn count, and final-answer-present features

The encoder is deterministic and fixed-dimensional; it does not train or fit on labels.

## Existing-Trajectory Validation

Before launching a full new smoke, v3 was tested against existing validated trajectories.

For the full corrected swe-grep verifier-loop Phase 1.5 data:

- trajectories: 2160
- feature dimension: 99
- active dimensions at `variance > 1e-8`: 84
- active fraction: 0.848485
- median mean-within-prompt variance: 0.004372
- P90 mean-within-prompt variance: 0.073971

For the available math500 single-turn diagnostic sample:

- trajectories: 50
- feature dimension: 62
- active dimensions at `variance > 1e-8`: 36
- active fraction: 0.580645
- median mean-within-prompt variance: 0.000152
- P90 mean-within-prompt variance: 0.004108

Both clear the Phase 1.5 active-fraction floor of 0.5 on the available validation data.

## Encoder-Only Phase 1.5 Recovery

I ran an encoder-only recovery using the already validated Qwen tool-use swe-grep trajectories. This did not generate new rollouts; it re-encoded the existing trajectories and re-ran the Phase 1.5 gate logic.

Run:

`runs/phase1_5_swe_grep_verifier_smoke_parametric_b_behavioral_v3_qwen_tooluse_float32_recovered/summary.json`

Result:

- gate: pass
- trajectories: 2160 / 2160
- failed jobs: 0
- tool calls: 18432
- reward mean: 0.890701
- nonzero-reward trajectories: 956
- mean generated tokens: 342.426
- median turn-level linearization cosine: 0.999913
- median turn-level linearization R2: 0.999816
- encoder active fraction: 0.848485
- completeness gate self-test: pass

This confirms the previous Phase 1.5 failure was specific to the sentence-encoder measurement instrument, not the verifier loop, task performance, linearization, or completeness checks.

## Discriminator Note

A prompt-split logistic discriminator on the v3 features from the existing swe-grep trajectories was still at chance:

- accuracy: 0.166667
- chance: 0.166667
- chance-corrected identifiability: 0.0

This is not part of the Phase 1.5 active-fraction gate, but it is important for Stage 5 interpretation. The v3 encoder now exposes task-behavior variation, but the existing reduced-scale data still does not show coordinate-identifiable behavior under a prompt-held-out discriminator. Phase 2 remains the power test for that question.

## Full Slurm Rerun

Submitted a full one-node Slurm rerun with `behavioral_encoder_v3`:

- job id: 12281
- script: `runs/slurm/phase1_5_smoke_behavioral_v3_qwen_tooluse_float32.sbatch`
- first run: `phase1_5_swe_grep_verifier_smoke_parametric_b_behavioral_v3_qwen_tooluse_float32`
- second run, only if swe-grep passes: `phase1_5_math500_behavioral_v3_smoke_parametric_b_qwen_tooluse_float32`

The job uses one node and one GPU. The encoder-only recovery already passes on swe-grep, but the full rerun remains useful as a clean end-to-end audit row under the new preregistered encoder.

## Recommendation

Do not rerun Stage 3 v2. The construction-validation result stands: full-agent turn-level linearization remains excellent and verifier reward is nonzero.

If the full v3 Phase 1.5 Slurm run passes, proceed to Phase 2 using `behavioral_encoder_v3`. If the full run fails on the encoder gate despite the recovery result, write a second diagnosis focused on differences between recovered and regenerated trajectories. If Stage 5 remains at chance after Phase 2, then the urgent framing question becomes whether DIAYN-style coordinate classification is the right operationalization of identifiability, and continuous-distance alternatives should be considered.
