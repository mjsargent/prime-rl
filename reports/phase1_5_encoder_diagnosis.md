# Phase 1.5 Encoder Diagnosis

## Question

The corrected Phase 1.5 swe-grep verifier-loop smoke validated tool-use rollouts and turn-level linearization, but failed the encoder active-fraction gate. This report diagnoses whether that failure is caused by a broken encoder, genuinely similar trajectories, or a mis-specified active-fraction gate.

## Existing Positive Validation

- Full tool-use trajectories completed: 2160 / 2160
- Failed jobs: 0
- Tool calls: 18432
- Nonzero-reward trajectories: 956
- Mean reward: 0.890701
- Mean generated tokens: 342.426
- Median turn-level linearization cosine: 0.999913
- Median turn-level linearization R2: 0.999816

These are positive construction/integration results. The multi-turn verifier loop no longer shows the previous degenerate short-output or bad-linearization behavior.

## Diagnostic 1: Same-Prompt Sentence-Embedding Similarity

Pairwise cosine similarity was computed between trajectories from the same prompt but steered along different coordinates, using only the 1024-dimensional sentence-encoder embedding block.

- Number of same-prompt/different-coordinate pairs: 97200
- Mean cosine: 1.00000000
- Median cosine: 1.00000000
- P10/P90 cosine: 1.00000000 / 1.00000000
- Min/Max cosine: 1.00000000 / 1.00000000

Interpretation: same-prompt embeddings are almost identical across coordinates. This means the current sentence-embedding representation is not exposing coordinate-conditioned behavioral variation to the discriminator.

## Diagnostic 2: Active-Fraction Calculation Audit

The Phase 1.5 gate applied threshold `variance > 1e-8` to the mean within-prompt variance of each feature dimension.

Sentence-embedding block:

- Feature dimensions: 1024
- Active dims at `1e-8`: 0
- Active fraction at `1e-8`: 0.000000
- Median within-prompt variance: 3.500e-16
- P90/P99 within-prompt variance: 1.517e-15 / 3.716e-15
- Max within-prompt variance: 9.501e-15

Full concatenated feature vector:

- Feature dimensions: 1040
- Active dims at `1e-8`: 8
- Active fraction at `1e-8`: 0.007692

Active fraction by threshold for sentence embeddings:

```json
{
  "1e-12": 0.0,
  "1e-11": 0.0,
  "1e-10": 0.0,
  "1e-09": 0.0,
  "1e-08": 0.0,
  "1e-07": 0.0,
  "1e-06": 0.0
}
```

The threshold is not merely too high by a small constant. The within-prompt sentence-embedding variance is effectively zero at this scale.

## Diagnostic 3: Coordinate Discriminator on Full Embeddings

A logistic discriminator was trained on the full 1024-dimensional sentence embeddings to predict coordinate, with train/test split by prompt and bootstrap CI over held-out prompts.

- Accuracy: 0.166667
- Bootstrap CI: [0.166667, 0.166667]
- Chance: 0.166667
- Chance-corrected identifiability: 0.000000
- Identifiability CI: [0.000000, 0.000000]
- Train/test prompts: 15 / 5
- Train/test trajectories: 1620 / 540

For comparison, the same discriminator on the full 1040-dimensional concatenated vector gives:

- Accuracy: 0.166667
- Bootstrap CI: [0.166667, 0.166667]
- Chance-corrected identifiability: 0.000000

The sentence-embedding discriminator does not recover coordinate identity above chance.

## Truncation Audit

The sentence encoder uses `max_length=512` and the trajectory text is encoded in chronological order: prompt first, assistant/tool behavior later.

Status: evaluated

- Sample size: 100
- Fraction where behavior starts after the 512-token encoder window: 1.000000
- Median prefix tokens before first behavior: 590.0
- P10/P90 prefix tokens before first behavior: 590.0 / 590.0
- Median full trajectory tokens under the sentence tokenizer: 3825.0

## Interpretation

Dominant explanation: **encoder_miscalibrated_prompt_prefix_truncation**.

The evidence points to encoder mis-calibration/integration rather than a construction failure. The verifier-loop trajectories are valid, non-degenerate, and first-order linearization holds at turn level. The active-fraction failure occurs because the current sentence encoder is effectively prompt-dominated within each prompt group. The most likely mechanism is chronological truncation: swe-grep prompts are long, while assistant and tool behavior appears after the prompt and can be outside the first 512 tokens embedded by E5.

This does not support moving to Phase 2. It also does not support Stage 3 v2. The right fix is an encoder fix: encode the behavior-bearing suffix, assistant/tool blocks, or chunked trajectory summaries rather than the prompt-prefix-dominated trajectory string.

## Gate Reanalysis

- Active-fraction gate status under existing rule: fail
- Replacement gate proposed now: False
- Replacement gate applied now: False
- Reason: Not applied because the full sentence-embedding discriminator is not above chance.

Because the discriminator on the full sentence embeddings remains at chance, a discriminator-accuracy replacement gate would also fail on the current embeddings. The gate is not simply too strict; the current embedding representation is not measuring the behavioral channel we need.

## Recommended Fix

1. Replace the sentence-encoder input text with behavior-focused text: assistant messages, tool calls, tool outputs, final answer, and compact metadata. Exclude or heavily downweight the original prompt.
2. Increase `max_length` or use chunked pooling over trajectory sections so tool-use behavior cannot be truncated away.
3. Re-run this exact encoder diagnosis before any Phase 2 scale-up. The minimum pass condition should be both:
   - held-out coordinate discriminator accuracy above chance with bootstrap CI lower bound above chance, and
   - non-degenerate same-prompt embedding spread, calibrated by the observed variance scale rather than a fixed `1e-8` threshold.

## Audit Note

The Phase 1.5 Qwen tool-use smoke should remain in the audit trail as positive evidence for full-agent first-order linearization and nonzero verifier reward. The encoder failure is a measurement-instrument failure, not evidence that the controllability construction failed.
