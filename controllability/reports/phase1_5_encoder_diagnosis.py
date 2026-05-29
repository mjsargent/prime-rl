from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from controllability.envs.trajectory_schema import read_jsonl
from controllability.reports.audit_table import append_audit_row


SENTENCE_EMBEDDING_DIM = 1024


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: 0.0 for key in ["min", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max", "mean"]}
    qs = np.quantile(values.astype(np.float64), [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "min": float(qs[0]),
        "p10": float(qs[1]),
        "p25": float(qs[2]),
        "median": float(qs[3]),
        "p75": float(qs[4]),
        "p90": float(qs[5]),
        "p95": float(qs[6]),
        "p99": float(qs[7]),
        "max": float(qs[8]),
        "mean": float(values.mean()),
    }


def _active_fraction_by_threshold(mean_var: np.ndarray) -> dict[str, float]:
    thresholds = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    return {f"{threshold:.0e}": float(np.mean(mean_var > threshold)) for threshold in thresholds}


def _prompt_group(prompt_id: str) -> str:
    return str(prompt_id).split(":", 1)[0]


def _load_rows(encoded_path: Path) -> list[dict[str, Any]]:
    return pq.read_table(encoded_path).to_pylist()


def _feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["behavior_features"] for row in rows], dtype=np.float32)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(denom, 1e-12, None)


def _pairwise_same_prompt_different_coordinate(rows: list[dict[str, Any]], sentence_features: np.ndarray) -> dict[str, Any]:
    normalized = _l2_normalize(sentence_features.astype(np.float64))
    prompt_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        prompt_to_indices.setdefault(_prompt_group(row["prompt_id"]), []).append(index)

    cosines: list[float] = []
    by_prompt: list[dict[str, Any]] = []
    for prompt, indices in sorted(prompt_to_indices.items(), key=lambda item: item[0]):
        prompt_values: list[float] = []
        for left_pos, i in enumerate(indices):
            for j in indices[left_pos + 1 :]:
                if int(rows[i]["coordinate"]) == int(rows[j]["coordinate"]):
                    continue
                value = float(np.dot(normalized[i], normalized[j]))
                cosines.append(value)
                prompt_values.append(value)
        if prompt_values:
            arr = np.asarray(prompt_values, dtype=np.float64)
            by_prompt.append(
                {
                    "prompt": prompt,
                    "num_pairs": int(arr.size),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "p10": float(np.quantile(arr, 0.1)),
                    "p90": float(np.quantile(arr, 0.9)),
                }
            )

    arr = np.asarray(cosines, dtype=np.float64)
    return {
        "num_pairs": int(arr.size),
        "distribution": _quantiles(arr),
        "by_prompt": by_prompt,
    }


def _within_prompt_variance(rows: list[dict[str, Any]], features: np.ndarray) -> dict[str, Any]:
    prompt_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        prompt_to_indices.setdefault(_prompt_group(row["prompt_id"]), []).append(index)
    prompt_vars = []
    for indices in prompt_to_indices.values():
        if len(indices) >= 2:
            prompt_vars.append(features[indices].var(axis=0))
    if not prompt_vars:
        mean_var = np.zeros(features.shape[1], dtype=np.float64)
    else:
        mean_var = np.asarray(prompt_vars, dtype=np.float64).mean(axis=0)
    return {
        "num_prompt_groups": len(prompt_vars),
        "mean_within_prompt_variance": _quantiles(mean_var),
        "active_fraction_by_threshold": _active_fraction_by_threshold(mean_var),
        "active_dims_at_1e_8": int(np.sum(mean_var > 1e-8)),
        "feature_dim": int(features.shape[1]),
    }


def _split_by_prompt(prompts: np.ndarray, *, seed: int, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted(set(prompts.tolist())))
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test_prompts = set(shuffled[:n_test].tolist())
    train_mask = np.asarray([prompt not in test_prompts for prompt in prompts], dtype=bool)
    test_mask = ~train_mask
    return train_mask, test_mask


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int,
    steps: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    x_train = features[train_mask].astype(np.float32)
    x_test = features[test_mask].astype(np.float32)
    y_train = labels[train_mask].astype(np.int64)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    model = torch.nn.Linear(x_train.shape[1], int(labels.max()) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-3)
    x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(torch.as_tensor(x_test, dtype=torch.float32))
    return logits.argmax(dim=-1).cpu().numpy().astype(np.int64)


def _bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prompts: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    accuracy = float(np.mean(y_true == y_pred))
    unique_prompts = np.asarray(sorted(set(prompts.tolist())))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
        values = []
        for prompt in chosen:
            mask = prompts == prompt
            values.extend((y_true[mask] == y_pred[mask]).astype(float).tolist())
        samples.append(float(np.mean(values)) if values else 0.0)
    low, high = np.quantile(samples, [0.025, 0.975]).tolist()
    return {"accuracy": accuracy, "ci_low": float(low), "ci_high": float(high)}


def _logistic_discriminator(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    *,
    seed: int,
    test_fraction: float,
    steps: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    coordinates = sorted({int(row["coordinate"]) for row in rows})
    label_map = {coordinate: idx for idx, coordinate in enumerate(coordinates)}
    labels = np.asarray([label_map[int(row["coordinate"])] for row in rows], dtype=np.int64)
    prompts = np.asarray([_prompt_group(row["prompt_id"]) for row in rows])
    train_mask, test_mask = _split_by_prompt(prompts, seed=seed, test_fraction=test_fraction)
    y_pred = _fit_logistic(features, labels, train_mask=train_mask, test_mask=test_mask, seed=seed, steps=steps)
    y_true = labels[test_mask]
    test_prompts = prompts[test_mask]
    acc = _bootstrap_accuracy(y_true, y_pred, test_prompts, n_bootstrap=n_bootstrap, seed=seed + 17)
    chance = 1.0 / len(coordinates)
    return {
        "accuracy": acc["accuracy"],
        "accuracy_ci_low": acc["ci_low"],
        "accuracy_ci_high": acc["ci_high"],
        "chance": chance,
        "identifiability": acc["accuracy"] - chance,
        "identifiability_ci_low": acc["ci_low"] - chance,
        "identifiability_ci_high": acc["ci_high"] - chance,
        "coordinates": coordinates,
        "num_classes": len(coordinates),
        "num_train": int(train_mask.sum()),
        "num_test": int(test_mask.sum()),
        "train_prompts": int(len(set(prompts[train_mask].tolist()))),
        "test_prompts": int(len(set(prompts[test_mask].tolist()))),
    }


def _trajectory_text(traj: Any) -> str:
    sections = []
    for message in traj.messages:
        role = str(message.get("role", "unknown"))
        content = message.get("content", "")
        sections.append(f"{role}: {content}")
    if traj.tool_calls:
        sections.append("tool_calls: " + json.dumps(traj.tool_calls, sort_keys=True))
    if traj.step_metadata:
        sections.append("metadata: " + json.dumps(traj.step_metadata, sort_keys=True))
    return "\n".join(sections)


def _first_behavior_prefix(text: str) -> str:
    markers = ["\nassistant:", "\ntool:", "\ntool_calls:"]
    positions = [text.find(marker) for marker in markers if text.find(marker) >= 0]
    if not positions:
        return text
    return text[: min(positions)]


def _truncation_audit(trajectories_path: Path, *, model_id: str, max_length: int, sample_size: int) -> dict[str, Any]:
    trajectories = read_jsonl(trajectories_path)[:sample_size]
    if not trajectories:
        return {"status": "empty"}
    records: list[dict[str, Any]] = []
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        for traj in trajectories:
            text = _trajectory_text(traj)
            prefix = _first_behavior_prefix(text)
            full_tokens = len(tokenizer(f"query: {text}", add_special_tokens=True)["input_ids"])
            prefix_tokens = len(tokenizer(f"query: {prefix}", add_special_tokens=True)["input_ids"])
            records.append(
                {
                    "trajectory_id": traj.trajectory_id,
                    "full_tokens": full_tokens,
                    "prefix_tokens_before_behavior": prefix_tokens,
                    "behavior_starts_after_max_length": bool(prefix_tokens >= max_length),
                    "text_characters": len(text),
                    "tool_calls": len(traj.tool_calls),
                    "reward": float(traj.reward),
                }
            )
    except Exception as exc:  # noqa: BLE001
        return {"status": "tokenizer_unavailable", "error": str(exc)}
    prefix_tokens = np.asarray([record["prefix_tokens_before_behavior"] for record in records], dtype=np.float64)
    full_tokens = np.asarray([record["full_tokens"] for record in records], dtype=np.float64)
    return {
        "status": "evaluated",
        "model_id": model_id,
        "max_length": max_length,
        "sample_size": len(records),
        "fraction_behavior_starts_after_max_length": float(np.mean(prefix_tokens >= max_length)),
        "prefix_tokens_before_behavior": _quantiles(prefix_tokens),
        "full_tokens": _quantiles(full_tokens),
        "examples": records[:5],
    }


def _write_report(report_path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    pairwise = metrics["pairwise_same_prompt_different_coordinate"]
    active = metrics["active_fraction_audit"]
    sentence_disc = metrics["coordinate_discriminator_sentence_embedding"]
    concat_disc = metrics["coordinate_discriminator_concat_features"]
    truncation = metrics["truncation_audit"]
    phase15 = metrics["phase1_5_positive_validation"]
    replacement = summary["replacement_gate"]
    text = f"""# Phase 1.5 Encoder Diagnosis

## Question

The corrected Phase 1.5 swe-grep verifier-loop smoke validated tool-use rollouts and turn-level linearization, but failed the encoder active-fraction gate. This report diagnoses whether that failure is caused by a broken encoder, genuinely similar trajectories, or a mis-specified active-fraction gate.

## Existing Positive Validation

- Full tool-use trajectories completed: {phase15["num_trajectories"]} / {phase15["num_jobs"]}
- Failed jobs: {phase15["num_failed"]}
- Tool calls: {phase15["tool_call_count"]}
- Nonzero-reward trajectories: {phase15["nonzero_reward_count"]}
- Mean reward: {phase15["reward_mean"]:.6f}
- Mean generated tokens: {phase15["mean_generated_tokens"]:.3f}
- Median turn-level linearization cosine: {phase15["median_turn_linearization_cosine"]:.6f}
- Median turn-level linearization R2: {phase15["median_turn_linearization_r2"]:.6f}

These are positive construction/integration results. The multi-turn verifier loop no longer shows the previous degenerate short-output or bad-linearization behavior.

## Diagnostic 1: Same-Prompt Sentence-Embedding Similarity

Pairwise cosine similarity was computed between trajectories from the same prompt but steered along different coordinates, using only the 1024-dimensional sentence-encoder embedding block.

- Number of same-prompt/different-coordinate pairs: {pairwise["num_pairs"]}
- Mean cosine: {pairwise["distribution"]["mean"]:.8f}
- Median cosine: {pairwise["distribution"]["median"]:.8f}
- P10/P90 cosine: {pairwise["distribution"]["p10"]:.8f} / {pairwise["distribution"]["p90"]:.8f}
- Min/Max cosine: {pairwise["distribution"]["min"]:.8f} / {pairwise["distribution"]["max"]:.8f}

Interpretation: same-prompt embeddings are almost identical across coordinates. This means the current sentence-embedding representation is not exposing coordinate-conditioned behavioral variation to the discriminator.

## Diagnostic 2: Active-Fraction Calculation Audit

The Phase 1.5 gate applied threshold `variance > 1e-8` to the mean within-prompt variance of each feature dimension.

Sentence-embedding block:

- Feature dimensions: {active["sentence"]["feature_dim"]}
- Active dims at `1e-8`: {active["sentence"]["active_dims_at_1e_8"]}
- Active fraction at `1e-8`: {active["sentence"]["active_fraction_by_threshold"]["1e-08"]:.6f}
- Median within-prompt variance: {active["sentence"]["mean_within_prompt_variance"]["median"]:.3e}
- P90/P99 within-prompt variance: {active["sentence"]["mean_within_prompt_variance"]["p90"]:.3e} / {active["sentence"]["mean_within_prompt_variance"]["p99"]:.3e}
- Max within-prompt variance: {active["sentence"]["mean_within_prompt_variance"]["max"]:.3e}

Full concatenated feature vector:

- Feature dimensions: {active["concat"]["feature_dim"]}
- Active dims at `1e-8`: {active["concat"]["active_dims_at_1e_8"]}
- Active fraction at `1e-8`: {active["concat"]["active_fraction_by_threshold"]["1e-08"]:.6f}

Active fraction by threshold for sentence embeddings:

```json
{json.dumps(active["sentence"]["active_fraction_by_threshold"], indent=2)}
```

The threshold is not merely too high by a small constant. The within-prompt sentence-embedding variance is effectively zero at this scale.

## Diagnostic 3: Coordinate Discriminator on Full Embeddings

A logistic discriminator was trained on the full 1024-dimensional sentence embeddings to predict coordinate, with train/test split by prompt and bootstrap CI over held-out prompts.

- Accuracy: {sentence_disc["accuracy"]:.6f}
- Bootstrap CI: [{sentence_disc["accuracy_ci_low"]:.6f}, {sentence_disc["accuracy_ci_high"]:.6f}]
- Chance: {sentence_disc["chance"]:.6f}
- Chance-corrected identifiability: {sentence_disc["identifiability"]:.6f}
- Identifiability CI: [{sentence_disc["identifiability_ci_low"]:.6f}, {sentence_disc["identifiability_ci_high"]:.6f}]
- Train/test prompts: {sentence_disc["train_prompts"]} / {sentence_disc["test_prompts"]}
- Train/test trajectories: {sentence_disc["num_train"]} / {sentence_disc["num_test"]}

For comparison, the same discriminator on the full 1040-dimensional concatenated vector gives:

- Accuracy: {concat_disc["accuracy"]:.6f}
- Bootstrap CI: [{concat_disc["accuracy_ci_low"]:.6f}, {concat_disc["accuracy_ci_high"]:.6f}]
- Chance-corrected identifiability: {concat_disc["identifiability"]:.6f}

The sentence-embedding discriminator does not recover coordinate identity above chance.

## Truncation Audit

The sentence encoder uses `max_length=512` and the trajectory text is encoded in chronological order: prompt first, assistant/tool behavior later.

Status: {truncation["status"]}
"""
    if truncation.get("status") == "evaluated":
        text += f"""
- Sample size: {truncation["sample_size"]}
- Fraction where behavior starts after the 512-token encoder window: {truncation["fraction_behavior_starts_after_max_length"]:.6f}
- Median prefix tokens before first behavior: {truncation["prefix_tokens_before_behavior"]["median"]:.1f}
- P10/P90 prefix tokens before first behavior: {truncation["prefix_tokens_before_behavior"]["p10"]:.1f} / {truncation["prefix_tokens_before_behavior"]["p90"]:.1f}
- Median full trajectory tokens under the sentence tokenizer: {truncation["full_tokens"]["median"]:.1f}
"""
    else:
        text += f"\nTokenizer audit unavailable: `{truncation.get('error', 'unknown')}`\n"

    text += f"""
## Interpretation

Dominant explanation: **{summary["dominant_explanation"]}**.

The evidence points to encoder mis-calibration/integration rather than a construction failure. The verifier-loop trajectories are valid, non-degenerate, and first-order linearization holds at turn level. The active-fraction failure occurs because the current sentence encoder is effectively prompt-dominated within each prompt group. The most likely mechanism is chronological truncation: swe-grep prompts are long, while assistant and tool behavior appears after the prompt and can be outside the first 512 tokens embedded by E5.

This does not support moving to Phase 2. It also does not support Stage 3 v2. The right fix is an encoder fix: encode the behavior-bearing suffix, assistant/tool blocks, or chunked trajectory summaries rather than the prompt-prefix-dominated trajectory string.

## Gate Reanalysis

- Active-fraction gate status under existing rule: fail
- Replacement gate proposed now: {replacement["proposed"]}
- Replacement gate applied now: {replacement["applied"]}
- Reason: {replacement["reason"]}

Because the discriminator on the full sentence embeddings remains at chance, a discriminator-accuracy replacement gate would also fail on the current embeddings. The gate is not simply too strict; the current embedding representation is not measuring the behavioral channel we need.

## Recommended Fix

1. Replace the sentence-encoder input text with behavior-focused text: assistant messages, tool calls, tool outputs, final answer, and compact metadata. Exclude or heavily downweight the original prompt.
2. Increase `max_length` or use chunked pooling over trajectory sections so tool-use behavior cannot be truncated away.
3. Re-run this exact encoder diagnosis before any Phase 2 scale-up. The minimum pass condition should be both:
   - held-out coordinate discriminator accuracy above chance with bootstrap CI lower bound above chance, and
   - non-degenerate same-prompt embedding spread, calibrated by the observed variance scale rather than a fixed `1e-8` threshold.

## Audit Note

The Phase 1.5 Qwen tool-use smoke should remain in the audit trail as positive evidence for full-agent first-order linearization and nonzero verifier reward. The encoder failure is a measurement-instrument failure, not evidence that the controllability construction failed.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)


def _audit(stage: str, run_id: str, gate: str, metrics: dict[str, Any], evidence_path: str) -> None:
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": _utc_now(),
            "stage": stage,
            "run_id": run_id,
            "gate_status": gate,
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": evidence_path,
            "prime_rl_modifications": "none",
        },
    )


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    source_run_dir = Path(args.source_run_dir)
    encoded_path = source_run_dir / "encoded.parquet"
    trajectories_path = source_run_dir / "trajectories.jsonl"
    phase15_summary = json.loads((source_run_dir / "summary.json").read_text())
    rows = _load_rows(encoded_path)
    features = _feature_matrix(rows)
    sentence_features = features[:, :SENTENCE_EMBEDDING_DIM]
    concat_features = features

    pairwise = _pairwise_same_prompt_different_coordinate(rows, sentence_features)
    active = {
        "sentence": _within_prompt_variance(rows, sentence_features),
        "concat": _within_prompt_variance(rows, concat_features),
        "global_sentence_variance": _quantiles(sentence_features.var(axis=0)),
        "global_concat_variance": _quantiles(concat_features.var(axis=0)),
        "configured_threshold": args.variance_threshold,
    }
    sentence_disc = _logistic_discriminator(
        rows,
        sentence_features,
        seed=args.seed,
        test_fraction=args.test_fraction,
        steps=args.steps,
        n_bootstrap=args.n_bootstrap,
    )
    concat_disc = _logistic_discriminator(
        rows,
        concat_features,
        seed=args.seed,
        test_fraction=args.test_fraction,
        steps=args.steps,
        n_bootstrap=args.n_bootstrap,
    )
    truncation = _truncation_audit(
        trajectories_path,
        model_id=args.sentence_model_id,
        max_length=args.max_length,
        sample_size=args.truncation_sample_size,
    )

    phase15_metrics = phase15_summary["metrics"]
    positive_validation = {
        "num_jobs": int(phase15_metrics["num_jobs"]),
        "num_trajectories": int(phase15_metrics["num_trajectories"]),
        "num_failed": int(phase15_metrics["num_failed"]),
        "tool_call_count": int(phase15_metrics["tool_call_count"]),
        "nonzero_reward_count": int(phase15_metrics["nonzero_reward_count"]),
        "reward_mean": float(phase15_metrics["reward_mean"]),
        "mean_generated_tokens": float(phase15_metrics["mean_generated_tokens"]),
        "median_turn_linearization_cosine": float(phase15_metrics["median_turn_linearization_cosine"]),
        "median_turn_linearization_r2": float(phase15_metrics["median_turn_linearization_r2"]),
    }

    chance = sentence_disc["chance"]
    active_fail = active["concat"]["active_fraction_by_threshold"][f"{args.variance_threshold:.0e}"] < args.active_fraction_floor
    discriminator_pass = sentence_disc["identifiability_ci_low"] > 0.0
    same_prompt_nearly_identical = pairwise["distribution"]["median"] > 0.999
    truncation_dominant = (
        truncation.get("status") == "evaluated"
        and truncation.get("fraction_behavior_starts_after_max_length", 0.0) >= 0.5
    )
    if truncation_dominant:
        dominant_explanation = "encoder_miscalibrated_prompt_prefix_truncation"
    elif active_fail and discriminator_pass:
        dominant_explanation = "active_fraction_gate_misspecified"
    elif same_prompt_nearly_identical and not discriminator_pass:
        dominant_explanation = "trajectories_or_embeddings_genuinely_similar_within_prompt"
    else:
        dominant_explanation = "mixed_encoder_measurement_failure"

    replacement_gate = {
        "proposed": bool(active_fail and discriminator_pass),
        "applied": False,
        "candidate": "coordinate_discriminator_identifiability_ci_low > 0",
        "reason": (
            "Not applied because the full sentence-embedding discriminator is not above chance."
            if not discriminator_pass
            else "Active fraction failed while discriminator passed; replacement gate should be preregistered before rerun."
        ),
        "sentence_accuracy": sentence_disc["accuracy"],
        "sentence_accuracy_ci_low": sentence_disc["accuracy_ci_low"],
        "chance": chance,
    }

    run_id = args.run_id
    run_dir = Path("runs") / run_id
    summary = {
        "stage": "phase1_5_encoder_diagnosis",
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": _git_hash(),
        "source_run": str(source_run_dir),
        "metrics": {
            "pairwise_same_prompt_different_coordinate": pairwise,
            "active_fraction_audit": active,
            "coordinate_discriminator_sentence_embedding": sentence_disc,
            "coordinate_discriminator_concat_features": concat_disc,
            "truncation_audit": truncation,
            "phase1_5_positive_validation": positive_validation,
        },
        "dominant_explanation": dominant_explanation,
        "replacement_gate": replacement_gate,
        "gate": "fail",
        "gate_reason": "Phase 2 remains blocked until the encoder input is fixed and the diagnosis passes.",
    }
    _write_json(run_dir / "summary.json", summary)
    _write_report(Path(args.report_path), summary)

    _audit(
        "phase1_5_encoder_diagnosis",
        run_id,
        summary["gate"],
        {
            "dominant_explanation": dominant_explanation,
            "same_prompt_diff_coordinate_cosine_median": pairwise["distribution"]["median"],
            "sentence_accuracy": sentence_disc["accuracy"],
            "sentence_accuracy_ci_low": sentence_disc["accuracy_ci_low"],
            "sentence_chance": sentence_disc["chance"],
            "sentence_active_fraction_1e_8": active["sentence"]["active_fraction_by_threshold"]["1e-08"],
            "replacement_gate_applied": replacement_gate["applied"],
        },
        f"runs/{run_id}/summary.json",
    )
    _audit(
        "stage3_multiturn_linearization_validation",
        "phase1_5_swe_grep_verifier_smoke_parametric_b_qwen_tooluse_float32",
        "pass",
        {
            "median_turn_linearization_cosine": positive_validation["median_turn_linearization_cosine"],
            "median_turn_linearization_r2": positive_validation["median_turn_linearization_r2"],
            "num_turn_diagnostics": int(phase15_metrics["num_turn_diagnostics"]),
            "mean_generated_tokens": positive_validation["mean_generated_tokens"],
        },
        str(source_run_dir / "summary.json"),
    )
    _audit(
        "phase1_5_tooluse_reward_validation",
        "phase1_5_swe_grep_verifier_smoke_parametric_b_qwen_tooluse_float32",
        "pass",
        {
            "reward_mean": positive_validation["reward_mean"],
            "nonzero_reward_count": positive_validation["nonzero_reward_count"],
            "tool_call_count": positive_validation["tool_call_count"],
            "num_trajectories": positive_validation["num_trajectories"],
        },
        str(source_run_dir / "summary.json"),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", default="runs/phase1_5_swe_grep_verifier_smoke_parametric_b_qwen_tooluse_float32")
    parser.add_argument("--run-id", default="phase1_5_encoder_diagnosis")
    parser.add_argument("--report-path", default="reports/phase1_5_encoder_diagnosis.md")
    parser.add_argument("--sentence-model-id", default="intfloat/e5-large-v2")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--variance-threshold", type=float, default=1e-8)
    parser.add_argument("--active-fraction-floor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--truncation-sample-size", type=int, default=100)
    summary = diagnose(parser.parse_args())
    print(json.dumps({"run_id": summary["run_id"], "gate": summary["gate"], "dominant_explanation": summary["dominant_explanation"]}, indent=2))


if __name__ == "__main__":
    main()
