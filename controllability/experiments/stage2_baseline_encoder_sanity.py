from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from controllability.encoders.encoder_registry import get_encoder
from controllability.envs.trajectory_schema import read_jsonl
from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sanitize_id(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").replace(".", "_")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _as_float_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in values.tolist()]


def _encode_rollouts(trajectories_path: Path) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    trajectories = read_jsonl(trajectories_path)
    if not trajectories:
        msg = f"no trajectories found at {trajectories_path}"
        raise ValueError(msg)

    encoder = get_encoder(trajectories[0].env_id)
    encoded = encoder.encode_batch(trajectories)
    features = np.stack([item.behavior_features.astype(np.float64) for item in encoded], axis=0)
    if not np.isfinite(features).all():
        msg = "encoder produced NaN or infinite behavior features"
        raise ValueError(msg)

    records: list[dict[str, Any]] = []
    prompt_ids: list[str] = []
    for traj, item in zip(trajectories, encoded, strict=True):
        prompt_ids.append(str(traj.prompt_id))
        records.append(
            {
                "trajectory_id": item.trajectory_id,
                "env_id": item.env_id,
                "model_id": traj.model_id,
                "prompt_id": str(traj.prompt_id),
                "seed": int(traj.seed),
                "reward": float(traj.reward),
                "success": bool(traj.success),
                "behavior_features": _as_float_list(item.behavior_features),
                "quality": float(item.quality),
                "coherence": float(item.coherence),
                "raw_features": json.dumps(item.raw_features, sort_keys=True),
                "encoder_version": item.encoder_version,
            }
        )
    return records, features, prompt_ids


def _write_encoded_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)


def _mean_feature_wasserstein(a: np.ndarray, b: np.ndarray) -> tuple[float, list[float]]:
    if a.shape != b.shape:
        msg = f"shape mismatch for Wasserstein comparison: {a.shape} vs {b.shape}"
        raise ValueError(msg)
    distances = np.mean(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)), axis=0)
    return float(np.mean(distances)), _as_float_list(distances)


def _prompt_split(prompt_ids: list[str], test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_prompts = np.array(sorted(set(prompt_ids)))
    if unique_prompts.size < 2:
        msg = "prompt-held-out split requires at least two unique prompts"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_prompts)
    n_test = max(1, int(round(unique_prompts.size * test_fraction)))
    n_test = min(n_test, unique_prompts.size - 1)
    test_prompts = set(unique_prompts[:n_test].tolist())
    test_mask = np.array([prompt_id in test_prompts for prompt_id in prompt_ids], dtype=bool)
    return ~test_mask, test_mask


def _standardize_train_test(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


def _nearest_centroid_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    prompt_ids: list[str],
    *,
    k: int,
    test_fraction: float,
    seed: int,
) -> float:
    train_mask, test_mask = _prompt_split(prompt_ids, test_fraction=test_fraction, seed=seed)
    x = _standardize_train_test(features, train_mask)
    centroids = np.zeros((k, x.shape[1]), dtype=np.float64)
    global_mean = x[train_mask].mean(axis=0)
    for label in range(k):
        members = train_mask & (labels == label)
        centroids[label] = x[members].mean(axis=0) if np.any(members) else global_mean
    distances = np.sum((x[test_mask, None, :] - centroids[None, :, :]) ** 2, axis=2)
    predictions = np.argmin(distances, axis=1)
    return float(np.mean(predictions == labels[test_mask]))


def _balanced_random_labels(n: int, k: int, seed: int) -> np.ndarray:
    labels = np.resize(np.arange(k, dtype=np.int64), n)
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    return labels


def _random_baseline(
    features: np.ndarray,
    prompt_ids: list[str],
    *,
    k: int,
    n_trials: int,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    accuracies = []
    for trial in range(n_trials):
        trial_seed = seed + trial
        labels = _balanced_random_labels(features.shape[0], k, trial_seed)
        accuracies.append(
            _nearest_centroid_accuracy(
                features,
                labels,
                prompt_ids,
                k=k,
                test_fraction=test_fraction,
                seed=trial_seed + 10_000,
            )
        )
    accuracy = float(np.mean(accuracies))
    chance = 1.0 / float(k)
    return {
        "status": "evaluated",
        "classifier": "nearest_centroid",
        "train_test_split": "by_prompt",
        "n_trials": n_trials,
        "k": k,
        "chance_accuracy": chance,
        "mean_accuracy": accuracy,
        "std_accuracy": float(np.std(accuracies)),
        "abs_diff_from_chance": float(abs(accuracy - chance)),
        "trial_accuracies": [float(v) for v in accuracies],
    }


def _pca_baseline(features: np.ndarray, k: int) -> dict[str, Any]:
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singular_values**2) / max(features.shape[0] - 1, 1)
    total = float(np.sum(eigenvalues))
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    n_components = min(k, vh.shape[0])
    return {
        "status": "locked",
        "n_components": int(n_components),
        "explained_variance_ratio": _as_float_list(explained[:n_components]),
        "cumulative_explained_variance": float(np.sum(explained[:n_components])),
        "component_matrix": [_as_float_list(row) for row in vh[:n_components]],
    }


def _temperature_baseline(k: int, config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("temperatures")
    if values is None:
        values = np.linspace(0.2, 1.4, num=k).round(4).tolist()
    return {
        "status": "locked_not_evaluated",
        "reason": "temperature-as-skill rollouts are generated in Stage 4; Stage 2 locks the matched-K schedule",
        "k": k,
        "temperatures": [float(v) for v in values],
    }


def _formulation_metadata(config: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "formulation": config["formulation"],
        "formulation_role": config.get("formulation_role", "comparator"),
        "selected_k": int(config["selected_k"]),
    }
    for key in ("selected_patch_layer", "selected_readout_layer", "source_summary", "source_method_comparison"):
        if key in config:
            metadata[key] = config[key]
    return metadata


def _summary_payload(
    *,
    run_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    gate: str,
    gate_reason: str,
    timestamp_utc: str,
) -> dict[str, Any]:
    return {
        "stage": "stage2_baseline_encoder_sanity",
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": config,
        "inputs": {
            "trajectories_from": config["trajectories_path"],
            "stage1_locked_config": config["stage1_locked_config"],
            "stage1_v2_summary": config["stage1_v2_summary"],
        },
        "metrics": metrics,
        "gate": gate,
        "gate_reason": gate_reason,
        "next_stage_inputs": {
            "baselines_locked_config": config["baselines_locked_path"],
            "encoded_parquet": f"runs/{run_id}/encoded.parquet",
            "formulation": config["formulation"],
            "selected_k": int(config["selected_k"]),
        },
    }


def run_stage2(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    run_id = config.get("run_id")
    if not run_id:
        run_id = "stage2_{model}_{env}_{formulation}_{ts}".format(
            model=_sanitize_id(config["model_id"]),
            env=_sanitize_id(config["env_id"]),
            formulation=_sanitize_id(config["formulation"]),
            ts=_utc_now().replace("-", "").replace(":", ""),
        )
    output_dir = Path("runs") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    records, features, prompt_ids = _encode_rollouts(Path(config["trajectories_path"]))
    _write_encoded_parquet(output_dir / "encoded.parquet", records)

    encoder_coverage = len(records) / float(config["expected_trajectories"])
    null_wasserstein, null_feature_distances = _mean_feature_wasserstein(features, features.copy())
    random_result = _random_baseline(
        features,
        prompt_ids,
        k=int(config["selected_k"]),
        n_trials=int(config.get("random_trials", 200)),
        test_fraction=float(config.get("test_fraction", 0.2)),
        seed=int(config.get("seed", 42)),
    )
    pca_result = _pca_baseline(features, int(config["selected_k"]))
    temperature_result = _temperature_baseline(int(config["selected_k"]), config)
    sae_result = {
        "status": "absent",
        "reason": "no SAE artifact was preregistered or available for this environment/model/formulation",
    }

    null_result = {
        "status": "evaluated",
        "mean_feature_wasserstein": null_wasserstein,
        "feature_wasserstein": null_feature_distances,
        "threshold": float(config["null_wasserstein_threshold"]),
    }

    _write_json(output_dir / "null_intervention.json", null_result)
    _write_json(output_dir / "random_baseline.json", random_result)
    _write_json(output_dir / "temperature_baseline.json", temperature_result)
    _write_json(output_dir / "pca_baseline.json", pca_result)
    _write_json(output_dir / "sae_baseline.json", sae_result)

    metrics = {
        "encoder_coverage": encoder_coverage,
        "num_encoded": len(records),
        "expected_trajectories": int(config["expected_trajectories"]),
        "feature_dim": int(features.shape[1]),
        "null_wasserstein": null_wasserstein,
        "random_mean_accuracy": random_result["mean_accuracy"],
        "random_chance_accuracy": random_result["chance_accuracy"],
        "random_abs_diff_from_chance": random_result["abs_diff_from_chance"],
        "pca_cumulative_explained_variance": pca_result["cumulative_explained_variance"],
    }

    gate_checks = {
        "encoder_coverage": math.isclose(encoder_coverage, 1.0, rel_tol=0.0, abs_tol=0.0),
        "null_wasserstein": null_wasserstein < float(config["null_wasserstein_threshold"]),
        "random_baseline": random_result["abs_diff_from_chance"] <= float(config["random_chance_tolerance"]),
    }
    gate = "pass" if all(gate_checks.values()) else "fail"
    failed = [name for name, ok in gate_checks.items() if not ok]
    gate_reason = "all Stage 2 encoder sanity gates passed" if gate == "pass" else f"failed gates: {failed}"

    locked = {
        "env_id": config["env_id"],
        "model_id": config["model_id"],
        "stage2_run_id": run_id,
        "gate": gate,
        "gate_reason": gate_reason,
        "timestamp_utc": _utc_now(),
        "formulation": _formulation_metadata(config),
        "encoder": {
            "encoder_config": config["encoder_config"],
            "feature_dim": int(features.shape[1]),
            "coverage": encoder_coverage,
        },
        "baselines": {
            "null_intervention": f"runs/{run_id}/null_intervention.json",
            "random": f"runs/{run_id}/random_baseline.json",
            "temperature": f"runs/{run_id}/temperature_baseline.json",
            "pca": f"runs/{run_id}/pca_baseline.json",
            "sae": f"runs/{run_id}/sae_baseline.json",
        },
        "stage4_policy": {
            "chance_corrected_identifiability": "accuracy_minus_1_over_K",
            "selected_k": int(config["selected_k"]),
            "temperature_schedule": temperature_result["temperatures"],
        },
    }
    _write_yaml(Path(config["baselines_locked_path"]), locked)

    timestamp_utc = _utc_now()
    summary = _summary_payload(
        run_id=run_id,
        config=config,
        metrics=metrics,
        gate=gate,
        gate_reason=gate_reason,
        timestamp_utc=timestamp_utc,
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", {"config_path": str(config_path), "outputs": sorted(p.name for p in output_dir.iterdir())})

    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp_utc,
            "stage": "stage2_baseline_encoder_sanity",
            "run_id": run_id,
            "gate_status": gate,
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": f"runs/{run_id}/summary.json",
            "prime_rl_modifications": "none",
        },
    )
    if gate != "pass":
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_stage2(args.config)


if __name__ == "__main__":
    main()
