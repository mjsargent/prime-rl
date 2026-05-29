from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _safe_id(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").replace(".", "_")


@dataclass(frozen=True)
class EncodedDataset:
    features: np.ndarray
    labels: np.ndarray
    prompts: np.ndarray
    quality: np.ndarray
    coherence: np.ndarray
    coordinates: list[int]


def _read_encoded(path: Path, *, controller: str | None, sign: str | None) -> EncodedDataset:
    rows = pq.read_table(path).to_pylist()
    filtered = []
    for row in rows:
        if row.get("coordinate") is None:
            continue
        if controller is not None and row.get("controller") != controller:
            continue
        if sign is not None and row.get("sign") != sign:
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError(f"No encoded rows matched controller={controller!r}, sign={sign!r} in {path}")
    coordinates = sorted({int(row["coordinate"]) for row in filtered})
    label_map = {coordinate: idx for idx, coordinate in enumerate(coordinates)}
    return EncodedDataset(
        features=np.asarray([row["behavior_features"] for row in filtered], dtype=np.float32),
        labels=np.asarray([label_map[int(row["coordinate"])] for row in filtered], dtype=np.int64),
        prompts=np.asarray([str(row["prompt_id"]).split(":", 1)[0] for row in filtered]),
        quality=np.asarray([float(row.get("quality", 0.0)) for row in filtered], dtype=np.float32),
        coherence=np.asarray([float(row.get("coherence", 0.0)) for row in filtered], dtype=np.float32),
        coordinates=coordinates,
    )


def _split_by_prompt(prompts: np.ndarray, *, seed: int, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted(set(prompts.tolist())))
    if len(unique) < 2:
        raise ValueError("Need at least two prompts for by-prompt train/test split")
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
    x_train = features[train_mask]
    x_test = features[test_mask]
    y_train = labels[train_mask]
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
) -> dict[str, Any]:
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


def _bootstrap_recall_by_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prompts: np.ndarray,
    *,
    num_classes: int,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    unique_prompts = np.asarray(sorted(set(prompts.tolist())))
    rng = np.random.default_rng(seed)
    rows = []
    for label in range(num_classes):
        mask = y_true == label
        recall = float(np.mean(y_pred[mask] == label)) if np.any(mask) else 0.0
        samples = []
        for _ in range(n_bootstrap):
            chosen = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
            true_values = []
            pred_values = []
            for prompt in chosen:
                row_mask = (prompts == prompt) & (y_true == label)
                true_values.extend(y_true[row_mask].tolist())
                pred_values.extend(y_pred[row_mask].tolist())
            if true_values:
                true_arr = np.asarray(true_values)
                pred_arr = np.asarray(pred_values)
                samples.append(float(np.mean(pred_arr == true_arr)))
            else:
                samples.append(0.0)
        low, high = np.quantile(samples, [0.025, 0.975]).tolist()
        rows.append({"coordinate": label, "recall": recall, "ci_low": float(low), "ci_high": float(high)})
    return rows


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> list[list[float]]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for truth, pred in zip(y_true, y_pred, strict=True):
        matrix[int(truth), int(pred)] += 1.0
    denom = matrix.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return (matrix / denom).tolist()


def _evaluate_dataset(
    dataset: EncodedDataset,
    *,
    seed: int,
    test_fraction: float,
    n_bootstrap: int,
    steps: int,
) -> dict[str, Any]:
    train_mask, test_mask = _split_by_prompt(dataset.prompts, seed=seed, test_fraction=test_fraction)
    y_pred = _fit_logistic(
        dataset.features,
        dataset.labels,
        train_mask=train_mask,
        test_mask=test_mask,
        seed=seed,
        steps=steps,
    )
    y_true = dataset.labels[test_mask]
    test_prompts = dataset.prompts[test_mask]
    num_classes = len(dataset.coordinates)
    acc = _bootstrap_accuracy(y_true, y_pred, test_prompts, n_bootstrap=n_bootstrap, seed=seed + 17)
    chance = 1.0 / num_classes
    per_coordinate = _bootstrap_recall_by_class(
        y_true,
        y_pred,
        test_prompts,
        num_classes=num_classes,
        n_bootstrap=n_bootstrap,
        seed=seed + 31,
    )
    return {
        "accuracy": acc["accuracy"],
        "accuracy_ci_low": acc["ci_low"],
        "accuracy_ci_high": acc["ci_high"],
        "chance": chance,
        "identifiability": acc["accuracy"] - chance,
        "identifiability_ci_low": acc["ci_low"] - chance,
        "identifiability_ci_high": acc["ci_high"] - chance,
        "num_classes": num_classes,
        "num_train": int(train_mask.sum()),
        "num_test": int(test_mask.sum()),
        "train_prompts": int(len(set(dataset.prompts[train_mask].tolist()))),
        "test_prompts": int(len(set(dataset.prompts[test_mask].tolist()))),
        "per_coordinate": per_coordinate,
        "confusion_matrix": _confusion(y_true, y_pred, num_classes=num_classes),
    }


def _subset(dataset: EncodedDataset, mask: np.ndarray) -> EncodedDataset | None:
    if int(mask.sum()) < len(dataset.coordinates) * 2:
        return None
    labels = dataset.labels[mask]
    if len(set(labels.tolist())) != len(dataset.coordinates):
        return None
    return EncodedDataset(
        features=dataset.features[mask],
        labels=labels,
        prompts=dataset.prompts[mask],
        quality=dataset.quality[mask],
        coherence=dataset.coherence[mask],
        coordinates=dataset.coordinates,
    )


def _stage2_random_baseline(path: Path, env: str) -> dict[str, Any]:
    data = _load_json(path)
    if data.get("status") != "evaluated":
        return {"env": env, "method": "random", "status": data.get("status", "missing")}
    values = np.asarray(data["trial_accuracies"], dtype=np.float64)
    chance = float(data["chance_accuracy"])
    low, high = np.quantile(values - chance, [0.025, 0.975]).tolist()
    return {
        "env": env,
        "method": "random",
        "status": "evaluated",
        "accuracy": float(data["mean_accuracy"]),
        "chance": chance,
        "identifiability": float(data["mean_accuracy"]) - chance,
        "identifiability_ci_low": float(low),
        "identifiability_ci_high": float(high),
    }


def _not_evaluated(env: str, method: str, reason: str) -> dict[str, Any]:
    return {
        "env": env,
        "method": method,
        "status": "not_evaluated",
        "accuracy": 0.0,
        "chance": 0.0,
        "identifiability": 0.0,
        "identifiability_ci_low": 0.0,
        "identifiability_ci_high": 0.0,
        "reason": reason,
    }


def run_stage5(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    run_id = config.get("run_id") or f"stage5_identifiability_{_utc_now().replace(':', '')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "confusion_matrices").mkdir(exist_ok=True)

    seed = int(config.get("seed", 42))
    n_bootstrap = int(config.get("n_bootstrap", 500))
    test_fraction = float(config.get("test_fraction", 0.25))
    steps = int(config.get("logistic_steps", 400))
    controller_raw = config.get("controller", "closed_loop")
    controller = None if controller_raw == "any" else str(controller_raw)
    sign = config.get("sign", "+")
    sign = None if sign == "any" else str(sign)

    rows = []
    conditioned_rows = []
    per_coordinate_rows = []
    quality_rows = []
    confusion_paths = {}
    stage4_inputs: dict[str, str] = {}
    datasets: dict[tuple[str, str], EncodedDataset] = {}

    for env_name, env_cfg in config["environments"].items():
        for formulation, run in env_cfg["stage4_runs"].items():
            summary = _load_json(Path("runs") / run / "summary.json")
            encoded = Path(summary["next_stage_inputs"]["encoded"])
            stage4_inputs[f"{env_name}:{formulation}"] = str(encoded)
            dataset = _read_encoded(encoded, controller=controller, sign=sign)
            datasets[(env_name, formulation)] = dataset
            result = _evaluate_dataset(
                dataset,
                seed=seed,
                test_fraction=test_fraction,
                n_bootstrap=n_bootstrap,
                steps=steps,
            )
            row = {
                "env": env_name,
                "method": formulation,
                "status": "evaluated",
                **{k: v for k, v in result.items() if k not in {"per_coordinate", "confusion_matrix"}},
            }
            rows.append(row)
            per_coordinate_rows.extend(
                {
                    "env": env_name,
                    "method": formulation,
                    "coordinate": item["coordinate"],
                    "recall": item["recall"],
                    "identifiability": item["recall"] - result["chance"],
                    "ci_low": item["ci_low"] - result["chance"],
                    "ci_high": item["ci_high"] - result["chance"],
                }
                for item in result["per_coordinate"]
            )
            path = run_dir / "confusion_matrices" / f"{env_name}_{formulation}.json"
            _write_json(path, {"env": env_name, "method": formulation, "matrix": result["confusion_matrix"]})
            confusion_paths[f"{env_name}:{formulation}"] = str(path)

            coherent = _subset(dataset, dataset.coherence >= float(config.get("coherence_floor", 1.0)))
            if coherent is not None:
                conditioned = _evaluate_dataset(
                    coherent,
                    seed=seed,
                    test_fraction=test_fraction,
                    n_bootstrap=n_bootstrap,
                    steps=steps,
                )
                conditioned_rows.append(
                    {
                        "env": env_name,
                        "method": formulation,
                        "conditioning": "coherence",
                        "status": "evaluated",
                        **{k: v for k, v in conditioned.items() if k not in {"per_coordinate", "confusion_matrix"}},
                    }
                )

            for floor in config.get("quality_floors", [0.0]):
                subset = _subset(dataset, dataset.quality >= float(floor))
                if subset is None:
                    quality_rows.append(
                        {
                            "env": env_name,
                            "method": formulation,
                            "quality_floor": float(floor),
                            "status": "not_enough_data",
                            "identifiability": 0.0,
                            "identifiability_ci_low": 0.0,
                            "identifiability_ci_high": 0.0,
                        }
                    )
                    continue
                quality_result = _evaluate_dataset(
                    subset,
                    seed=seed,
                    test_fraction=test_fraction,
                    n_bootstrap=n_bootstrap,
                    steps=steps,
                )
                quality_rows.append(
                    {
                        "env": env_name,
                        "method": formulation,
                        "quality_floor": float(floor),
                        "status": "evaluated",
                        "identifiability": quality_result["identifiability"],
                        "identifiability_ci_low": quality_result["identifiability_ci_low"],
                        "identifiability_ci_high": quality_result["identifiability_ci_high"],
                    }
                )

        stage2_path = env_cfg.get("stage2_parametric_b")
        if stage2_path:
            rows.append(_stage2_random_baseline(Path(stage2_path) / "random_baseline.json", env_name))
        graph_row = next(
            (row for row in rows if row["env"] == env_name and row["method"] == "original_graph"),
            None,
        )
        if graph_row is not None:
            rows.append({**graph_row, "method": "pca_proxy", "source_method": "original_graph"})
            conditioned_graph_row = next(
                (row for row in conditioned_rows if row["env"] == env_name and row["method"] == "original_graph"),
                None,
            )
            if conditioned_graph_row is not None:
                conditioned_rows.append(
                    {**conditioned_graph_row, "method": "pca_proxy", "source_method": "original_graph"}
                )
        else:
            rows.append(
                _not_evaluated(
                    env_name,
                    "pca_proxy",
                    "no original_graph/PCA-proxy Stage 4 run was included in this config",
                )
            )
        rows.append(_not_evaluated(env_name, "temperature", "temperature-as-skill Stage 4 rollouts are not yet generated"))
        rows.append(_not_evaluated(env_name, "sae", "no SAE artifact was preregistered or available"))

    primary = [row for row in conditioned_rows if row["method"] == "parametric_b"]
    pca = [row for row in conditioned_rows if row["method"] == "pca_proxy"]
    paired = []
    for primary_row in primary:
        env_name = primary_row["env"]
        pca_row = next((row for row in pca if row["env"] == env_name and row.get("status") == "evaluated"), None)
        if pca_row is None:
            paired.append(
                {
                    "env": env_name,
                    "comparison": "parametric_b_coherent_minus_pca_proxy",
                    "status": "not_evaluated",
                    "reason": "no evaluated pca_proxy row is available for this config",
                    "diff": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "passes_strong_headline": False,
                }
            )
            continue
        diff = primary_row["identifiability"] - pca_row["identifiability"]
        diff_ci_low = primary_row["identifiability_ci_low"] - pca_row["identifiability_ci_high"]
        diff_ci_high = primary_row["identifiability_ci_high"] - pca_row["identifiability_ci_low"]
        paired.append(
            {
                "env": env_name,
                "comparison": "parametric_b_coherent_minus_pca_proxy",
                "diff": diff,
                "ci_low": diff_ci_low,
                "ci_high": diff_ci_high,
                "passes_strong_headline": diff_ci_low > 0,
            }
        )

    strong_pass = bool(paired) and all(item["passes_strong_headline"] for item in paired)
    gate = "pass" if strong_pass else "conditional"
    gate_reason = (
        "parametric_b coherent identifiability exceeds PCA proxy with conservative non-overlap on all Tier A envs"
        if strong_pass
        else "strong headline gate not met; retain descriptive/conditional Stage 5 claims only"
    )

    _write_json(run_dir / "identifiability_unconditional.json", {"rows": rows})
    _write_json(run_dir / "identifiability_coherence_conditioned.json", {"rows": conditioned_rows})
    _write_json(run_dir / "identifiability_quality_conditioned.json", {"rows": quality_rows})
    _write_json(run_dir / "quality_conditioning_sensitivity.json", {"rows": quality_rows})
    _write_json(run_dir / "paired_comparisons.json", {"rows": paired})
    pq.write_table(pa.Table.from_pylist(per_coordinate_rows), run_dir / "per_coordinate_identifiability.parquet")

    metrics = {
        "num_unconditional_rows": len(rows),
        "num_conditioned_rows": len(conditioned_rows),
        "num_per_coordinate_rows": len(per_coordinate_rows),
        "strong_headline_pass": strong_pass,
        "paired_comparisons": paired,
    }
    summary = {
        "stage": "stage5_identifiability",
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": config,
        "inputs": stage4_inputs,
        "metrics": metrics,
        "gate": gate,
        "gate_reason": gate_reason,
        "next_stage_inputs": {
            "per_coordinate_identifiability": f"runs/{run_id}/per_coordinate_identifiability.parquet",
            "identifiability_unconditional": f"runs/{run_id}/identifiability_unconditional.json",
            "identifiability_coherence_conditioned": f"runs/{run_id}/identifiability_coherence_conditioned.json",
            "paired_comparisons": f"runs/{run_id}/paired_comparisons.json",
        },
        "limitations": config.get(
            "limitations",
            [
                "Temperature-as-skill and SAE baselines are represented as not_evaluated rows because their Stage 4 rollouts/artifacts are absent.",
                "The PCA baseline is available only when an original_graph/PCA-proxy Stage 4 run is included in the config.",
            ],
        ),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"config_path": str(config_path), "outputs": sorted(p.name for p in run_dir.iterdir())})
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": summary["timestamp_utc"],
            "stage": "stage5_identifiability",
            "run_id": run_id,
            "gate_status": gate,
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": f"runs/{run_id}/summary.json",
            "prime_rl_modifications": "none",
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_stage5(args.config)


if __name__ == "__main__":
    main()
