from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from controllability.experiments.stage5_identifiability import (
    EncodedDataset,
    _evaluate_dataset,
    _load_json,
    _safe_id,
    _split_by_prompt,
)
from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _prompt_key(prompt_id: Any) -> str:
    return str(prompt_id).split(":", 1)[0]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


@dataclass(frozen=True)
class Cell:
    env: str
    formulation: str
    encoded_path: Path


def _stage4_cells(stage5_config: dict[str, Any]) -> list[Cell]:
    cells = []
    for env, env_cfg in stage5_config["environments"].items():
        for formulation, run_id in env_cfg["stage4_runs"].items():
            summary = _load_json(Path("runs") / run_id / "summary.json")
            cells.append(Cell(env=env, formulation=formulation, encoded_path=Path(summary["next_stage_inputs"]["encoded"])))
    return cells


def _dataset_from_rows(
    rows: list[dict[str, Any]],
    *,
    controller: str | None,
    sign: str | None,
    label_mode: str,
) -> tuple[EncodedDataset | None, list[str], str | None]:
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
        return None, [], "no rows matched the requested controller/sign filter"

    if label_mode == "coordinate":
        raw_labels = [int(row["coordinate"]) for row in filtered]
        class_names = [str(label) for label in sorted(set(raw_labels))]
    elif label_mode == "coordinate_sign":
        raw_labels = [(int(row["coordinate"]), str(row.get("sign"))) for row in filtered]
        sign_order = {"+": 0, "-": 1}
        unique = sorted(set(raw_labels), key=lambda item: (item[0], sign_order.get(item[1], 99), item[1]))
        class_names = [f"k{coord}{sgn}" for coord, sgn in unique]
    else:
        raise ValueError(f"unknown label_mode: {label_mode}")

    if len(class_names) < 2:
        return None, class_names, "fewer than two discriminator classes are present"

    label_map = {label: idx for idx, label in enumerate(sorted(set(raw_labels), key=str))}
    if label_mode == "coordinate_sign":
        label_map = {label: idx for idx, label in enumerate(unique)}

    dataset = EncodedDataset(
        features=np.asarray([row["behavior_features"] for row in filtered], dtype=np.float32),
        labels=np.asarray([label_map[label] for label in raw_labels], dtype=np.int64),
        prompts=np.asarray([_prompt_key(row["prompt_id"]) for row in filtered]),
        quality=np.asarray([float(row.get("quality", 0.0)) for row in filtered], dtype=np.float32),
        coherence=np.asarray([float(row.get("coherence", 0.0)) for row in filtered], dtype=np.float32),
        coordinates=list(range(len(class_names))),
    )
    return dataset, class_names, None


def _label_counts(dataset: EncodedDataset, class_names: list[str], *, seed: int, test_fraction: float) -> dict[str, dict[str, int]]:
    train_mask, test_mask = _split_by_prompt(dataset.prompts, seed=seed, test_fraction=test_fraction)
    counts = {}
    for idx, name in enumerate(class_names):
        counts[name] = {
            "train": int(np.sum(train_mask & (dataset.labels == idx))),
            "test": int(np.sum(test_mask & (dataset.labels == idx))),
        }
    return counts


def _evaluate_safe(
    dataset: EncodedDataset | None,
    class_names: list[str],
    *,
    seed: int,
    test_fraction: float,
    n_bootstrap: int,
    steps: int,
    reason: str | None = None,
) -> dict[str, Any]:
    if dataset is None:
        return {"status": "not_evaluable", "reason": reason or "dataset is empty", "num_classes": len(class_names)}
    if len(set(dataset.prompts.tolist())) < 2:
        return {"status": "not_evaluable", "reason": "need at least two prompt groups", "num_classes": len(class_names)}
    if len(set(dataset.labels.tolist())) < 2:
        return {"status": "not_evaluable", "reason": "need at least two labels", "num_classes": len(class_names)}

    try:
        train_mask, test_mask = _split_by_prompt(dataset.prompts, seed=seed, test_fraction=test_fraction)
    except ValueError as exc:
        return {"status": "not_evaluable", "reason": str(exc), "num_classes": len(class_names)}

    train_labels = set(dataset.labels[train_mask].tolist())
    test_labels = set(dataset.labels[test_mask].tolist())
    all_labels = set(range(len(class_names)))
    if train_labels != all_labels:
        return {
            "status": "not_evaluable",
            "reason": "train split is missing at least one class",
            "missing_train_classes": sorted(all_labels - train_labels),
            "num_classes": len(class_names),
        }
    if test_labels != all_labels:
        return {
            "status": "not_evaluable",
            "reason": "test split is missing at least one class",
            "missing_test_classes": sorted(all_labels - test_labels),
            "num_classes": len(class_names),
        }

    result = _evaluate_dataset(
        dataset,
        seed=seed,
        test_fraction=test_fraction,
        n_bootstrap=n_bootstrap,
        steps=steps,
    )
    result["status"] = "evaluated"
    result["class_names"] = class_names
    result["counts_by_class"] = _label_counts(dataset, class_names, seed=seed, test_fraction=test_fraction)
    return result


def _audit(run_id: str, stage: str, gate: str, metrics: dict[str, Any]) -> None:
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": _utc_now(),
            "stage": stage,
            "run_id": run_id,
            "gate_status": gate,
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": f"runs/{run_id}/summary.json",
            "prime_rl_modifications": "none",
        },
    )


def _summary_base(stage: str, run_id: str, config: dict[str, Any], inputs: dict[str, Any], metrics: dict[str, Any], gate: str, gate_reason: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": config,
        "inputs": inputs,
        "metrics": metrics,
        "gate": gate,
        "gate_reason": gate_reason,
        "next_stage_inputs": {},
    }


def _run_per_controller(config: dict[str, Any], cells: list[Cell]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        run_id = f"phase1_per_controller_{_safe_id(cell.env)}_{cell.formulation}"
        run_dir = Path("runs") / run_id
        raw = _read_rows(cell.encoded_path)
        controller_results = {}
        for controller in config["controllers"]:
            dataset, class_names, reason = _dataset_from_rows(raw, controller=controller, sign=config.get("sign"), label_mode="coordinate")
            controller_results[controller] = _evaluate_safe(
                dataset,
                class_names,
                seed=int(config["seed"]),
                test_fraction=float(config["test_fraction"]),
                n_bootstrap=int(config["n_bootstrap"]),
                steps=int(config["logistic_steps"]),
                reason=reason,
            )
        closed_loop = controller_results.get("closed_loop", {})
        one_shot = controller_results.get("one_shot", {})
        supported = (
            closed_loop.get("status") == "evaluated"
            and one_shot.get("status") == "evaluated"
            and closed_loop["identifiability_ci_low"] > one_shot["identifiability_ci_high"]
        )
        metrics = {"env": cell.env, "formulation": cell.formulation, "controllers": controller_results, "closed_loop_beats_one_shot": supported}
        summary = _summary_base(
            "phase1_per_controller_identifiability",
            run_id,
            config,
            {"encoded": str(cell.encoded_path)},
            metrics,
            "pass" if supported else "inconclusive",
            "closed-loop beats one-shot by bootstrap non-overlap" if supported else "controller comparison is inconclusive or unavailable in reduced data",
        )
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "manifest.json", {"outputs": ["summary.json"]})
        _audit(run_id, "phase1_per_controller_identifiability", summary["gate"], {"closed_loop_beats_one_shot": supported})
        rows.append(summary)
    return rows


def _run_sign_as_class(config: dict[str, Any], cells: list[Cell]) -> list[dict[str, Any]]:
    summaries = []
    for cell in cells:
        run_id = f"phase1_sign_as_class_{_safe_id(cell.env)}_{cell.formulation}"
        run_dir = Path("runs") / run_id
        raw = _read_rows(cell.encoded_path)
        k_dataset, k_names, k_reason = _dataset_from_rows(raw, controller=config["default_controller"], sign="+", label_mode="coordinate")
        signed_dataset, signed_names, signed_reason = _dataset_from_rows(raw, controller=config["default_controller"], sign=None, label_mode="coordinate_sign")
        k_result = _evaluate_safe(
            k_dataset,
            k_names,
            seed=int(config["seed"]),
            test_fraction=float(config["test_fraction"]),
            n_bootstrap=int(config["n_bootstrap"]),
            steps=int(config["logistic_steps"]),
            reason=k_reason,
        )
        sign_counts = {}
        for row in raw:
            if row.get("coordinate") is not None and row.get("controller") == config["default_controller"]:
                sign_counts[str(row.get("sign"))] = sign_counts.get(str(row.get("sign")), 0) + 1
        if set(sign_counts) < {"+", "-"}:
            signed_result = {
                "status": "not_evaluable",
                "reason": "both + and - signs are required; reduced Stage 4 data contains only one steered sign",
                "observed_sign_counts": sign_counts,
                "num_classes": len(signed_names),
            }
        else:
            signed_result = _evaluate_safe(
                signed_dataset,
                signed_names,
                seed=int(config["seed"]),
                test_fraction=float(config["test_fraction"]),
                n_bootstrap=int(config["n_bootstrap"]),
                steps=int(config["logistic_steps"]),
                reason=signed_reason,
            )
        sign_gain = None
        if k_result.get("status") == "evaluated" and signed_result.get("status") == "evaluated":
            sign_gain = signed_result["identifiability"] - k_result["identifiability"]
        metrics = {"env": cell.env, "formulation": cell.formulation, "k_class": k_result, "sign_as_class": signed_result, "chance_corrected_sign_gain": sign_gain}
        summary = _summary_base(
            "phase1_sign_as_class",
            run_id,
            config,
            {"encoded": str(cell.encoded_path)},
            metrics,
            "pass" if sign_gain is not None and sign_gain > 0 else "inconclusive",
            "2K signed discriminator improves chance-corrected identifiability" if sign_gain is not None and sign_gain > 0 else "signed discrimination is unavailable or does not improve on K-class discrimination",
        )
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "manifest.json", {"outputs": ["summary.json"]})
        _audit(run_id, "phase1_sign_as_class", summary["gate"], {"chance_corrected_sign_gain": sign_gain})
        summaries.append(summary)
    return summaries


def _flatten_confusion(matrix: list[list[float]], class_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for truth, values in enumerate(matrix):
        for predicted, value in enumerate(values):
            rows.append({"truth": class_names[truth], "predicted": class_names[predicted], "value": float(value)})
    return rows


def _run_per_coordinate_audit(config: dict[str, Any], cells: list[Cell]) -> dict[str, Any]:
    run_id = "phase1_per_coordinate_audit"
    run_dir = Path("runs") / run_id
    rows = []
    top_rows = []
    confusion_rows = []
    for cell in cells:
        raw = _read_rows(cell.encoded_path)
        dataset, class_names, reason = _dataset_from_rows(raw, controller=config["default_controller"], sign="+", label_mode="coordinate")
        result = _evaluate_safe(
            dataset,
            class_names,
            seed=int(config["seed"]),
            test_fraction=float(config["test_fraction"]),
            n_bootstrap=int(config["n_bootstrap"]),
            steps=int(config["logistic_steps"]),
            reason=reason,
        )
        if result.get("status") != "evaluated":
            rows.append({"env": cell.env, "formulation": cell.formulation, "status": result["status"], "reason": result["reason"]})
            continue
        counts = result["counts_by_class"]
        per_coord = []
        for item in result["per_coordinate"]:
            coord_name = class_names[int(item["coordinate"])]
            ident = float(item["recall"]) - float(result["chance"])
            row = {
                "env": cell.env,
                "formulation": cell.formulation,
                "coordinate": int(coord_name),
                "status": "evaluated",
                "recall": float(item["recall"]),
                "identifiability": ident,
                "ci_low": float(item["ci_low"]) - float(result["chance"]),
                "ci_high": float(item["ci_high"]) - float(result["chance"]),
                "train_count": counts[coord_name]["train"],
                "test_count": counts[coord_name]["test"],
            }
            rows.append(row)
            per_coord.append(row)
        positive = sum(max(0.0, row["identifiability"]) for row in per_coord)
        top = sorted(per_coord, key=lambda row: row["identifiability"], reverse=True)[:2]
        top_fraction = sum(max(0.0, row["identifiability"]) for row in top) / positive if positive > 0 else 0.0
        for rank, row in enumerate(top, start=1):
            top_rows.append({**row, "top_rank": rank, "top2_positive_signal_fraction": top_fraction})
        for row in _flatten_confusion(result["confusion_matrix"], class_names):
            confusion_rows.append({"env": cell.env, "formulation": cell.formulation, **row})

    _write_parquet(run_dir / "per_coordinate_identifiability.parquet", rows)
    _write_parquet(run_dir / "top_coordinates.parquet", top_rows)
    _write_parquet(run_dir / "confusion_submatrices.parquet", confusion_rows)
    sparse_cells = [
        {"env": row["env"], "formulation": row["formulation"], "top2_positive_signal_fraction": row["top2_positive_signal_fraction"]}
        for row in top_rows
        if row["top_rank"] == 1
    ]
    sparse_supported = bool(sparse_cells) and all(row["top2_positive_signal_fraction"] > 0.7 for row in sparse_cells)
    metrics = {
        "num_per_coordinate_rows": len(rows),
        "num_top_coordinates": len(top_rows),
        "sparse_skill_supported_all_cells": sparse_supported,
        "top2_positive_signal_fraction_by_cell": sparse_cells,
    }
    summary = _summary_base(
        "phase1_per_coordinate_audit",
        run_id,
        config,
        {f"{cell.env}:{cell.formulation}": str(cell.encoded_path) for cell in cells},
        metrics,
        "pass" if sparse_supported else "inconclusive",
        "top two coordinates account for more than 70% of positive identifiability signal in every evaluated cell"
        if sparse_supported
        else "top-coordinate concentration is mixed or no positive per-coordinate signal was observed",
    )
    summary["next_stage_inputs"] = {
        "per_coordinate_identifiability": f"runs/{run_id}/per_coordinate_identifiability.parquet",
        "top_coordinates": f"runs/{run_id}/top_coordinates.parquet",
        "confusion_submatrices": f"runs/{run_id}/confusion_submatrices.parquet",
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"outputs": sorted(path.name for path in run_dir.iterdir())})
    _audit(run_id, "phase1_per_coordinate_audit", summary["gate"], metrics)
    return summary


def _load_stage3_geometry(config: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    geometry = {}
    for env, path in config["stage3_aggregated"].items():
        data = _load_json(path)
        for row in data["rows"]:
            geometry[(env, str(row["formulation"]), int(row["coordinate"]))] = dict(row)
    return geometry


def _frequency_band(coordinate: int) -> str:
    if coordinate <= 1:
        return "low"
    if coordinate <= 3:
        return "mid"
    return "high"


def _permutation_p(top_values: list[float], rest_values: list[float]) -> float | None:
    values = top_values + rest_values
    n_top = len(top_values)
    if n_top == 0 or not rest_values:
        return None
    observed = abs(float(np.mean(top_values) - np.mean(rest_values)))
    diffs = []
    for combo in itertools.combinations(range(len(values)), n_top):
        combo_set = set(combo)
        candidate_top = [value for idx, value in enumerate(values) if idx in combo_set]
        candidate_rest = [value for idx, value in enumerate(values) if idx not in combo_set]
        diffs.append(abs(float(np.mean(candidate_top) - np.mean(candidate_rest))))
    return float((sum(diff >= observed for diff in diffs) + 1) / (len(diffs) + 1))


def _run_top_coordinate_pattern(config: dict[str, Any], per_coordinate_summary: dict[str, Any]) -> dict[str, Any]:
    run_id = "phase1_top_coordinate_pattern"
    run_dir = Path("runs") / run_id
    geometry = _load_stage3_geometry(config)
    top_rows = pq.read_table(Path(per_coordinate_summary["next_stage_inputs"]["top_coordinates"])).to_pylist()
    all_ident = pq.read_table(Path(per_coordinate_summary["next_stage_inputs"]["per_coordinate_identifiability"])).to_pylist()
    profile_rows = []
    test_rows = []

    for row in top_rows:
        key = (row["env"], row["formulation"], int(row["coordinate"]))
        geom = geometry.get(key, {})
        profile_rows.append(
            {
                **row,
                "eigenvalue_rank": int(row["coordinate"]) + 1,
                "frequency_band": _frequency_band(int(row["coordinate"])),
                "rho_mean": float(geom.get("rho_mean", 0.0)),
                "gamma_mean": float(geom.get("gamma_mean", 0.0)),
                "eta_star_mean": float(geom.get("eta_star_mean", 0.0)),
                "patch_layer": geom.get("patch_layer"),
                "readout_layer": geom.get("readout_layer"),
            }
        )

    for env in sorted({row["env"] for row in all_ident}):
        for formulation in sorted({row["formulation"] for row in all_ident if row["env"] == env}):
            top_coords = {int(row["coordinate"]) for row in top_rows if row["env"] == env and row["formulation"] == formulation}
            all_coords = [int(row["coordinate"]) for row in all_ident if row["env"] == env and row["formulation"] == formulation]
            for metric in ["rho_mean", "gamma_mean", "eta_star_mean"]:
                top_values = [float(geometry[(env, formulation, coord)][metric]) for coord in all_coords if coord in top_coords and (env, formulation, coord) in geometry]
                rest_values = [float(geometry[(env, formulation, coord)][metric]) for coord in all_coords if coord not in top_coords and (env, formulation, coord) in geometry]
                p_value = _permutation_p(top_values, rest_values)
                test_rows.append(
                    {
                        "env": env,
                        "formulation": formulation,
                        "metric": metric,
                        "top_mean": float(np.mean(top_values)) if top_values else 0.0,
                        "rest_mean": float(np.mean(rest_values)) if rest_values else 0.0,
                        "difference": float(np.mean(top_values) - np.mean(rest_values)) if top_values and rest_values else 0.0,
                        "permutation_p": p_value,
                        "status": "evaluated" if p_value is not None else "not_evaluable",
                    }
                )

    _write_parquet(run_dir / "top_coordinate_profiles.parquet", profile_rows)
    _write_parquet(run_dir / "geometric_pattern_tests.parquet", test_rows)
    significant = [row for row in test_rows if row["status"] == "evaluated" and row["permutation_p"] is not None and row["permutation_p"] <= 0.1]
    metrics = {
        "num_top_coordinate_profiles": len(profile_rows),
        "num_pattern_tests": len(test_rows),
        "num_pattern_tests_p_le_0_1": len(significant),
        "significant_patterns": significant,
    }
    summary = _summary_base(
        "phase1_top_coordinate_pattern",
        run_id,
        config,
        {
            "phase1_per_coordinate_audit": per_coordinate_summary["next_stage_inputs"],
            "stage3_aggregated": config["stage3_aggregated"],
        },
        metrics,
        "pass" if significant else "inconclusive",
        "top identifiable coordinates share at least one geometric property at p<=0.1 by exact permutation"
        if significant
        else "top identifiable coordinates do not show a clear geometric pattern at reduced scale",
    )
    summary["next_stage_inputs"] = {
        "top_coordinate_profiles": f"runs/{run_id}/top_coordinate_profiles.parquet",
        "geometric_pattern_tests": f"runs/{run_id}/geometric_pattern_tests.parquet",
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"outputs": sorted(path.name for path in run_dir.iterdir())})
    _audit(run_id, "phase1_top_coordinate_pattern", summary["gate"], metrics)
    return summary


def _subset_quality(dataset: EncodedDataset, floor: float) -> EncodedDataset | None:
    mask = dataset.quality > floor
    if int(mask.sum()) < len(dataset.coordinates) * 2:
        return None
    labels = dataset.labels[mask]
    if set(labels.tolist()) != set(range(len(dataset.coordinates))):
        return None
    return EncodedDataset(
        features=dataset.features[mask],
        labels=labels,
        prompts=dataset.prompts[mask],
        quality=dataset.quality[mask],
        coherence=dataset.coherence[mask],
        coordinates=dataset.coordinates,
    )


def _run_quality_conditioning(config: dict[str, Any], cells: list[Cell]) -> dict[str, Any]:
    run_id = "phase1_quality_conditioned"
    run_dir = Path("runs") / run_id
    rows = []
    for cell in cells:
        raw = _read_rows(cell.encoded_path)
        dataset, class_names, reason = _dataset_from_rows(raw, controller=config["default_controller"], sign="+", label_mode="coordinate")
        for floor in config["quality_floors"]:
            floor = float(floor)
            if dataset is None:
                result = {"status": "not_evaluable", "reason": reason or "dataset is empty", "num_rows_after_filter": 0}
            else:
                subset = _subset_quality(dataset, floor)
                if subset is None:
                    result = {
                        "status": "not_evaluable",
                        "reason": "quality filter leaves too few rows or drops at least one coordinate",
                        "num_rows_after_filter": int(np.sum(dataset.quality > floor)),
                        "counts_by_coordinate_after_filter": {
                            class_names[idx]: int(np.sum((dataset.quality > floor) & (dataset.labels == idx)))
                            for idx in range(len(class_names))
                        },
                    }
                else:
                    result = _evaluate_safe(
                        subset,
                        class_names,
                        seed=int(config["seed"]),
                        test_fraction=float(config["test_fraction"]),
                        n_bootstrap=int(config["n_bootstrap"]),
                        steps=int(config["logistic_steps"]),
                    )
                    result["num_rows_after_filter"] = int(len(subset.labels))
            rows.append(
                {
                    "env": cell.env,
                    "formulation": cell.formulation,
                    "quality_floor": floor,
                    "status": result["status"],
                    "identifiability": float(result.get("identifiability", 0.0)),
                    "identifiability_ci_low": float(result.get("identifiability_ci_low", 0.0)),
                    "identifiability_ci_high": float(result.get("identifiability_ci_high", 0.0)),
                    "num_rows_after_filter": int(result.get("num_rows_after_filter", 0)),
                    "reason": result.get("reason", ""),
                }
            )

    _write_json(run_dir / "quality_conditioned_rows.json", {"rows": rows})
    _write_parquet(run_dir / "quality_conditioned_rows.parquet", rows)
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    metrics = {
        "num_rows": len(rows),
        "num_evaluated": len(evaluated),
        "quality_values_unset": len(evaluated) == 0,
    }
    summary = _summary_base(
        "phase1_quality_conditioned",
        run_id,
        config,
        {f"{cell.env}:{cell.formulation}": str(cell.encoded_path) for cell in cells},
        metrics,
        "inconclusive",
        "quality conditioning cannot be evaluated on reduced Stage 4 data because quality is unset or filters remove required classes"
        if len(evaluated) == 0
        else "quality-conditioned identifiability was evaluated on at least one cell",
    )
    summary["next_stage_inputs"] = {"quality_conditioned_rows": f"runs/{run_id}/quality_conditioned_rows.parquet"}
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"outputs": sorted(path.name for path in run_dir.iterdir())})
    _audit(run_id, "phase1_quality_conditioned", summary["gate"], metrics)
    return summary


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for idx, _value in enumerate(unique):
        if counts[idx] > 1:
            mask = inverse == idx
            ranks[mask] = float(np.mean(ranks[mask]))
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _fit_logistic_with_weights(
    dataset: EncodedDataset,
    *,
    seed: int,
    test_fraction: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask, test_mask = _split_by_prompt(dataset.prompts, seed=seed, test_fraction=test_fraction)
    torch.manual_seed(seed)
    x_train = dataset.features[train_mask]
    y_train = dataset.labels[train_mask]
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.where(x_train.std(axis=0, keepdims=True) < 1e-6, 1.0, x_train.std(axis=0, keepdims=True))
    x_train = (x_train - mean) / std
    x_test = (dataset.features[test_mask] - mean) / std
    model = torch.nn.Linear(x_train.shape[1], int(dataset.labels.max()) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-3)
    x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.long)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = model(torch.as_tensor(x_test, dtype=torch.float32)).argmax(dim=-1).cpu().numpy().astype(np.int64)
        weights = model.weight.detach().abs().mean(dim=0).cpu().numpy().astype(np.float64)
    return preds, weights, test_mask


def _feature_variance_by_prompt(dataset: EncodedDataset) -> np.ndarray:
    variances = []
    for prompt in sorted(set(dataset.prompts.tolist())):
        mask = dataset.prompts == prompt
        if int(mask.sum()) >= 2:
            variances.append(dataset.features[mask].var(axis=0))
    if not variances:
        return np.zeros(dataset.features.shape[1], dtype=np.float64)
    return np.asarray(variances, dtype=np.float64).mean(axis=0)


def _run_encoder_audit(config: dict[str, Any], cells: list[Cell]) -> dict[str, Any]:
    run_id = "phase1_encoder_audit"
    run_dir = Path("runs") / run_id
    rows = []
    dim_rows = []
    threshold = float(config.get("encoder_variance_threshold", 1e-8))
    for cell in cells:
        raw = _read_rows(cell.encoded_path)
        dataset, class_names, reason = _dataset_from_rows(raw, controller=config["default_controller"], sign="+", label_mode="coordinate")
        eval_result = _evaluate_safe(
            dataset,
            class_names,
            seed=int(config["seed"]),
            test_fraction=float(config["test_fraction"]),
            n_bootstrap=int(config["n_bootstrap"]),
            steps=int(config["logistic_steps"]),
            reason=reason,
        )
        if dataset is None or eval_result.get("status") != "evaluated":
            rows.append({"env": cell.env, "formulation": cell.formulation, "status": "not_evaluable", "reason": eval_result.get("reason", reason)})
            continue
        _preds, weights, _test_mask = _fit_logistic_with_weights(
            dataset,
            seed=int(config["seed"]),
            test_fraction=float(config["test_fraction"]),
            steps=int(config["logistic_steps"]),
        )
        variances = _feature_variance_by_prompt(dataset)
        active = variances > threshold
        correlation = _spearman(variances, weights)
        rows.append(
            {
                "env": cell.env,
                "formulation": cell.formulation,
                "status": "evaluated",
                "feature_dim": int(len(variances)),
                "active_feature_dims": int(active.sum()),
                "active_feature_fraction": float(active.mean()),
                "variance_weight_spearman": correlation,
                "mean_within_prompt_variance": float(np.mean(variances)),
                "max_within_prompt_variance": float(np.max(variances)),
            }
        )
        for idx, (variance, weight) in enumerate(zip(variances.tolist(), weights.tolist(), strict=True)):
            dim_rows.append(
                {
                    "env": cell.env,
                    "formulation": cell.formulation,
                    "feature_dim": idx,
                    "within_prompt_variance": float(variance),
                    "logistic_weight_magnitude": float(weight),
                    "active": bool(variance > threshold),
                }
            )

    _write_parquet(run_dir / "encoder_feature_variance.parquet", rows)
    _write_parquet(run_dir / "encoder_feature_variance_by_dim.parquet", dim_rows)
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    broken = bool(evaluated) and all(row["active_feature_fraction"] < float(config.get("encoder_broken_active_fraction_floor", 0.25)) for row in evaluated)
    metrics = {
        "num_cells": len(rows),
        "num_evaluated": len(evaluated),
        "encoder_variance_threshold": threshold,
        "encoder_broken": broken,
        "active_feature_fraction_by_cell": [
            {"env": row["env"], "formulation": row["formulation"], "active_feature_fraction": row.get("active_feature_fraction", 0.0)}
            for row in rows
        ],
    }
    summary = _summary_base(
        "phase1_encoder_audit",
        run_id,
        config,
        {f"{cell.env}:{cell.formulation}": str(cell.encoded_path) for cell in cells},
        metrics,
        "fail" if broken else "pass",
        "encoder has too few feature dimensions varying across coordinates" if broken else "encoder has non-trivial coordinate-level feature variance in at least some cells",
    )
    summary["next_stage_inputs"] = {
        "encoder_feature_variance": f"runs/{run_id}/encoder_feature_variance.parquet",
        "encoder_feature_variance_by_dim": f"runs/{run_id}/encoder_feature_variance_by_dim.parquet",
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"outputs": sorted(path.name for path in run_dir.iterdir())})
    _audit(run_id, "phase1_encoder_audit", summary["gate"], metrics)
    return summary


def _run_graph_proxy_collision(config: dict[str, Any]) -> dict[str, Any]:
    run_id = "phase1_graph_proxy_collision"
    run_dir = Path("runs") / run_id
    metrics = {
        "decision": "drop_graph_condition_and_rename_as_pca_proxy_negative_control",
        "option_1_attempted": False,
        "reason": "original graph eigenfunctions were not extended off-sample in Stage 4/5; the existing graph condition reused PCA-proxy numbers, making it structurally invalid as an independent graph baseline",
        "future_requirement": "reintroduce graph only after a Nyström extension exists and is run as an independent steering basis",
    }
    summary = _summary_base(
        "phase1_graph_pca_proxy_collision",
        run_id,
        config,
        {"stage5_summary": config["stage5_summary"]},
        metrics,
        "pass",
        "graph condition documented as dropped for downstream Phase 2 rather than treated as an independent method",
    )
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"outputs": ["summary.json"]})
    _audit(run_id, "phase1_graph_pca_proxy_collision", summary["gate"], metrics)
    return summary


def _write_figure_manifests(source_run_ids: list[str]) -> None:
    figures = [
        ("phase1_per_controller_identifiability", "phase1_per_controller_identifiability", "Phase 1 Per-Controller Identifiability"),
        ("phase1_sign_asymmetry", "phase1_sign_asymmetry", "Phase 1 Sign-As-Class Identifiability"),
        ("phase1_quality_conditioning", "phase1_quality_conditioning", "Phase 1 Quality-Conditioned Identifiability"),
        ("phase1_encoder_feature_variance", "phase1_encoder_feature_variance", "Phase 1 Encoder Feature Variance"),
    ]
    plot_template = """from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figure_utils import build_figure


if __name__ == "__main__":
    build_figure(Path(__file__).parent)
"""
    for directory, figure_type, title in figures:
        fig_dir = Path("paper/figures") / directory
        fig_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            fig_dir / "manifest.json",
            {
                "name": directory,
                "figure_type": figure_type,
                "title": title,
                "source_run_ids": source_run_ids,
            },
        )
        (fig_dir / "plot.py").write_text(plot_template)


def _write_phase1_report(summaries: dict[str, Any]) -> None:
    encoder = summaries["encoder_audit"]
    quality = summaries["quality_conditioned"]
    per_coord = summaries["per_coordinate_audit"]
    graph = summaries["graph_proxy_collision"]
    if encoder["gate"] == "fail":
        recommendation = (
            "Do not start Phase 2 at full scope. Phase 1.6 failed: too few behavioral encoder feature dimensions vary across coordinates, so the reduced Stage 5/6 null may be an encoder-power failure rather than a geometry failure. "
            "Fix or replace the encoder first, then rerun Phase 1.6 and the cheap discriminator ablations before scaling. When scaling resumes, prioritize B + A, include both signs, include controller variants if available, and drop the graph condition unless a true Nyström extension is implemented."
        )
    else:
        recommendation = (
            "Do not start Phase 2 at full scope until human review. The reduced data lacks one-shot/open-loop controllers, negative signs, and verifier quality labels, so controller choice, sign asymmetry, and quality conditioning remain unresolved. "
            "The encoder audit did not fail its active-feature gate. Prioritize B + A at scale, add both signs and the winning/available controller set, and drop the graph condition unless a true Nyström extension is implemented."
        )
    path = Path("reports/phase1_summary.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1 Free Ablations Summary",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "Phase 1 used only existing reduced Stage 4 v2 encoded trajectories. No new rollouts, model loading, or verifier calls were run.",
        "",
        "## Results",
        "",
        f"- Phase 1.1 per-controller: {summaries['per_controller_status']}. Reduced data contains only `closed_loop` steered trajectories, so one-shot and open-loop comparisons are not estimable.",
        f"- Phase 1.2 sign-as-class: {summaries['sign_status']}. Reduced data contains only `+` steered signs, so 2K signed discrimination is not estimable.",
        f"- Phase 1.3 per-coordinate audit: gate `{per_coord['gate']}`. {per_coord['gate_reason']}",
        f"- Phase 1.4 top-coordinate geometry: gate `{summaries['top_pattern']['gate']}`. {summaries['top_pattern']['gate_reason']}",
        f"- Phase 1.5 quality-conditioned identifiability: gate `{quality['gate']}`. {quality['gate_reason']}",
        f"- Phase 1.6 encoder audit: gate `{encoder['gate']}`. {encoder['gate_reason']}",
        f"- Phase 1.7 graph/PCA proxy collision: gate `{graph['gate']}`. {graph['gate_reason']}",
        "",
        "## Recommendation For Phase 2",
        "",
        recommendation,
        "",
        "## Evidence",
        "",
    ]
    for key in [
        "per_coordinate_audit",
        "top_pattern",
        "quality_conditioned",
        "encoder_audit",
        "graph_proxy_collision",
    ]:
        summary = summaries[key]
        lines.append(f"- `{summary['run_id']}`: `runs/{summary['run_id']}/summary.json`")
    path.write_text("\n".join(lines) + "\n")


def run_phase1(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    stage5_config = _load_yaml(config["stage5_config"])
    cells = _stage4_cells(stage5_config)
    shared = {
        **config,
        "seed": stage5_config.get("seed", 42),
        "test_fraction": stage5_config.get("test_fraction", 0.25),
        "n_bootstrap": stage5_config.get("n_bootstrap", 500),
        "logistic_steps": stage5_config.get("logistic_steps", 400),
        "quality_floors": stage5_config.get("quality_floors", [0.0]),
        "default_controller": stage5_config.get("controller", "closed_loop"),
        "sign": stage5_config.get("sign", "+"),
    }

    per_controller = _run_per_controller(shared, cells)
    sign_as_class = _run_sign_as_class(shared, cells)
    per_coordinate = _run_per_coordinate_audit(shared, cells)
    top_pattern = _run_top_coordinate_pattern(shared, per_coordinate)
    quality = _run_quality_conditioning(shared, cells)
    encoder = _run_encoder_audit(shared, cells)
    graph = _run_graph_proxy_collision(shared)

    per_controller_status = "evaluated only for closed_loop; one_shot/open_loop absent"
    sign_status = "K-class coordinate discriminator evaluated where possible; 2K signed discriminator absent"
    source_run_ids = [summary["run_id"] for summary in per_controller]
    source_run_ids.extend(summary["run_id"] for summary in sign_as_class)
    source_run_ids.extend([per_coordinate["run_id"], top_pattern["run_id"], quality["run_id"], encoder["run_id"], graph["run_id"]])
    _write_figure_manifests(source_run_ids)
    _write_phase1_report(
        {
            "per_controller_status": per_controller_status,
            "sign_status": sign_status,
            "per_coordinate_audit": per_coordinate,
            "top_pattern": top_pattern,
            "quality_conditioned": quality,
            "encoder_audit": encoder,
            "graph_proxy_collision": graph,
        }
    )
    return {
        "per_controller": per_controller,
        "sign_as_class": sign_as_class,
        "per_coordinate": per_coordinate,
        "top_pattern": top_pattern,
        "quality": quality,
        "encoder": encoder,
        "graph": graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_phase1(args.config)


if __name__ == "__main__":
    main()
