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

from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _sanitize_id(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").replace(".", "_")


def _mean_ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if arr.size == 1:
        value = float(arr[0])
        return {"mean": value, "ci_low": value, "ci_high": value}
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def _normalization_scale(values: list[float]) -> float:
    finite = np.asarray([v for v in values if math.isfinite(v) and v > 0], dtype=np.float64)
    if finite.size == 0:
        return 1.0
    return float(np.max(finite))


def _rho_from_gamma(gamma: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(np.clip(math.sqrt(max(gamma, 0.0) / scale), 0.0, 1.0))


def _eta_from_gamma(gamma: float, median_gamma: float, base_eta: float) -> float:
    gamma = max(gamma, 1e-12)
    median_gamma = max(median_gamma, 1e-12)
    return float(base_eta * math.sqrt(median_gamma / gamma))


def _parametric_rows(config: dict[str, Any], comparison: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method = comparison["parametric_b"]
    pair_path = Path(config["parametric_b_pair_path"])
    pair = _load_json(pair_path)
    k = int(method["selected_k"])
    test_energies = pair["test"]["rayleigh_energies"][:k]
    train_energies = pair["train"]["rayleigh_energies"][:k]
    gammas = [1.0 / max(float(v), 1e-12) for v in test_energies]
    scale = _normalization_scale(gammas)
    median_gamma = float(np.median(gammas))
    rows = []
    for coordinate, (train_energy, test_energy) in enumerate(zip(train_energies, test_energies, strict=False)):
        for state_id, energy in (("train_aggregate", train_energy), ("heldout_aggregate", test_energy)):
            gamma = 1.0 / max(float(energy), 1e-12)
            rows.append(
                {
                    "env_id": config["env_id"],
                    "model_id": config["model_id"],
                    "formulation": "parametric_b",
                    "method_role": "primary",
                    "coordinate": coordinate,
                    "state_id": state_id,
                    "rho": _rho_from_gamma(gamma, scale),
                    "gamma": gamma,
                    "eta_star_state": _eta_from_gamma(gamma, median_gamma, float(config["base_eta"])),
                    "source_metric": "inverse_rayleigh_energy",
                    "patch_layer": int(method["selected_patch_layer"]),
                    "readout_layer": int(method["selected_readout_layer"]),
                }
            )
    spectrum = {
        "formulation": "parametric_b",
        "selected_k": k,
        "values": gammas,
        "value_name": "inverse_rayleigh_energy",
    }
    return rows, spectrum


def _best_average_row(path: str, method: dict[str, Any]) -> dict[str, Any]:
    payload = _load_json(path)
    for row in payload["rows"]:
        if (
            row["patch_layer"] == method["best_patch_layer"]
            and row["readout_layer"] == method["best_readout_layer"]
            and row["residual_metric"] == method["best_metric"]
        ):
            return row
    msg = f"average-controllability row not found in {path}"
    raise ValueError(msg)


def _average_rows(config: dict[str, Any], comparison: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method = comparison["average_controllability_a"]
    row = _best_average_row(config["average_controllability_path"], method)
    values = [float(v) for v in row["generalized_top_eigenvalues"][: int(config["comparator_k"])]]
    scale = _normalization_scale(values)
    median_gamma = float(np.median(values))
    rows = [
        {
            "env_id": config["env_id"],
            "model_id": config["model_id"],
            "formulation": "average_controllability_a",
            "method_role": "comparator",
            "coordinate": coordinate,
            "state_id": "aggregate",
            "rho": _rho_from_gamma(gamma, scale),
            "gamma": gamma,
            "eta_star_state": _eta_from_gamma(gamma, median_gamma, float(config["base_eta"])),
            "source_metric": "generalized_eigenvalue",
            "patch_layer": int(method["best_patch_layer"]),
            "readout_layer": int(method["best_readout_layer"]),
        }
        for coordinate, gamma in enumerate(values)
    ]
    return rows, {
        "formulation": "average_controllability_a",
        "selected_k": len(values),
        "values": values,
        "value_name": "generalized_eigenvalue",
    }


def _best_koopman_row(path: str, method: dict[str, Any]) -> dict[str, Any]:
    payload = _load_json(path)
    for row in payload["rows"]:
        if (
            row["patch_layer"] == method["best_patch_layer"]
            and row["readout_layer"] == method["best_readout_layer"]
            and row["feature_dim"] == method["best_feature_dim"]
        ):
            return row
    msg = f"Koopman/DMD row not found in {path}"
    raise ValueError(msg)


def _koopman_rows(config: dict[str, Any], comparison: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method = comparison["koopman_dmd_c"]
    row = _best_koopman_row(config["koopman_dmd_path"], method)
    singular_values = [float(v) for v in row["top_singular_values"][: int(config["comparator_k"])]]
    values = [v * v for v in singular_values]
    scale = _normalization_scale(values)
    median_gamma = float(np.median(values))
    rows = [
        {
            "env_id": config["env_id"],
            "model_id": config["model_id"],
            "formulation": "koopman_dmd_c",
            "method_role": "comparator",
            "coordinate": coordinate,
            "state_id": "aggregate",
            "rho": _rho_from_gamma(gamma, scale),
            "gamma": gamma,
            "eta_star_state": _eta_from_gamma(gamma, median_gamma, float(config["base_eta"])),
            "source_metric": "squared_operator_singular_value",
            "patch_layer": int(method["best_patch_layer"]),
            "readout_layer": int(method["best_readout_layer"]),
        }
        for coordinate, gamma in enumerate(values)
    ]
    return rows, {
        "formulation": "koopman_dmd_c",
        "selected_k": len(values),
        "values": values,
        "value_name": "squared_operator_singular_value",
    }


def _original_graph_rows(config: dict[str, Any], comparison: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = _load_json(config["original_graph_heatmap_path"])
    selected_patch = int(config["original_graph_selected_patch_layer"])
    selected_readout = int(config["original_graph_selected_readout_layer"])
    matching = None
    for row in graph["rank_sweep_rows"]:
        if row["patch_layer"] == selected_patch and (selected_readout == 0 or row["readout_layer"] == selected_readout):
            matching = row
            break
    if matching is None:
        matching = max(graph["rank_sweep_rows"], key=lambda item: item["effective_rank_median"])
    state_eigenvalues = matching.get("state_eigenvalues", [])
    flat_values = [float(v) for v in np.mean(np.asarray(state_eigenvalues, dtype=np.float64), axis=0)[: int(config["original_graph_k"])]]
    scale = _normalization_scale(flat_values)
    median_gamma = float(np.median(flat_values))
    rows = []
    for state_idx, eigenvalues in enumerate(state_eigenvalues):
        for coordinate, gamma in enumerate([float(v) for v in eigenvalues[: int(config["original_graph_k"])]]):
            rows.append(
                {
                    "env_id": config["env_id"],
                    "model_id": config["model_id"],
                    "formulation": "original_graph",
                    "method_role": "baseline_negative_control",
                    "coordinate": coordinate,
                    "state_id": f"state_{state_idx}",
                    "rho": _rho_from_gamma(gamma, scale),
                    "gamma": gamma,
                    "eta_star_state": _eta_from_gamma(gamma, median_gamma, float(config["base_eta"])),
                    "source_metric": "local_controllability_eigenvalue",
                    "patch_layer": int(matching["patch_layer"]),
                    "readout_layer": int(matching["readout_layer"]),
                }
            )
    return rows, {
        "formulation": "original_graph",
        "selected_k": int(config["original_graph_k"]),
        "values": flat_values,
        "value_name": "mean_local_controllability_eigenvalue",
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["formulation"], int(row["coordinate"])), []).append(row)

    aggregates = []
    for (formulation, coordinate), items in sorted(grouped.items()):
        rho = _mean_ci([float(item["rho"]) for item in items])
        gamma = _mean_ci([float(item["gamma"]) for item in items])
        eta = _mean_ci([float(item["eta_star_state"]) for item in items])
        first = items[0]
        aggregates.append(
            {
                "formulation": formulation,
                "coordinate": coordinate,
                "rho_mean": rho["mean"],
                "rho_ci_low": rho["ci_low"],
                "rho_ci_high": rho["ci_high"],
                "gamma_mean": gamma["mean"],
                "gamma_ci_low": gamma["ci_low"],
                "gamma_ci_high": gamma["ci_high"],
                "eta_star_mean": eta["mean"],
                "eta_star_ci_low": eta["ci_low"],
                "eta_star_ci_high": eta["ci_high"],
                "source_metric": first["source_metric"],
                "patch_layer": first["patch_layer"],
                "readout_layer": first["readout_layer"],
            }
        )
    return aggregates


def _linearization_rows(aggregates: list[dict[str, Any]], eta_grid: list[float]) -> list[dict[str, Any]]:
    rows = []
    for item in aggregates:
        eta_star = max(float(item["eta_star_mean"]), 1e-12)
        gamma = float(item["gamma_mean"])
        for eta in eta_grid:
            ratio = eta / eta_star
            rows.append(
                {
                    "formulation": item["formulation"],
                    "coordinate": int(item["coordinate"]),
                    "eta": float(eta),
                    "eta_star": eta_star,
                    "predicted_displacement": float(eta * math.sqrt(max(gamma, 0.0))),
                    "linearization_r2_surrogate": float(math.exp(-(ratio**2))),
                    "linearization_cosine_surrogate": float(max(0.0, 1.0 - 0.5 * ratio**2)),
                }
            )
    return rows


def run_stage3(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    comparison = _load_json(config["method_comparison_path"])
    timestamp = _utc_now()
    run_id = config.get("run_id") or (
        f"stage3_v2_{_sanitize_id(config['model_id'])}_{_sanitize_id(config['env_id'])}_{timestamp.replace(':', '').replace('-', '')}"
    )
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    for loader in (_parametric_rows, _average_rows, _koopman_rows, _original_graph_rows):
        method_rows, spectrum = loader(config, comparison)
        rows.extend(method_rows)
        spectra.append(spectrum)

    aggregates = _aggregate_rows(rows)
    eta_grid = [float(value) for value in config.get("eta_grid", [0.025, 0.05, 0.1, 0.2, 0.4])]
    linearization = _linearization_rows(aggregates, eta_grid)

    _write_parquet(run_dir / "per_coordinate_geometry.parquet", rows)
    _write_parquet(run_dir / "linearization_sweep.parquet", linearization)
    _write_json(run_dir / "aggregated_per_coordinate.json", {"rows": aggregates})
    _write_json(run_dir / "controllability_spectrum.json", {"spectra": spectra})

    primary = [row for row in aggregates if row["formulation"] == "parametric_b"]
    metrics = {
        "primary_coordinates": len(primary),
        "primary_mean_rho": float(np.mean([row["rho_mean"] for row in primary])),
        "primary_mean_gamma": float(np.mean([row["gamma_mean"] for row in primary])),
        "primary_min_eta_star": float(np.min([row["eta_star_mean"] for row in primary])),
        "formulations": sorted({row["formulation"] for row in aggregates}),
    }
    summary = {
        "stage": "stage3_geometric_profile",
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": config,
        "inputs": {
            "stage1_v2_summary": config["stage1_v2_summary"],
            "stage2_primary_summary": config["stage2_primary_summary"],
            "method_comparison": config["method_comparison_path"],
        },
        "metrics": metrics,
        "gate": "pass",
        "gate_reason": "descriptive Stage 3 v2 profile completed",
        "next_stage_inputs": {
            "per_coordinate_geometry": f"runs/{run_id}/per_coordinate_geometry.parquet",
            "aggregated_per_coordinate": f"runs/{run_id}/aggregated_per_coordinate.json",
            "linearization_sweep": f"runs/{run_id}/linearization_sweep.parquet",
        },
        "limitations": [
            "Stage 3 v2 uses saved Stage 1 spectra and aggregate train/heldout energy profiles; it is not a steering outcome.",
            "The rho field is a normalized gain/reachability proxy for cross-formulation profiling.",
            "Linearization sweep values are analytic trust-region surrogates; finite-difference steering validation remains downstream.",
        ],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"config_path": str(config_path), "outputs": sorted(p.name for p in run_dir.iterdir())})
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp,
            "stage": "stage3_geometric_profile",
            "run_id": run_id,
            "gate_status": "pass",
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
    run_stage3(args.config)


if __name__ == "__main__":
    main()
