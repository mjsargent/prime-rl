from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy import stats

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


def _correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    if len(rows) < 3:
        return {
            "n": len(rows),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
            "status": "insufficient_points",
        }
    x = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {
            "n": len(rows),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
            "status": "constant_input",
        }
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n": len(rows),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "status": "ok",
    }


def _correlation_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "log_gamma_vs_identifiability": _correlation(rows, "log_gamma_mean", "identifiability"),
        "rho_vs_identifiability": _correlation(rows, "rho_mean", "identifiability"),
        "eta_star_vs_identifiability": _correlation(rows, "eta_star_mean", "identifiability"),
        "controllability_score_vs_identifiability": _correlation(rows, "controllability_score", "identifiability"),
    }


def _band_for_coordinate(coordinate: int, bands: list[dict[str, Any]]) -> str:
    for band in bands:
        lo = int(band["min_coordinate"])
        hi_raw = band.get("max_coordinate")
        if hi_raw is None:
            if coordinate >= lo:
                return str(band["name"])
            continue
        if lo <= coordinate <= int(hi_raw):
            return str(band["name"])
    return "unassigned"


def _stage3_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    formulation = str(config.get("primary_formulation", "parametric_b"))
    for env_name, env_config in config["environments"].items():
        data = _load_json(env_config["stage3_aggregated_path"])
        for row in data["rows"]:
            if row.get("formulation") != formulation:
                continue
            rows.append(
                {
                    "env": env_name,
                    "env_id": env_config["env_id"],
                    "method": formulation,
                    "coordinate": int(row["coordinate"]),
                    "rho_mean": float(row["rho_mean"]),
                    "gamma_mean": float(row["gamma_mean"]),
                    "eta_star_mean": float(row["eta_star_mean"]),
                    "patch_layer": int(row["patch_layer"]),
                    "readout_layer": int(row["readout_layer"]),
                    "source_metric": str(row.get("source_metric", "")),
                }
            )
    return rows


def _stage5_rows(path: Path, method: str) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    return [
        {
            "env": str(row["env"]),
            "method": str(row["method"]),
            "coordinate": int(row["coordinate"]),
            "recall": float(row["recall"]),
            "identifiability": float(row["identifiability"]),
            "identifiability_ci_low": float(row["ci_low"]),
            "identifiability_ci_high": float(row["ci_high"]),
        }
        for row in rows
        if row.get("method") == method
    ]


def _merge_rows(stage3_rows: list[dict[str, Any]], stage5_rows: list[dict[str, Any]], bands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    geometry = {(row["env"], row["method"], row["coordinate"]): row for row in stage3_rows}
    merged = []
    for behavior in stage5_rows:
        key = (behavior["env"], behavior["method"], behavior["coordinate"])
        if key not in geometry:
            continue
        row = {**geometry[key], **behavior}
        row["log_gamma_mean"] = float(np.log(max(row["gamma_mean"], 1e-12)))
        row["controllability_score"] = float(row["rho_mean"] * row["log_gamma_mean"])
        row["frequency_band"] = _band_for_coordinate(int(row["coordinate"]), bands)
        merged.append(row)
    return merged


def run_stage6(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    run_id = config.get("run_id") or f"stage6_geometric_behavioral_correlation_{_safe_id(_utc_now())}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    formulation = str(config.get("primary_formulation", "parametric_b"))
    bands = list(config.get("frequency_bands", [])) or [
        {"name": "low", "min_coordinate": 0, "max_coordinate": 1},
        {"name": "mid", "min_coordinate": 2, "max_coordinate": 3},
        {"name": "high", "min_coordinate": 4, "max_coordinate": None},
    ]
    scatter_rows = _merge_rows(
        _stage3_rows(config),
        _stage5_rows(Path(config["stage5_per_coordinate_path"]), formulation),
        bands,
    )
    if not scatter_rows:
        raise ValueError("No Stage 3/Stage 5 rows matched for Stage 6 correlation")

    pq.write_table(pa.Table.from_pylist(scatter_rows), run_dir / "scatter_data.parquet")

    pooled = _correlation_block(scatter_rows)
    per_env = {
        env: _correlation_block([row for row in scatter_rows if row["env"] == env])
        for env in sorted({row["env"] for row in scatter_rows})
    }
    per_band = {
        band: _correlation_block([row for row in scatter_rows if row["frequency_band"] == band])
        for band in sorted({row["frequency_band"] for row in scatter_rows})
    }
    _write_json(run_dir / "correlation_pooled.json", pooled)
    _write_json(run_dir / "correlation_per_env.json", per_env)
    _write_json(run_dir / "correlation_per_freq_band.json", per_band)

    primary = pooled["controllability_score_vs_identifiability"]
    metrics = {
        "num_rows": len(scatter_rows),
        "num_envs": len(per_env),
        "primary_formulation": formulation,
        "pooled_controllability_score_pearson_r": primary["pearson_r"],
        "pooled_controllability_score_pearson_p": primary["pearson_p"],
        "pooled_controllability_score_spearman_r": primary["spearman_r"],
        "pooled_controllability_score_spearman_p": primary["spearman_p"],
        "pooled_status": primary["status"],
    }
    summary = {
        "stage": "stage6_geometric_behavioral_correlation",
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": config,
        "inputs": {
            "stage5_per_coordinate": config["stage5_per_coordinate_path"],
            **{
                f"{env}:stage3_aggregated": env_config["stage3_aggregated_path"]
                for env, env_config in config["environments"].items()
            },
        },
        "metrics": metrics,
        "gate": "pass",
        "gate_reason": "descriptive correlation stage completed; null or weak correlations are reportable",
        "next_stage_inputs": {
            "scatter_data": f"runs/{run_id}/scatter_data.parquet",
            "correlation_pooled": f"runs/{run_id}/correlation_pooled.json",
            "correlation_per_env": f"runs/{run_id}/correlation_per_env.json",
            "correlation_per_freq_band": f"runs/{run_id}/correlation_per_freq_band.json",
        },
        "limitations": config.get(
            "limitations",
            [
                "Per-coordinate identifiability is available only for the primary parametric_b formulation in the current Stage 5 artifact.",
            ],
        ),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"config_path": str(config_path), "outputs": sorted(p.name for p in run_dir.iterdir())})
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": summary["timestamp_utc"],
            "stage": "stage6_geometric_behavioral_correlation",
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
    run_stage6(args.config)


if __name__ == "__main__":
    main()
