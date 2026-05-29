from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from controllability.reports.audit_table import append_audit_row


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=True)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _select_b_pair(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Formulation B summary has no pair rows")

    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        test = row["test"]
        return (
            float(test["natural_k_by_energy_gap"]),
            float(test["energy_effective_rank_inverse"]),
            -_mean([float(value) for value in test["rayleigh_energies"]]),
        )

    return max(rows, key=key)


def _summarize_graph(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    return {
        "stage": summary.get("stage"),
        "run_id": summary.get("run_id"),
        "gate": summary.get("gate"),
        "metrics": {
            "chart_invariance_median_leading10": metrics.get("chart_invariance_median_leading10"),
            "graph_vs_parametric_leading10": metrics.get("graph_vs_parametric_leading10"),
            "sample_convergence_10000_vs_50000": metrics.get("sample_convergence_10000_vs_50000"),
            "effective_rank_median": metrics.get("effective_rank_median"),
        },
    }


def _summarize_a_c(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = summary["metrics"]
    formulation_a = {
        "run_id": summary["run_id"],
        "best_patch_layer": metrics["best_average_patch_layer"],
        "best_readout_layer": metrics["best_average_readout_layer"],
        "best_metric": metrics["best_average_metric"],
        "generalized_effective_rank": metrics["best_average_generalized_effective_rank"],
        "generalized_stable_rank": metrics["best_average_generalized_stable_rank"],
    }
    formulation_c = {
        "run_id": summary["run_id"],
        "best_patch_layer": metrics["best_dmd_patch_layer"],
        "best_readout_layer": metrics["best_dmd_readout_layer"],
        "best_feature_dim": metrics["best_dmd_feature_dim"],
        "test_r2": metrics["best_dmd_test_r2"],
        "persistent_modes_abs_ge_0_90": metrics["best_dmd_persistent_modes_abs_ge_0_90"],
    }
    return formulation_a, formulation_c


def _summarize_b(summary: dict[str, Any]) -> dict[str, Any]:
    selected = _select_b_pair(summary["formulation_b_parametric_eigenfunctions"])
    test = selected["test"]
    return {
        "run_id": summary["run_id"],
        "selection_rule": "max test natural_k_by_energy_gap, then max inverse-energy effective rank, then min mean Rayleigh energy",
        "selected_patch_layer": selected["patch_layer"],
        "selected_readout_layer": selected["readout_layer"],
        "selected_k": test["natural_k_by_energy_gap"],
        "inverse_energy_effective_rank": test["energy_effective_rank_inverse"],
        "inverse_energy_stable_rank": test["energy_stable_rank_inverse"],
        "mean_rayleigh_energy": _mean([float(value) for value in test["rayleigh_energies"]]),
        "rayleigh_energies": test["rayleigh_energies"],
        "output_covariance_max_abs_offdiag": test["output_covariance_max_abs_offdiag"],
    }


def run_stage(config_path: Path) -> int:
    config = _load_yaml(config_path)
    timestamp = _utc_now()
    env_id = config["env_id"]
    model_id = config["model_id"]
    run_id = config.get("run_id") or (
        f"stage1_v2_graph_free_{_safe_name(model_id)}_"
        f"{_safe_name(env_id)}_{timestamp.replace(':', '').replace('-', '')}"
    )
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    prereg_path = Path(config["preregistration"])
    prereg = _load_yaml(prereg_path)
    inputs = {name: Path(path) for name, path in config["method_summaries"].items()}
    missing = [f"{name}: {path}" for name, path in inputs.items() if not path.exists()]

    summary_base = {
        "stage": "stage1_v2_graph_free",
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": config,
        "preregistration": prereg,
        "inputs": {name: str(path) for name, path in inputs.items()},
    }
    if missing:
        summary = {
            **summary_base,
            "metrics": {},
            "methods": {},
            "gate": "fail",
            "gate_reason": "missing method summaries: " + "; ".join(missing),
            "next_stage_inputs": {},
        }
        _write_json(run_dir / "summary.json", summary)
        append_audit_row(
            Path("runs/audit_table.md"),
            {
                "timestamp_utc": timestamp,
                "stage": "stage1_v2_graph_free",
                "run_id": run_id,
                "gate_status": "fail",
                "key_metrics": summary["gate_reason"],
                "evidence_path": str(run_dir / "summary.json"),
                "prime_rl_modifications": "none",
            },
        )
        return 1

    graph = _summarize_graph(_load_json(inputs["original_graph"]))
    a, c = _summarize_a_c(_load_json(inputs["graph_free_a_c"]))
    b = _summarize_b(_load_json(inputs["parametric_b"]))

    gates = prereg["gates"]
    gate_checks = {
        "primary_b_selected_k": b["selected_k"],
        "primary_b_inverse_energy_effective_rank": b["inverse_energy_effective_rank"],
        "primary_b_output_covariance_max_abs_offdiag": b["output_covariance_max_abs_offdiag"],
        "comparator_a_generalized_effective_rank": a["generalized_effective_rank"],
        "comparator_c_test_r2": c["test_r2"],
    }
    pass_checks = {
        "primary_b_selected_k": gate_checks["primary_b_selected_k"] >= gates["primary_b_min_k"],
        "primary_b_inverse_energy_effective_rank": gate_checks["primary_b_inverse_energy_effective_rank"]
        >= gates["primary_b_min_inverse_energy_effective_rank"],
        "primary_b_output_covariance_max_abs_offdiag": gate_checks["primary_b_output_covariance_max_abs_offdiag"]
        <= gates["primary_b_max_output_covariance_abs_offdiag"],
        "comparator_a_generalized_effective_rank": gate_checks["comparator_a_generalized_effective_rank"]
        >= gates["comparator_a_min_generalized_effective_rank"],
        "comparator_c_test_r2": gate_checks["comparator_c_test_r2"] >= gates["comparator_c_min_test_r2"],
    }
    gate_pass = all(pass_checks.values())
    gate = "pass" if gate_pass else "fail"
    gate_reason = "all Stage 1 v2 graph-free gates passed" if gate_pass else json.dumps(pass_checks, sort_keys=True)

    locked_path = Path(config["locked_output"])
    locked_payload = {
        "env_id": env_id,
        "model_id": model_id,
        "stage1_v2_run_id": run_id,
        "primary_method": "parametric_b",
        "primary_method_summary": str(inputs["parametric_b"]),
        "selected_patch_layer": b["selected_patch_layer"],
        "selected_readout_layer": b["selected_readout_layer"],
        "selected_k": int(b["selected_k"]),
        "residual_metric": prereg["methods"]["parametric_b"]["residual_metric"],
        "chart_dim": prereg["methods"]["parametric_b"]["chart_dim"],
        "comparators": {
            "average_controllability_a": str(inputs["graph_free_a_c"]),
            "koopman_dmd_c": str(inputs["graph_free_a_c"]),
            "original_graph": str(inputs["original_graph"]),
        },
        "gate": gate,
        "gate_reason": gate_reason,
        "timestamp_utc": timestamp,
    }
    if gate_pass:
        _write_yaml(locked_path, locked_payload)

    methods = {
        "original_graph": graph,
        "average_controllability_a": a,
        "parametric_b": b,
        "koopman_dmd_c": c,
    }
    metrics = {
        **gate_checks,
        "original_graph_gate": graph["gate"],
    }
    summary = {
        **summary_base,
        "methods": methods,
        "metrics": metrics,
        "gate_checks": pass_checks,
        "gate": gate,
        "gate_reason": gate_reason,
        "next_stage_inputs": {
            "locked_config": str(locked_path) if gate_pass else None,
            "primary_method": "parametric_b",
            "selected_k": int(b["selected_k"]),
        },
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "method_comparison.json", methods)
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp,
            "stage": "stage1_v2_graph_free",
            "run_id": run_id,
            "gate_status": gate,
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": str(run_dir / "summary.json"),
            "prime_rl_modifications": "none",
        },
    )
    return 0 if gate_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_stage(args.config))


if __name__ == "__main__":
    main()
