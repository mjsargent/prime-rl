from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _stage4_runs() -> list[Path]:
    return sorted(Path("runs").glob("stage4_v2_*_stage5/summary.json"))


def _audit(stage: str, run_id: str, metrics: dict[str, Any]) -> None:
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": _utc_now(),
            "stage": stage,
            "run_id": run_id,
            "gate_status": "documented",
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": "runs/phase1_5_pipeline_corrections/summary.json",
            "prime_rl_modifications": "none",
        },
    )


def write_corrections() -> dict[str, Any]:
    run_id = "phase1_5_pipeline_corrections"
    run_dir = Path("runs") / run_id
    stage4_rows = []
    for path in _stage4_runs():
        summary = _load_json(path)
        metrics = summary.get("metrics", {})
        stage4_rows.append(
            {
                "run_id": summary.get("run_id", path.parent.name),
                "env_id": summary.get("config", {}).get("env_id"),
                "formulation": metrics.get("formulation"),
                "controllers": metrics.get("controllers", []),
                "signs": metrics.get("signs", []),
                "coordinates": metrics.get("coordinates", []),
                "limitation": "reduced Stage 4 v2 stage5 run contained only closed_loop and + sign steering cells",
            }
        )
    metrics = {
        "num_stage4_stage5_runs": len(stage4_rows),
        "stage4_restricted_to_closed_loop_positive_sign": all(
            row["controllers"] in [["closed_loop", "none"], ["closed_loop"]] and row["signs"] in [[], ["+"]]
            for row in stage4_rows
        ),
        "stage5_status": "underpowered_restricted_data_not_final",
        "stage6_status": "underpowered_restricted_data_not_final",
        "required_followup": "Phase 1.5 pipeline smoke with verifier loop, revised encoder, all controllers, and both signs before Phase 2",
    }
    summary = {
        "stage": "phase1_5_pipeline_corrections",
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "metrics": metrics,
        "stage4_runs": stage4_rows,
        "corrections": [
            "Original reduced Stage 4 v2 stage5 runs generated closed_loop-only, positive-sign-only steered data.",
            "Original reduced Stage 5 null/weak-positive findings are underpowered and on restricted data, not final evidence.",
            "Original reduced Stage 6 correlation null is underpowered and on restricted data, not final evidence.",
            "Graph condition is a PCA proxy unless a real off-sample graph/Nyström extension is implemented.",
        ],
        "gate": "documented",
        "gate_reason": "pipeline limitations recorded before Phase 1.5 and Phase 2",
    }
    _write_json(run_dir / "summary.json", summary)
    report = Path("reports/phase1_5_pipeline_corrections.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Phase 1.5 Pipeline Corrections\n\n"
        "The reduced Stage 4 v2 stage5 runs are corrected in the audit record as closed-loop-only and positive-sign-only. "
        "The reduced Stage 5 and Stage 6 null/weak-positive results are therefore underpowered restricted-data results, not final outcomes.\n\n"
        "Phase 2 remains blocked until a Phase 1.5 smoke run passes with verifier-loop trajectories, real quality scores, revised encoder variation, and complete controller/sign coverage.\n"
    )
    _audit("phase1_5_stage4_grid_correction", run_id, metrics)
    _audit("phase1_5_stage5_restricted_data_correction", run_id, metrics)
    _audit("phase1_5_stage6_restricted_data_correction", run_id, metrics)
    return summary


if __name__ == "__main__":
    write_corrections()
