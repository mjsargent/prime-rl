from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def _concat_jsonl(inputs: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as dst:
        for path in inputs:
            if not path.exists():
                continue
            with path.open() as src:
                for line in src:
                    if line.strip():
                        dst.write(line)
                        count += 1
    return count


def _concat_parquet(inputs: list[Path], output: Path) -> int:
    tables = [pq.read_table(path) for path in inputs if path.exists()]
    if not tables:
        pq.write_table(pa.table({}), output)
        return 0
    table = pa.concat_tables(tables, promote_options="default")
    pq.write_table(table, output)
    return table.num_rows


def _weighted_mean(items: list[tuple[float, int]]) -> float:
    denom = sum(weight for _value, weight in items)
    if denom == 0:
        return 0.0
    return float(sum(value * weight for value, weight in items) / denom)


def merge_phase2_shards(
    *,
    base_run_id: str,
    output_run_id: str,
    num_shards: int,
) -> dict[str, Any]:
    output_dir = Path("runs") / output_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = [Path("runs") / f"{base_run_id}_shard{index:02d}" for index in range(num_shards)]
    summaries = [_read_json(path / "summary.json") for path in shard_dirs]
    missing = [str(path) for path in shard_dirs if not (path / "summary.json").exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard summaries: {missing}")

    gates = [summary.get("gate") for summary in summaries]
    metrics_rows = [summary["metrics"] for summary in summaries]

    trajectory_count = _concat_jsonl([path / "trajectories.jsonl" for path in shard_dirs], output_dir / "trajectories.jsonl")
    raw_count = _concat_jsonl([path / "raw_rollouts.jsonl" for path in shard_dirs], output_dir / "raw_rollouts.jsonl")
    failed_rows = _concat_jsonl([path / "failed_jobs.jsonl" for path in shard_dirs], output_dir / "failed_jobs.jsonl")
    encoded_count = _concat_parquet([path / "encoded.parquet" for path in shard_dirs], output_dir / "encoded.parquet")

    diagnostic_tables = []
    for shard_dir in shard_dirs:
        path = shard_dir / "turn_linearization_diagnostics.json"
        if path.exists():
            rows = _read_json(path).get("rows", [])
            diagnostic_tables.extend(rows)
    _write_json(output_dir / "turn_linearization_diagnostics.json", {"rows": diagnostic_tables})

    cosines = np.asarray([float(row.get("linearization_cosine", 0.0)) for row in diagnostic_tables], dtype=np.float32)
    r2s = np.asarray([float(row.get("linearization_r2", 0.0)) for row in diagnostic_tables], dtype=np.float32)
    if len(cosines):
        linearization = {
            "num_turn_diagnostics": int(len(cosines)),
            "median_turn_linearization_cosine": float(np.median(cosines)),
            "median_turn_linearization_r2": float(np.median(r2s)),
            "turn_linearization_cosine_below_0_7": bool(float(np.median(cosines)) < 0.7),
            "turn_linearization_cosine_p10": float(np.quantile(cosines, 0.1)),
            "turn_linearization_r2_p10": float(np.quantile(r2s, 0.1)),
        }
    else:
        linearization = {
            "num_turn_diagnostics": 0,
            "median_turn_linearization_cosine": 0.0,
            "median_turn_linearization_r2": 0.0,
            "turn_linearization_cosine_below_0_7": True,
        }

    num_trajectories = sum(int(metrics.get("num_trajectories", 0)) for metrics in metrics_rows)
    num_failed = sum(int(metrics.get("num_failed", 0)) for metrics in metrics_rows)
    num_jobs = sum(int(metrics.get("num_jobs", 0)) for metrics in metrics_rows)
    total_jobs = max(int(metrics.get("num_total_jobs_unsharded") or 0) for metrics in metrics_rows)
    metrics = {
        "num_jobs": num_jobs,
        "num_total_jobs_unsharded": total_jobs,
        "num_trajectories": num_trajectories,
        "num_failed": num_failed,
        "raw_rollout_rows": raw_count,
        "trajectory_rows": trajectory_count,
        "failed_rows": failed_rows,
        "encoded_rows": encoded_count,
        "tool_call_count": sum(int(metrics.get("tool_call_count", 0)) for metrics in metrics_rows),
        "reward_mean": _weighted_mean(
            [(float(metrics.get("reward_mean", 0.0)), int(metrics.get("num_trajectories", 0))) for metrics in metrics_rows]
        ),
        "nonzero_reward_count": sum(int(metrics.get("nonzero_reward_count", 0)) for metrics in metrics_rows),
        "mean_generated_tokens": _weighted_mean(
            [
                (float(metrics.get("mean_generated_tokens", 0.0)), int(metrics.get("num_trajectories", 0)))
                for metrics in metrics_rows
            ]
        ),
        "encoder_active_feature_fraction": _weighted_mean(
            [
                (float(metrics.get("encoder_active_feature_fraction", 0.0)), int(metrics.get("num_trajectories", 0)))
                for metrics in metrics_rows
            ]
        ),
        **linearization,
        "shard_gates": gates,
        "shard_run_ids": [summary["run_id"] for summary in summaries],
    }
    gate = "pass" if all(gate == "pass" for gate in gates) and num_failed == 0 and num_trajectories == num_jobs else "fail"
    first = summaries[0]
    summary = {
        "stage": "phase2_verifier_scaleup",
        "run_id": output_run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": first.get("git_hash"),
        "prime_rl_git_hash": first.get("prime_rl_git_hash"),
        "prime_rl_modifications": first.get("prime_rl_modifications", []),
        "config": {
            **first.get("config", {}),
            "merged_from_base_run_id": base_run_id,
            "num_shards": num_shards,
        },
        "metrics": metrics,
        "gate": gate,
        "gate_reason": "All Phase 2 shards passed and merged cleanly"
        if gate == "pass"
        else "One or more Phase 2 shards failed or merge counts are incomplete",
        "next_stage_inputs": {
            "encoded": f"runs/{output_run_id}/encoded.parquet",
            "trajectories": f"runs/{output_run_id}/trajectories.jsonl",
            "raw_rollouts": f"runs/{output_run_id}/raw_rollouts.jsonl",
            "failed_jobs": f"runs/{output_run_id}/failed_jobs.jsonl",
            "turn_linearization_diagnostics": f"runs/{output_run_id}/turn_linearization_diagnostics.json",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    for name in ("completeness_gate_self_test.json", "completeness_negative_config.yaml"):
        source = shard_dirs[0] / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": summary["timestamp_utc"],
            "stage": summary["stage"],
            "run_id": output_run_id,
            "gate_status": gate,
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": f"runs/{output_run_id}/summary.json",
            "prime_rl_modifications": "none",
        },
    )
    if gate != "pass":
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--num-shards", required=True, type=int)
    args = parser.parse_args()
    summary = merge_phase2_shards(
        base_run_id=args.base_run_id,
        output_run_id=args.output_run_id,
        num_shards=args.num_shards,
    )
    print(json.dumps({"gate": summary["gate"], "metrics": summary["metrics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
