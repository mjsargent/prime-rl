from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from controllability.encoders.sentence_encoder import _trajectory_text
from controllability.envs.trajectory_schema import read_jsonl
from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _audit(stage: str, run_id: str, gate: str, metrics: dict[str, Any]) -> None:
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


def steering_application() -> dict[str, Any]:
    run_id = "diagnostic_steering_application"
    run_dir = Path("runs") / run_id
    source_path = Path("controllability/envs/hf_steered_client.py")
    source = source_path.read_text()
    stage4_source = Path("controllability/experiments/stage4_trajectory_generation.py").read_text()
    metrics = {
        "closed_loop_recomputes_delta_inside_generation_chunks": "while remaining > 0:" in source
        and "delta, _direction, _grad_hidden = _direction_and_delta" in source,
        "closed_loop_applies_patch_per_chunk": "max_new_tokens=min(self.closed_loop_chunk_tokens, remaining)" in source
        and "apply_once=False" in source,
        "closed_loop_records_only_first_chunk_by_default": 'linearization_diagnostic_scope: str = "turn"' in source
        and 'chunk_index == 0' in source,
        "patch_position": "last_token",
        "patch_position_consistent_with_stage4_single_completion": "patch_position=-1" in source
        and "patch_position=-1" in stage4_source,
        "trust_region_scope": "per_edit_eta_star_only",
        "cumulative_trust_region_cap_implemented": False,
        "per_turn_or_per_trajectory_application": "per_assistant_turn_for_one_shot/open_loop; per_generation_chunk_for_closed_loop",
    }
    findings = [
        {
            "finding": "Closed-loop verifier steering applies residual edits inside every generation chunk, not once per trajectory.",
            "evidence": "HFSteeredChatClient.get_native_response recomputes delta inside a while loop and calls _generate_with_delta(... apply_once=False) for each chunk.",
        },
        {
            "finding": "The diagnostic summary records only chunk_index=0 by default for closed-loop, so Phase 1.5 under-reports within-turn repeated edits.",
            "evidence": "linearization_diagnostic_scope defaults to 'turn', and closed-loop records only when chunk_index == 0.",
        },
        {
            "finding": "Patch position is last-token at application time, which is syntactically consistent with Stage 3/Stage 4 single-completion code but semantically different across tool-loop turns.",
            "evidence": "All patch and VJP calls use patch_position=-1 / readout_position=-1.",
        },
        {
            "finding": "Trust-region checking is per edit, not cumulative over chunks or turns.",
            "evidence": "trust_region_violation is edit_norm > eta_star for each diagnostic row; no cumulative cap is tracked.",
        },
    ]
    summary = {
        "stage": "diagnostic_steering_application",
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": _git_hash(),
        "metrics": metrics,
        "findings": findings,
        "gate": "fail",
        "gate_reason": "Steering application semantics differ from the single-application trust-region assumption; fix before Stage 3 v2 or Phase 2.",
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "findings.json", findings)
    _audit(summary["stage"], run_id, summary["gate"], metrics)
    return summary


def encoder_audit() -> dict[str, Any]:
    run_id = "diagnostic_encoder_audit"
    run_dir = Path("runs") / run_id
    traj_path = Path("runs/phase1_5_swe_grep_verifier_smoke_parametric_b/trajectories.jsonl")
    encoded_path = Path("runs/phase1_5_swe_grep_verifier_smoke_parametric_b/encoded.parquet")
    trajectories = read_jsonl(traj_path)[:10]
    rows = pq.read_table(encoded_path).to_pylist()
    if len(rows) < 10:
        raise ValueError(f"expected at least 10 encoded rows, found {len(rows)}")
    features = np.asarray([row["behavior_features"] for row in rows[:10]], dtype=np.float32)
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)
    full_features = np.asarray([row["behavior_features"] for row in rows], dtype=np.float32)
    global_var = full_features.var(axis=0)
    prompt_variances = []
    for prompt_id in sorted({str(row["prompt_id"]).split(":", 1)[0] for row in rows}):
        prompt_rows = [row for row in rows if str(row["prompt_id"]).split(":", 1)[0] == prompt_id]
        if len(prompt_rows) >= 2:
            prompt_features = np.asarray([row["behavior_features"] for row in prompt_rows], dtype=np.float32)
            prompt_variances.append(prompt_features.var(axis=0))
    mean_within_prompt_var = np.asarray(prompt_variances).mean(axis=0) if prompt_variances else np.zeros(full_features.shape[1])
    text_records = []
    for index, traj in enumerate(trajectories):
        text = _trajectory_text(traj)
        text_records.append(
            {
                "index": index,
                "trajectory_id": traj.trajectory_id,
                "reward": traj.reward,
                "tool_call_count": len(traj.tool_calls),
                "message_count": len(traj.messages),
                "character_count": len(text),
                "text": text,
            }
        )
    vector_records = [
        {
            "index": index,
            "trajectory_id": rows[index]["trajectory_id"],
            "feature_dim": int(features.shape[1]),
            "l2_norm": float(np.linalg.norm(features[index])),
            "vector": features[index].tolist(),
        }
        for index in range(10)
    ]
    pairwise_path = run_dir / "pairwise_distances.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    with pairwise_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "distance"])
        for i in range(10):
            for j in range(10):
                writer.writerow([i, j, float(distances[i, j])])
    upper = distances[np.triu_indices(10, 1)]
    metrics = {
        "sample_size": 10,
        "feature_dim": int(features.shape[1]),
        "global_active_fraction_var_gt_1e_8": float(np.mean(global_var > 1e-8)),
        "within_prompt_active_fraction_var_gt_1e_8": float(np.mean(mean_within_prompt_var > 1e-8)),
        "num_prompt_groups": int(len(prompt_variances)),
        "sample_pairwise_distance_min": float(upper.min()),
        "sample_pairwise_distance_mean": float(upper.mean()),
        "sample_pairwise_distance_max": float(upper.max()),
        "sample_text_character_min": min(item["character_count"] for item in text_records),
        "sample_text_character_mean": float(np.mean([item["character_count"] for item in text_records])),
        "sample_text_character_max": max(item["character_count"] for item in text_records),
        "sample_tool_call_count_mean": float(np.mean([item["tool_call_count"] for item in text_records])),
    }
    gate = "fail" if metrics["within_prompt_active_fraction_var_gt_1e_8"] < 0.5 else "pass"
    summary = {
        "stage": "diagnostic_encoder_audit",
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": _git_hash(),
        "metrics": metrics,
        "gate": gate,
        "gate_reason": "Within-prompt encoder feature variation remains below 50% active-fraction threshold"
        if gate == "fail"
        else "Encoder outputs show sufficient within-prompt feature variation",
        "artifacts": {
            "trajectory_texts": f"runs/{run_id}/trajectory_texts.json",
            "encoder_vectors": f"runs/{run_id}/encoder_vectors.json",
            "pairwise_distances": f"runs/{run_id}/pairwise_distances.csv",
        },
    }
    _write_json(run_dir / "trajectory_texts.json", text_records)
    _write_json(run_dir / "encoder_vectors.json", vector_records)
    _write_json(run_dir / "summary.json", summary)
    _audit(summary["stage"], run_id, gate, metrics)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostic", choices=["steering_application", "encoder_audit"])
    args = parser.parse_args()
    if args.diagnostic == "steering_application":
        summary = steering_application()
    else:
        summary = encoder_audit()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
