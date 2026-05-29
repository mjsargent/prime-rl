from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from controllability.envs.trajectory_schema import Trajectory, read_jsonl
from controllability.experiments.phase1_5_verifier_smoke import (
    _active_feature_fraction,
    _aggregate_linearization,
    _configured_completeness_report,
    _git_hash,
    _utc_now,
    _write_json,
    _write_summary_and_audit,
)
from controllability.experiments.stage4_trajectory_generation import _write_encoded


def _completion_tokens(traj: Trajectory) -> int:
    total = 0
    for step in traj.step_metadata:
        if not isinstance(step, dict):
            continue
        if "generated_tokens" in step:
            total += int(step["generated_tokens"])
            continue
        usage = step.get("usage") or {}
        if isinstance(usage, dict):
            total += int(usage.get("completion_tokens", 0))
    return total


def recover_smoke_summary(config_path: Path, *, limitation: str | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text()) or {}
    run_id = config["run_id"]
    run_dir = Path("runs") / run_id
    trajectories = read_jsonl(run_dir / "trajectories.jsonl")
    encoded_records = _write_encoded(
        run_dir,
        trajectories,
        encoder_name=str(config.get("encoder_name", "sentence_e5_structured_v1")),
        encoder_config=config.get("encoder_config") or {},
    )

    linearization_diagnostics: list[dict[str, Any]] = []
    for traj in trajectories:
        if not traj.steering:
            continue
        for item in traj.steering.applied_edits:
            linearization_diagnostics.append(
                {
                    **item,
                    "prompt_id": traj.prompt_id,
                    "seed": traj.seed,
                }
            )
    _write_json(run_dir / "turn_linearization_diagnostics.json", {"rows": linearization_diagnostics})
    linearization_metrics = _aggregate_linearization(linearization_diagnostics)

    failed_path = run_dir / "failed_jobs.jsonl"
    num_failed = sum(1 for _ in failed_path.open()) if failed_path.exists() else 0
    tool_call_count = sum(len(traj.tool_calls) for traj in trajectories)
    rewards = [traj.reward for traj in trajectories]
    generated_tokens = [_completion_tokens(traj) for traj in trajectories]
    active_fraction = (
        _active_feature_fraction(run_dir / "encoded.parquet", float(config.get("encoder_variance_threshold", 1e-8)))
        if encoded_records
        else 0.0
    )
    completeness_configured = _configured_completeness_report(config)
    self_test_path = run_dir / "completeness_gate_self_test.json"
    completeness_gate_test = json.loads(self_test_path.read_text()) if self_test_path.exists() else {"enabled": False}
    smoke_checks = {
        "has_trajectories": len(trajectories) > 0,
        "has_tool_calls": (not bool(config.get("require_tool_calls", True))) or tool_call_count > 0,
        "has_nonzero_quality": any(value != 0.0 for value in rewards),
        "agent_length_tokens": bool(generated_tokens)
        and float(np.mean(generated_tokens)) >= float(config.get("agent_length_token_floor", 128)),
        "encoder_active_fraction_pass": active_fraction > float(config.get("encoder_active_fraction_floor", 0.5)),
        "completeness_gate_self_test_pass": bool(
            completeness_gate_test.get("nonzero_exit_verified", not config.get("run_completeness_gate_self_test", False))
        ),
        "configured_grid_complete": bool(completeness_configured["configured_complete"]),
        "turn_linearization_cosine_pass": not linearization_metrics["turn_linearization_cosine_below_0_7"],
    }
    gate = "pass" if all(smoke_checks.values()) and not num_failed else "fail"
    metrics = {
        "num_jobs": int(config["n_prompts"])
        * len(config["seeds"])
        * len(config["controllers"])
        * len(config["coordinates"])
        * len(config["signs"]),
        "num_trajectories": len(trajectories),
        "num_failed": num_failed,
        "tool_call_count": tool_call_count,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "nonzero_reward_count": sum(1 for value in rewards if value != 0.0),
        "mean_generated_tokens": float(np.mean(generated_tokens)) if generated_tokens else 0.0,
        "encoder_active_feature_fraction": active_fraction,
        "configured_completeness": completeness_configured,
        "completeness_gate_self_test": completeness_gate_test,
        **linearization_metrics,
        "smoke_checks": smoke_checks,
    }
    limitations = []
    if limitation:
        limitations.append(limitation)
    summary = {
        "stage": "phase1_5_verifier_smoke",
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": config,
        "metrics": metrics,
        "gate": gate,
        "gate_reason": "Phase 1.5 verifier-loop smoke criteria passed"
        if gate == "pass"
        else "Phase 1.5 verifier-loop smoke criteria failed; do not start Phase 2",
        "limitations": limitations,
        "next_stage_inputs": {
            "encoded": f"runs/{run_id}/encoded.parquet",
            "trajectories": f"runs/{run_id}/trajectories.jsonl",
            "raw_rollouts": f"runs/{run_id}/raw_rollouts.jsonl",
            "failed_jobs": f"runs/{run_id}/failed_jobs.jsonl",
            "turn_linearization_diagnostics": f"runs/{run_id}/turn_linearization_diagnostics.json",
        },
    }
    return _write_summary_and_audit(run_dir, summary, exit_on_fail=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limitation", default=None)
    args = parser.parse_args()
    summary = recover_smoke_summary(args.config, limitation=args.limitation)
    print(json.dumps({"gate": summary["gate"], "metrics": summary["metrics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
