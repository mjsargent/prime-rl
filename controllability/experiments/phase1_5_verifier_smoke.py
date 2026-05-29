from __future__ import annotations

import argparse
import asyncio
import gc
import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import verifiers as vf
import yaml
from verifiers.utils.save_utils import make_serializable

from controllability.envs.hf_steered_client import HFSteeredChatClient
from controllability.envs.trajectory_schema import SteeringMetadata, Trajectory, trajectory_to_record
from controllability.experiments.stage4_trajectory_generation import _fit_stage4_basis, _write_encoded
from controllability.models.frozen_model import FrozenModel
from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def _rollout_input_from_example(example: dict[str, Any]) -> vf.RolloutInput:
    data = dict(example)
    task = data.get("task")
    if isinstance(task, str):
        try:
            parsed = json.loads(task)
        except json.JSONDecodeError:
            data.pop("task", None)
        else:
            if isinstance(parsed, dict):
                data["task"] = parsed
            else:
                data.pop("task", None)
    return vf.RolloutInput(**data)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, default=make_serializable, sort_keys=True) + "\n")


def _clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_id(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").replace(".", "_")


def _as_record(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _as_record(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_record(item) for item in value]
    if hasattr(value, "model_dump"):
        return _as_record(value.model_dump())
    return value


def _cell_key(row: dict[str, Any]) -> str:
    return f"{row.get('controller')}|k={int(row.get('coordinate'))}|sign={row.get('sign')}"


def _excluded_cell_keys(config: dict[str, Any]) -> set[str]:
    return {_cell_key(cell) for cell in config.get("excluded_cells", [])}


def _expected_steering_cell_keys(config: dict[str, Any]) -> set[str]:
    controllers = config.get("required_controllers", config.get("controllers", []))
    signs = config.get("required_signs", config.get("signs", []))
    coordinates = config.get("required_coordinates", config.get("coordinates", []))
    return {
        f"{controller}|k={int(coordinate)}|sign={sign}"
        for controller in controllers
        for coordinate in coordinates
        for sign in signs
    }


def _configured_steering_cell_keys(config: dict[str, Any]) -> set[str]:
    excluded = _excluded_cell_keys(config)
    return {
        f"{controller}|k={int(coordinate)}|sign={sign}"
        for controller in config.get("controllers", [])
        for coordinate in config.get("coordinates", [])
        for sign in config.get("signs", [])
        if f"{controller}|k={int(coordinate)}|sign={sign}" not in excluded
    }


def _configured_completeness_report(config: dict[str, Any]) -> dict[str, Any]:
    expected = _expected_steering_cell_keys(config)
    configured = _configured_steering_cell_keys(config)
    return {
        "require_complete_steering_grid": bool(config.get("require_complete_steering_grid", False)),
        "expected_steering_cells": sorted(expected),
        "configured_steering_cells": sorted(configured),
        "excluded_steering_cells": sorted(_excluded_cell_keys(config)),
        "missing_configured_cells": sorted(expected - configured),
        "extra_configured_cells": sorted(configured - expected),
        "configured_complete": expected <= configured,
    }


def _messages_from_rollout(output: vf.RolloutOutput) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    prompt = output.get("prompt")
    if isinstance(prompt, list):
        messages.extend(_as_record(prompt))
    for step in output.get("trajectory") or []:
        for key in ("prompt", "completion"):
            value = step.get(key)
            if isinstance(value, list):
                messages.extend(_as_record(value))
    completion = output.get("completion")
    if isinstance(completion, list):
        messages.extend(_as_record(completion))
    elif isinstance(completion, str) and completion:
        messages.append({"role": "assistant", "content": completion})
    return messages


def _tool_calls_from_rollout(output: vf.RolloutOutput) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for step in output.get("trajectory") or []:
        for message in step.get("completion") or []:
            record = _as_record(message)
            calls = record.get("tool_calls") if isinstance(record, dict) else None
            if calls:
                tool_calls.extend(_as_record(calls))
    return tool_calls


def _step_metadata_from_rollout(output: vf.RolloutOutput) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for idx, step in enumerate(output.get("trajectory") or []):
        response = _as_record(step.get("response") or {})
        usage = response.get("usage") if isinstance(response, dict) else None
        message = response.get("message") if isinstance(response, dict) else None
        metadata.append(
            {
                "step_index": idx,
                "reward": step.get("reward"),
                "is_truncated": step.get("is_truncated"),
                "usage": usage,
                "finish_reason": message.get("finish_reason") if isinstance(message, dict) else None,
            }
        )
    return metadata


def _trajectory_from_vf_output(
    output: vf.RolloutOutput,
    *,
    env_id: str,
    model_id: str,
    prompt_id: str,
    seed: int,
    wall_clock_seconds: float,
    git_hash: str,
) -> Trajectory:
    reward = float(output.get("reward") or 0.0)
    return Trajectory(
        env_id=env_id,
        model_id=model_id,
        prompt_id=prompt_id,
        seed=seed,
        messages=_messages_from_rollout(output),
        tool_calls=_tool_calls_from_rollout(output),
        step_metadata=_step_metadata_from_rollout(output),
        reward=reward,
        success=bool(output.get("solved", output.get("success", reward > 0.0))),
        steering=None,
        git_hash=git_hash,
        prime_rl_git_hash=git_hash,
        timestamp_utc=_utc_now(),
        wall_clock_seconds=wall_clock_seconds,
    )


def _rollout_error(output: vf.RolloutOutput) -> str | None:
    error = output.get("error")
    if not error:
        return None
    if isinstance(error, dict):
        return str(error.get("error_chain_str") or error.get("error_chain_repr") or error)
    return str(error)


def _is_group_scoring(env: vf.Environment) -> bool:
    return any(env.rubric._is_group_func(func) for func in env.rubric._get_reward_funcs())


def _active_feature_fraction(encoded_path: Path, threshold: float) -> float:
    rows = pq.read_table(encoded_path).to_pylist()
    steered = [row for row in rows if row.get("coordinate") is not None]
    if not steered:
        return 0.0
    prompts = sorted({str(row["prompt_id"]).split(":", 1)[0] for row in steered})
    variances = []
    for prompt in prompts:
        prompt_rows = [row for row in steered if str(row["prompt_id"]).split(":", 1)[0] == prompt]
        if len(prompt_rows) >= 2:
            variances.append(np.asarray([row["behavior_features"] for row in prompt_rows], dtype=np.float32).var(axis=0))
    if not variances:
        return 0.0
    mean_var = np.asarray(variances).mean(axis=0)
    return float(np.mean(mean_var > threshold))


def _jobs(config: dict[str, Any], examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    job_id = 0
    excluded = _excluded_cell_keys(config)
    for prompt_idx, example in enumerate(examples[: int(config["n_prompts"])]):
        prompt_id = str(example.get("example_id", example.get("id", prompt_idx)))
        for seed in config["seeds"]:
            for controller in config["controllers"]:
                for coordinate in config["coordinates"]:
                    for sign in config["signs"]:
                        if f"{controller}|k={int(coordinate)}|sign={sign}" in excluded:
                            continue
                        jobs.append(
                            {
                                "job_id": f"job_{job_id:06d}",
                                "prompt_idx": prompt_idx,
                                "prompt_id": prompt_id,
                                "example": example,
                                "seed": int(seed),
                                "controller": controller,
                                "coordinate": int(coordinate),
                                "sign": str(sign),
                            }
                        )
                        job_id += 1
    return jobs


def _write_summary_and_audit(
    run_dir: Path,
    summary: dict[str, Any],
    *,
    exit_on_fail: bool = True,
) -> dict[str, Any]:
    _write_json(run_dir / "summary.json", summary)
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": summary["timestamp_utc"],
            "stage": summary["stage"],
            "run_id": summary["run_id"],
            "gate_status": summary["gate"],
            "key_metrics": json.dumps(summary["metrics"], sort_keys=True),
            "evidence_path": f"runs/{summary['run_id']}/summary.json",
            "prime_rl_modifications": "none",
        },
    )
    if exit_on_fail and summary["gate"] != "pass":
        raise SystemExit(1)
    return summary


def _run_completeness_gate_self_test(config_path: Path, config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if not config.get("run_completeness_gate_self_test", False):
        return {"enabled": False}
    first_cell = {
        "controller": config["required_controllers"][0],
        "coordinate": int(config["required_coordinates"][0]),
        "sign": config["required_signs"][0],
    }
    negative = dict(config)
    negative["run_id"] = f"{config['run_id']}_completeness_negative"
    negative["run_completeness_gate_self_test"] = False
    negative["dry_run_completeness_only"] = True
    negative["excluded_cells"] = [first_cell]
    negative_path = run_dir / "completeness_negative_config.yaml"
    negative_path.write_text(yaml.safe_dump(negative, sort_keys=False))
    completed = subprocess.run(
        [sys.executable, "-m", "controllability.experiments.phase1_5_verifier_smoke", "--config", str(negative_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    result = {
        "enabled": True,
        "negative_config_path": str(negative_path),
        "removed_cell": first_cell,
        "exit_code": int(completed.returncode),
        "nonzero_exit_verified": completed.returncode != 0,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "summary_path": f"runs/{negative['run_id']}/summary.json",
    }
    _write_json(run_dir / "completeness_gate_self_test.json", result)
    return result


def _aggregate_linearization(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    if not diagnostics:
        return {
            "num_turn_diagnostics": 0,
            "median_turn_linearization_cosine": 0.0,
            "median_turn_linearization_r2": 0.0,
            "turn_linearization_cosine_below_0_7": True,
        }
    cosines = np.asarray([float(item.get("linearization_cosine", 0.0)) for item in diagnostics], dtype=np.float32)
    r2s = np.asarray([float(item.get("linearization_r2", 0.0)) for item in diagnostics], dtype=np.float32)
    return {
        "num_turn_diagnostics": int(len(diagnostics)),
        "median_turn_linearization_cosine": float(np.median(cosines)),
        "median_turn_linearization_r2": float(np.median(r2s)),
        "turn_linearization_cosine_below_0_7": bool(float(np.median(cosines)) < 0.7),
        "turn_linearization_cosine_p10": float(np.quantile(cosines, 0.1)),
        "turn_linearization_r2_p10": float(np.quantile(r2s, 0.1)),
    }


def _is_phase2_run(config: dict[str, Any]) -> bool:
    return str(config.get("summary_stage", "")).startswith("phase2")


def run_smoke(
    config_path: Path,
    *,
    job_shard_index: int | None = None,
    num_job_shards: int | None = None,
    run_id_suffix: str | None = None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if job_shard_index is not None:
        config["job_shard_index"] = int(job_shard_index)
    if num_job_shards is not None:
        config["num_job_shards"] = int(num_job_shards)
    run_id = config.get("run_id") or f"phase1_5_verifier_smoke_{_safe_id(config['env_id'])}_{_utc_now().replace(':', '')}"
    if run_id_suffix:
        run_id = f"{run_id}{run_id_suffix}"
        config["run_id"] = run_id
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    git_hash = _git_hash()

    completeness_configured = _configured_completeness_report(config)
    completeness_gate_test = _run_completeness_gate_self_test(config_path, {**config, "run_id": run_id}, run_dir)
    if config.get("require_complete_steering_grid", False) and not completeness_configured["configured_complete"]:
        metrics = {
            "configured_completeness": completeness_configured,
            "completeness_gate_self_test": completeness_gate_test,
        }
        summary = {
            "stage": str(config.get("summary_stage", "phase1_5_verifier_smoke")),
            "run_id": run_id,
            "timestamp_utc": _utc_now(),
            "git_hash": git_hash,
            "prime_rl_git_hash": git_hash,
            "prime_rl_modifications": [],
            "config": config,
            "metrics": metrics,
            "gate": "fail",
            "gate_reason": "Configured controller/sign/coordinate grid is incomplete; hard fail before model loading",
            "next_stage_inputs": {},
        }
        return _write_summary_and_audit(run_dir, summary)
    if config.get("dry_run_completeness_only", False):
        metrics = {
            "configured_completeness": completeness_configured,
            "completeness_gate_self_test": completeness_gate_test,
        }
        summary = {
            "stage": str(config.get("summary_stage", "phase1_5_verifier_smoke")),
            "run_id": run_id,
            "timestamp_utc": _utc_now(),
            "git_hash": git_hash,
            "prime_rl_git_hash": git_hash,
            "prime_rl_modifications": [],
            "config": config,
            "metrics": metrics,
            "gate": "pass",
            "gate_reason": "Completeness-only dry run passed",
            "next_stage_inputs": {},
        }
        return _write_summary_and_audit(run_dir, summary)

    env = vf.load_environment(config["env_id"])
    examples = env.get_eval_dataset(n=int(config["n_prompts"])).to_list()
    all_jobs = _jobs(config, examples)
    shard_index = config.get("job_shard_index")
    num_shards = config.get("num_job_shards")
    if shard_index is not None or num_shards is not None:
        if shard_index is None or num_shards is None:
            raise ValueError("job_shard_index and num_job_shards must be set together")
        shard_index = int(shard_index)
        num_shards = int(num_shards)
        if num_shards < 1:
            raise ValueError("num_job_shards must be positive")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError("job_shard_index must be in [0, num_job_shards)")
        jobs = [job for index, job in enumerate(all_jobs) if index % num_shards == shard_index]
    else:
        jobs = all_jobs
    stage3_rows = _load_json(config["stage3_aggregated_path"])["rows"]
    eta_by_coordinate = {
        int(row["coordinate"]): float(row["eta_star_mean"])
        for row in stage3_rows
        if row["formulation"] == config["formulation"]
    }
    frozen = FrozenModel(
        config["model_id"],
        dtype=str(config.get("dtype", "bfloat16")),
        device_map=config.get("device_map", "auto"),
    )
    basis = _fit_stage4_basis(frozen, config)
    sampling_args = {
        "temperature": float(config.get("temperature", 0.0)),
        "max_completion_tokens": int(config.get("max_completion_tokens", 512)),
        "seed": int(config.get("seed", 42)),
    }
    for key in ("top_p", "top_k", "min_p", "repetition_penalty"):
        if key in config:
            sampling_args[key] = config[key]
    trajectories = []
    raw_rollouts = []
    failed = []
    linearization_diagnostics: list[dict[str, Any]] = []
    max_rollout_retries = int(config.get("max_rollout_retries", 0))

    for job in jobs:
        torch.manual_seed(int(job["seed"]))
        edit_norm = float(config.get("eta_multiplier", 1.0)) * eta_by_coordinate[int(job["coordinate"])]
        started = time.perf_counter()
        client = None
        try:
            output = None
            last_error: str | None = None
            for attempt in range(max_rollout_retries + 1):
                client = HFSteeredChatClient(
                    frozen,
                    basis=basis,
                    patch_layer=int(config["patch_layer"]),
                    readout_layer=int(config["readout_layer"]),
                    coordinate=int(job["coordinate"]),
                    sign=str(job["sign"]),
                    controller=str(job["controller"]),
                    edit_norm=edit_norm,
                    eta_star=eta_by_coordinate[int(job["coordinate"])],
                    linearization_probe_norm=float(config.get("linearization_probe_norm", 1e-3)),
                    linearization_diagnostic_scope=str(config.get("linearization_diagnostic_scope", "turn")),
                    max_prefix_tokens=int(config.get("max_prefix_tokens", 512)),
                    closed_loop_chunk_tokens=int(config.get("closed_loop_chunk_tokens", 8)),
                    closed_loop_budget_mode=str(config.get("closed_loop_budget_mode", "l2_per_turn")),
                    enable_thinking=bool(config.get("enable_thinking", False)),
                    generate_use_cache=bool(config.get("generate_use_cache", True)),
                )
                if _is_group_scoring(env):
                    raise NotImplementedError("Phase 1.5 smoke does not yet support group-scored verifier environments")
                output = asyncio.run(
                    env.run_rollout(
                        _rollout_input_from_example(job["example"]),
                        client=client,
                        model=config["model_id"],
                        sampling_args=sampling_args,
                        max_retries=int(config.get("max_retries", 0)),
                        state_columns=["trajectory", "sampling_args"],
                    )
                )
                last_error = _rollout_error(output)
                if last_error is None:
                    break
                _clear_cuda_cache()
            if output is None:
                raise RuntimeError("rollout produced no output")
            if last_error is not None:
                raise RuntimeError(last_error)
            raw_rollouts.append(dict(output))
            traj = _trajectory_from_vf_output(
                output,
                env_id=config["env_id"],
                model_id=config["model_id"],
                prompt_id=f"{job['prompt_id']}:{job['job_id']}",
                seed=int(job["seed"]),
                wall_clock_seconds=time.perf_counter() - started,
                git_hash=git_hash,
            )
            trajectories.append(
                replace(
                    traj,
                    steering=SteeringMetadata(
                        controller=job["controller"],
                        coordinate=int(job["coordinate"]),
                        sign=job["sign"],
                        edit_norm=edit_norm,
                        trust_region_violations=sum(1 for item in client.diagnostics if item.get("trust_region_violation")),
                        applied_edits=client.diagnostics,
                    ),
                )
            )
            for item in client.diagnostics:
                linearization_diagnostics.append(
                    {
                        **item,
                        "job_id": job["job_id"],
                        "prompt_id": job["prompt_id"],
                        "seed": int(job["seed"]),
                    }
                )
            _append_jsonl(run_dir / "raw_rollouts.jsonl", dict(output))
            _append_jsonl(run_dir / "trajectories.jsonl", trajectory_to_record(trajectories[-1]))
        except Exception as exc:
            failed.append(
                {
                    "job_id": job["job_id"],
                    "prompt_id": job["prompt_id"],
                    "controller": job["controller"],
                    "coordinate": job["coordinate"],
                    "sign": job["sign"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _append_jsonl(run_dir / "failed_jobs.jsonl", failed[-1])
        finally:
            client = None
            if bool(config.get("empty_cache_between_jobs", True)):
                _clear_cuda_cache()

    encoded_records = _write_encoded(
        run_dir,
        trajectories,
        encoder_name=str(config.get("encoder_name", "sentence_e5_structured_v1")),
        encoder_config=config.get("encoder_config") or {},
    ) if trajectories else []
    if not encoded_records:
        pq.write_table(pa.table({}), run_dir / "encoded.parquet")
    tool_call_count = sum(len(traj.tool_calls) for traj in trajectories)
    rewards = [traj.reward for traj in trajectories]
    generated_tokens = [
        sum(
            int((step.get("usage") or {}).get("completion_tokens", 0))
            for step in traj.step_metadata
            if isinstance(step, dict)
        )
        for traj in trajectories
    ]
    active_fraction = _active_feature_fraction(run_dir / "encoded.parquet", float(config.get("encoder_variance_threshold", 1e-8))) if encoded_records else 0.0
    _write_json(run_dir / "turn_linearization_diagnostics.json", {"rows": linearization_diagnostics})
    linearization_metrics = _aggregate_linearization(linearization_diagnostics)
    smoke_checks = {
        "has_trajectories": len(trajectories) > 0,
        "has_tool_calls": (not bool(config.get("require_tool_calls", True))) or tool_call_count > 0,
        "has_nonzero_quality": any(value != 0.0 for value in rewards),
        "agent_length_tokens": bool(generated_tokens) and float(np.mean(generated_tokens)) >= float(config.get("agent_length_token_floor", 128)),
        "encoder_active_fraction_pass": active_fraction > float(config.get("encoder_active_fraction_floor", 0.5)),
        "completeness_gate_self_test_pass": bool(completeness_gate_test.get("nonzero_exit_verified", not config.get("run_completeness_gate_self_test", False))),
        "configured_grid_complete": bool(completeness_configured["configured_complete"]),
        "turn_linearization_cosine_pass": not linearization_metrics["turn_linearization_cosine_below_0_7"],
    }
    if _is_phase2_run(config):
        phase2_required_checks = {
            key: smoke_checks[key]
            for key in (
                "has_trajectories",
                "has_tool_calls",
                "encoder_active_fraction_pass",
                "completeness_gate_self_test_pass",
                "configured_grid_complete",
                "turn_linearization_cosine_pass",
            )
        }
        gate = "pass" if all(phase2_required_checks.values()) and not failed else "fail"
    else:
        phase2_required_checks = {}
        gate = "pass" if all(smoke_checks.values()) and not failed else "fail"
    metrics = {
        "num_jobs": len(jobs),
        "num_total_jobs_unsharded": len(all_jobs),
        "job_shard_index": int(shard_index) if shard_index is not None else None,
        "num_job_shards": int(num_shards) if num_shards is not None else None,
        "num_trajectories": len(trajectories),
        "num_failed": len(failed),
        "tool_call_count": tool_call_count,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "nonzero_reward_count": sum(1 for value in rewards if value != 0.0),
        "mean_generated_tokens": float(np.mean(generated_tokens)) if generated_tokens else 0.0,
        "encoder_active_feature_fraction": active_fraction,
        "configured_completeness": completeness_configured,
        "completeness_gate_self_test": completeness_gate_test,
        **linearization_metrics,
        "smoke_checks": smoke_checks,
        "phase2_required_checks": phase2_required_checks,
    }
    summary = {
        "stage": str(config.get("summary_stage", "phase1_5_verifier_smoke")),
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": git_hash,
        "prime_rl_git_hash": git_hash,
        "prime_rl_modifications": [],
        "config": config,
        "metrics": metrics,
        "gate": gate,
        "gate_reason": (
            "Phase 2 verifier-scale required checks passed"
            if gate == "pass" and _is_phase2_run(config)
            else "Phase 1.5 verifier-loop smoke criteria passed"
            if gate == "pass"
            else "Phase 2 verifier-scale required checks failed"
            if _is_phase2_run(config)
            else "Phase 1.5 verifier-loop smoke criteria failed; do not start Phase 2"
        ),
        "limitations": [],
        "next_stage_inputs": {
            "encoded": f"runs/{run_id}/encoded.parquet",
            "trajectories": f"runs/{run_id}/trajectories.jsonl",
            "raw_rollouts": f"runs/{run_id}/raw_rollouts.jsonl",
            "failed_jobs": f"runs/{run_id}/failed_jobs.jsonl",
            "turn_linearization_diagnostics": f"runs/{run_id}/turn_linearization_diagnostics.json",
        },
    }
    return _write_summary_and_audit(run_dir, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--job-shard-index", type=int, default=None)
    parser.add_argument("--num-job-shards", type=int, default=None)
    parser.add_argument("--run-id-suffix", default=None)
    args = parser.parse_args()
    run_smoke(
        args.config,
        job_shard_index=args.job_shard_index,
        num_job_shards=args.num_job_shards,
        run_id_suffix=args.run_id_suffix,
    )


if __name__ == "__main__":
    main()
