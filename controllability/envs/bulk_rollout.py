from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import verifiers as vf
import yaml
from verifiers.utils.save_utils import make_serializable

from controllability.envs.prime_rl_adapter import trajectory_from_vf_output
from controllability.envs.rollout_collector import SamplingConfig, write_failed_rows, write_manifest, write_summary
from controllability.envs.trajectory_schema import Trajectory, write_jsonl
from prime_rl.configs.orchestrator import EvalEnvConfig, EvalSamplingConfig
from prime_rl.orchestrator.envs import REQUIRED_STATE_COLUMNS, EvalEnv


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _sampling_args(sampling: SamplingConfig, seed: int, cache_salt: str) -> dict[str, Any]:
    args = sampling.to_sampling_args()
    args["seed"] = seed
    extra_body = dict(args.get("extra_body", {}))
    extra_body["cache_salt"] = cache_salt
    args["extra_body"] = extra_body
    return args


async def collect_from_config(config: dict[str, Any]) -> None:
    env_id = config["env_id"]
    model_id = config["model_id"]
    n_prompts = int(config["n_prompts"])
    n_seeds_per_prompt = int(config["n_seeds_per_prompt"])
    output_dir = Path(config.get("output_dir") or f"runs/rollouts/{_safe_name(env_id)}_{_safe_name(model_id)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampling = SamplingConfig(**(config.get("sampling") or {}))
    timeout_seconds = float(config.get("timeout_seconds", 600.0))
    concurrency = int(config.get("concurrency", 64))
    env_workers = int(config.get("env_workers", 4))
    git_hash = _git_hash()
    timestamp = _utc_now()

    eval_sampling = EvalSamplingConfig(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        max_completion_tokens=sampling.max_completion_tokens,
        min_tokens=sampling.min_tokens,
        extra_body=sampling.extra_body,
    )
    env_config = EvalEnvConfig(
        id=env_id,
        num_examples=n_prompts,
        rollouts_per_example=1,
        num_workers=env_workers,
        max_retries=int(config.get("max_retries", 0)),
        timeout=timeout_seconds,
        max_total_completion_tokens=sampling.max_completion_tokens or -1,
        sampling=eval_sampling,
    )
    env = EvalEnv(env_config)
    if len(env.examples) < n_prompts:
        raise ValueError(f"{env_id} provided {len(env.examples)} examples, fewer than requested n_prompts={n_prompts}")

    client = vf.ClientConfig(
        client_type=config.get("client_type", "openai_chat_completions"),
        api_base_url=config.get("base_url", "http://127.0.0.1:8000/v1"),
        api_key_var=config.get("api_key_var", "VLLM_API_KEY"),
        timeout=timeout_seconds,
        connect_timeout=float(config.get("connect_timeout", 30.0)),
        max_retries=int(config.get("client_max_retries", 0)),
    )
    semaphore = asyncio.Semaphore(concurrency)
    trajectories: list[Trajectory] = []
    raw_rollouts: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    async def run_cell(prompt_idx: int, seed: int) -> None:
        example = env.examples[prompt_idx]
        prompt_id = str(example.get("example_id", prompt_idx))
        started = time.perf_counter()
        cache_salt = f"stage1-{_safe_name(env_id)}-{_safe_name(model_id)}-{prompt_id}-{seed}"
        try:
            async with semaphore:
                if env.requires_group_scoring:
                    outputs = await env.env.run_group(
                        [vf.RolloutInput(**example)],
                        client=client,
                        model=model_id,
                        sampling_args=_sampling_args(sampling, seed, cache_salt),
                        max_retries=env.config.max_retries,
                        state_columns=REQUIRED_STATE_COLUMNS,
                        env_client=env.env_client,
                    )
                    output = outputs[0]
                else:
                    output = await env.env.run_rollout(
                        vf.RolloutInput(**example),
                        client=client,
                        model=model_id,
                        sampling_args=_sampling_args(sampling, seed, cache_salt),
                        max_retries=env.config.max_retries,
                        state_columns=REQUIRED_STATE_COLUMNS,
                        env_client=env.env_client,
                    )
            raw_rollouts.append(dict(output))
            trajectories.append(
                trajectory_from_vf_output(
                    output,
                    env_id=env_id,
                    model_id=model_id,
                    prompt_id=prompt_id,
                    seed=seed,
                    wall_clock_seconds=time.perf_counter() - started,
                    git_hash=git_hash,
                    prime_rl_git_hash=git_hash,
                    timestamp_utc=timestamp,
                )
            )
        except Exception as exc:
            failed.append(
                {
                    "env_id": env_id,
                    "model_id": model_id,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    manifest = {
        "env_id": env_id,
        "model_id": model_id,
        "n_prompts": n_prompts,
        "n_seeds_per_prompt": n_seeds_per_prompt,
        "expected_cells": n_prompts * n_seeds_per_prompt,
        "sampling": sampling.to_sampling_args(),
        "base_url": client.api_base_url,
        "concurrency": concurrency,
        "env_workers": env_workers,
        "git_hash": git_hash,
        "timestamp_utc": timestamp,
    }
    write_manifest(output_dir, manifest)

    await env.start(log_dir=Path(config.get("env_server_log_dir", "runs/env_server_logs")), log_level="INFO")
    try:
        tasks = [run_cell(prompt_idx, seed) for prompt_idx in range(n_prompts) for seed in range(n_seeds_per_prompt)]
        for idx in range(0, len(tasks), int(config.get("checkpoint_interval", 100))):
            await asyncio.gather(*tasks[idx : idx + int(config.get("checkpoint_interval", 100))])
            write_jsonl(output_dir / "trajectories.jsonl", trajectories)
            write_failed_rows(output_dir / "failed.jsonl", failed)
    finally:
        env.shutdown()

    with (output_dir / "raw_rollouts.jsonl").open("w") as f:
        for rollout in raw_rollouts:
            json.dump(rollout, f, default=make_serializable, sort_keys=True)
            f.write("\n")

    write_summary(
        output_dir,
        {
            **manifest,
            "num_trajectories": len(trajectories),
            "num_failed": len(failed),
            "gate": "pass" if len(trajectories) + len(failed) == n_prompts * n_seeds_per_prompt else "fail",
            "reward_mean": sum(traj.reward for traj in trajectories) / len(trajectories) if trajectories else 0.0,
            "success_rate": sum(traj.success for traj in trajectories) / len(trajectories) if trajectories else 0.0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(collect_from_config(_load_config(args.config)))


if __name__ == "__main__":
    main()
