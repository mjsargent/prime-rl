from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import verifiers as vf
import yaml

from controllability.envs.prime_rl_adapter import PrimeRLRolloutClient
from controllability.envs.rollout_collector import SamplingConfig, write_manifest, write_rollout_outputs, write_summary
from controllability.envs.trajectory_schema import Trajectory


class EnvironmentNotInstalled(RuntimeError):
    """Raised when a Hub environment cannot be loaded by verifiers."""


def load_environment(env_id: str) -> vf.Environment:
    try:
        return vf.load_environment(env_id)
    except Exception as exc:
        raise EnvironmentNotInstalled(
            f"Unable to load {env_id!r}. Run `prime env install {env_id}` and retry."
        ) from exc


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _registered_env_ids() -> set[str]:
    path = Path("configs/controllability/preregistration/environment_list.yaml")
    if not path.exists():
        return set()
    config = _load_yaml(path)
    env_ids: set[str] = set()
    for section in ("tier_a", "tier_b"):
        for value in (config.get(section) or {}).values():
            if isinstance(value, dict) and value.get("env_id"):
                env_ids.add(value["env_id"])
    return env_ids


def _registered_model_ids() -> set[str]:
    path = Path("configs/controllability/preregistration/model_list.yaml")
    if not path.exists():
        return set()
    config = _load_yaml(path)
    model_ids: set[str] = set()
    for value in (config.get("models") or {}).values():
        if isinstance(value, dict) and value.get("resolved_model_id") and "pass" in value.get("status", ""):
            model_ids.add(value["resolved_model_id"])
    return model_ids


def _validate_preregistration(env_id: str, model_id: str) -> None:
    env_ids = _registered_env_ids()
    if env_ids and env_id not in env_ids:
        raise ValueError(f"{env_id!r} is not listed in configs/controllability/preregistration/environment_list.yaml")
    model_ids = _registered_model_ids()
    if model_ids and model_id not in model_ids:
        raise ValueError(f"{model_id!r} is not a passing resolved model in preregistration/model_list.yaml")


def collect_rollouts(
    env_id: str,
    model_id: str,
    n_prompts: int,
    n_seeds_per_prompt: int,
    sampling: SamplingConfig,
    output_dir: Path,
    backend: Literal["vllm", "hf"] = "vllm",
) -> list[Trajectory]:
    _validate_preregistration(env_id, model_id)
    if backend != "vllm":
        raise NotImplementedError("HF environment rollouts are reserved for steered Stage 4 controllers.")

    output_dir.mkdir(parents=True, exist_ok=True)
    client = PrimeRLRolloutClient(model_id=model_id, inference_config_path=None)
    trajectories: list[Trajectory] = []
    failed: list[dict] = []
    for prompt_idx in range(n_prompts):
        prompt_id = str(prompt_idx)
        for seed in range(n_seeds_per_prompt):
            try:
                trajectories.append(client.rollout_one(env_id, prompt_id, seed, sampling))
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

    write_rollout_outputs(output_dir, trajectories, failed)
    manifest = {
        "env_id": env_id,
        "model_id": model_id,
        "n_prompts": n_prompts,
        "n_seeds_per_prompt": n_seeds_per_prompt,
        "backend": backend,
        "sampling": sampling.to_sampling_args(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_manifest(output_dir, manifest)
    write_summary(
        output_dir,
        {
            **manifest,
            "num_trajectories": len(trajectories),
            "num_failed": len(failed),
            "reward_mean": sum(traj.reward for traj in trajectories) / len(trajectories) if trajectories else 0.0,
            "success_rate": sum(traj.success for traj in trajectories) / len(trajectories) if trajectories else 0.0,
        },
    )
    return trajectories
