from __future__ import annotations

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

from controllability.envs.rollout_collector import SamplingConfig, SteeringSpec
from controllability.envs.trajectory_schema import Trajectory, none_steering_metadata
from prime_rl.configs.orchestrator import EvalEnvConfig, EvalSamplingConfig
from prime_rl.orchestrator.envs import EvalEnv


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_inference_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        if config_path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        return json.load(f)


def _messages_from_rollout(output: vf.RolloutOutput) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    prompt = output.get("prompt")
    if isinstance(prompt, list):
        messages.extend(prompt)
    for step in output.get("trajectory") or []:
        for key in ("prompt", "completion"):
            value = step.get(key)
            if isinstance(value, list):
                messages.extend(value)
    completion = output.get("completion")
    if isinstance(completion, list):
        messages.extend(completion)
    elif isinstance(completion, str) and completion:
        messages.append({"role": "assistant", "content": completion})
    return messages


def _tool_calls_from_rollout(output: vf.RolloutOutput) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for step in output.get("trajectory") or []:
        for message in step.get("completion") or []:
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            if calls:
                tool_calls.extend(calls)
        extras = step.get("extras") or {}
        if isinstance(extras, dict) and extras.get("tool_calls"):
            tool_calls.extend(extras["tool_calls"])
    return tool_calls


def _step_metadata_from_rollout(output: vf.RolloutOutput) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for idx, step in enumerate(output.get("trajectory") or []):
        response = step.get("response") or {}
        metadata.append(
            {
                "step_index": idx,
                "reward": step.get("reward"),
                "is_truncated": step.get("is_truncated"),
                "trajectory_id": step.get("trajectory_id"),
                "usage": response.get("usage") if isinstance(response, dict) else None,
                "finish_reason": response.get("finish_reason") if isinstance(response, dict) else None,
            }
        )
    return metadata


def trajectory_from_vf_output(
    output: vf.RolloutOutput,
    *,
    env_id: str,
    model_id: str,
    prompt_id: str,
    seed: int,
    wall_clock_seconds: float,
    git_hash: str | None = None,
    prime_rl_git_hash: str | None = None,
    timestamp_utc: str | None = None,
) -> Trajectory:
    reward = float(output.get("reward") or 0.0)
    success = bool(output.get("solved", output.get("success", reward > 0.0)))
    resolved_git_hash = git_hash or _git_hash()
    return Trajectory(
        env_id=env_id,
        model_id=model_id,
        prompt_id=prompt_id,
        seed=seed,
        messages=_messages_from_rollout(output),
        tool_calls=_tool_calls_from_rollout(output),
        step_metadata=_step_metadata_from_rollout(output),
        reward=reward,
        success=success,
        steering=none_steering_metadata(),
        git_hash=resolved_git_hash,
        prime_rl_git_hash=prime_rl_git_hash or resolved_git_hash,
        timestamp_utc=timestamp_utc or _utc_now(),
        wall_clock_seconds=wall_clock_seconds,
    )


class PrimeRLRolloutClient:
    """
    Wrapper around prime-rl's EvalEnv/verifiers rollout path.

    Base-policy rollouts use prime-rl's OpenAI-compatible inference path, normally
    backed by vLLM. Residual steering and capture are handled by FrozenModel in the
    HF backend; the vLLM adapter intentionally does not monkeypatch prime-rl.
    """

    def __init__(self, model_id: str, inference_config_path: str | Path | None = None):
        self.model_id = model_id
        self.config = _load_inference_config(inference_config_path)

    def rollout_one(self, env_id: str, prompt_id: str, seed: int, sampling: SamplingConfig) -> Trajectory:
        return asyncio.run(self._rollout_one(env_id, prompt_id, seed, sampling))

    async def _rollout_one(self, env_id: str, prompt_id: str, seed: int, sampling: SamplingConfig) -> Trajectory:
        eval_sampling = EvalSamplingConfig(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
            max_completion_tokens=sampling.max_completion_tokens,
            min_tokens=sampling.min_tokens,
            seed=seed if sampling.seed is None else sampling.seed,
            extra_body=sampling.extra_body,
        )
        env_config = EvalEnvConfig(
            id=env_id,
            num_examples=-1,
            rollouts_per_example=1,
            num_workers=int(self.config.get("num_workers", 1)),
            max_retries=int(self.config.get("max_retries", 0)),
            timeout=self.config.get("timeout_seconds"),
            max_total_completion_tokens=sampling.max_completion_tokens or -1,
            sampling=eval_sampling,
        )
        env = EvalEnv(env_config)
        example = self._select_example(env.examples, prompt_id)
        client = vf.ClientConfig(
            client_type=self.config.get("client_type", "openai_chat_completions"),
            api_base_url=self.config.get("base_url", "http://127.0.0.1:8000/v1"),
            api_key_var=self.config.get("api_key_var", "VLLM_API_KEY"),
            timeout=float(self.config.get("timeout_seconds", 600.0)),
            connect_timeout=float(self.config.get("connect_timeout", 30.0)),
            max_retries=int(self.config.get("client_max_retries", 0)),
        )

        log_dir = Path(self.config.get("env_server_log_dir", "runs/env_server_logs"))
        started = time.perf_counter()
        await env.start(log_dir=log_dir, log_level=self.config.get("log_level", "INFO"))
        try:
            if env.requires_group_scoring:
                outputs = await env.run_group(
                    client=client,
                    example=example,
                    model_name=self.model_id,
                    rollouts_per_example=1,
                    cache_salt=f"controllability-{env_id}-{prompt_id}-{seed}",
                )
                output = outputs[0]
            else:
                output = await env.run_rollout(
                    client=client,
                    example=example,
                    model_name=self.model_id,
                    cache_salt=f"controllability-{env_id}-{prompt_id}-{seed}",
                )
        finally:
            env.shutdown()

        _ = json.dumps(output, default=make_serializable)
        return trajectory_from_vf_output(
            output,
            env_id=env_id,
            model_id=self.model_id,
            prompt_id=prompt_id,
            seed=seed,
            wall_clock_seconds=time.perf_counter() - started,
        )

    def rollout_with_residual_capture(
        self,
        env_id: str,
        prompt_id: str,
        seed: int,
        sampling: SamplingConfig,
        capture_layers: list[int],
    ) -> tuple[Trajectory, dict[int, Any]]:
        raise NotImplementedError(
            "Residual capture is not available through prime-rl's vLLM rollout path. "
            "Use controllability.models.FrozenModel for HF-backed Jacobian and capture work."
        )

    def rollout_with_steering(
        self,
        env_id: str,
        prompt_id: str,
        seed: int,
        sampling: SamplingConfig,
        steering: SteeringSpec,
    ) -> Trajectory:
        if steering.controller == "none":
            return self.rollout_one(env_id=env_id, prompt_id=prompt_id, seed=seed, sampling=sampling)
        raise NotImplementedError(
            "Steered environment rollouts require the HF backend and closed-loop controller, "
            "not the base-policy vLLM adapter."
        )

    @staticmethod
    def _select_example(examples: list[dict[str, Any]], prompt_id: str) -> dict[str, Any]:
        if prompt_id.isdigit():
            idx = int(prompt_id)
            return examples[idx]
        for example in examples:
            if str(example.get("id", example.get("example_id", ""))) == prompt_id:
                return example
        raise KeyError(f"Prompt {prompt_id!r} not found in environment examples")
