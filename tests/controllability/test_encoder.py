from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from controllability.encoders.encoder_registry import get_encoder
from controllability.envs.prime_rl_adapter import trajectory_from_vf_output
from controllability.envs.trajectory_schema import Trajectory, none_steering_metadata


def _synthetic_trajectory() -> Trajectory:
    return Trajectory(
        env_id="primeintellect/math500",
        model_id="Qwen/Qwen3-8B",
        prompt_id="0",
        seed=0,
        messages=[{"role": "user", "content": "Solve 1+1."}, {"role": "assistant", "content": "Because 1+1=2."}],
        tool_calls=[],
        step_metadata=[{"step_index": 0}],
        reward=1.0,
        success=True,
        steering=none_steering_metadata(),
        git_hash="abc123",
        prime_rl_git_hash="def456",
        timestamp_utc="2026-05-07T00:00:00Z",
        wall_clock_seconds=1.0,
    )


def test_fixed_feature_encoder_returns_declared_shape():
    traj = _synthetic_trajectory()
    encoder = get_encoder(traj.env_id)
    encoded = encoder.encode(traj)

    assert encoded.behavior_features.shape == (encoder.feature_dim,)
    assert encoded.behavior_features.dtype == np.float32
    assert encoded.trajectory_id == traj.trajectory_id


def test_encoder_runs_on_preflight_smoke_trajectory_if_present():
    smoke_path = Path("runs/preflight/smoke_rollout_primeintellect_math500.json")
    if smoke_path.exists():
        payload = json.loads(smoke_path.read_text())
        traj = trajectory_from_vf_output(
            payload["rollout"],
            env_id=payload["env_id"],
            model_id=payload["model_id"],
            prompt_id=str(payload["rollout"].get("example_id", 0)),
            seed=payload["seed"],
            wall_clock_seconds=payload["wall_clock_seconds"],
        )
    else:
        traj = _synthetic_trajectory()

    encoder = get_encoder(traj.env_id)
    encoded = encoder.encode(traj)
    assert encoded.behavior_features.shape == (encoder.feature_dim,)
