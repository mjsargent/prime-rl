from __future__ import annotations

from dataclasses import replace

from controllability.envs.trajectory_schema import (
    SteeringMetadata,
    Trajectory,
    read_jsonl,
    read_parquet,
    write_jsonl,
    write_parquet,
)


def _trajectory() -> Trajectory:
    return Trajectory(
        env_id="primeintellect/math500",
        model_id="Qwen/Qwen3-8B",
        prompt_id="0",
        seed=7,
        messages=[{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
        tool_calls=[{"name": "calculator", "arguments": {"expr": "2+2"}}],
        step_metadata=[{"step_index": 0, "reward": 1.0}],
        reward=1.0,
        success=True,
        steering=SteeringMetadata(
            controller="one_shot",
            coordinate=2,
            sign="+",
            edit_norm=0.5,
            trust_region_violations=0,
            applied_edits=[{"layer": 12, "norm": 0.5}],
        ),
        git_hash="abc123",
        prime_rl_git_hash="def456",
        timestamp_utc="2026-05-07T00:00:00Z",
        wall_clock_seconds=3.25,
    )


def test_trajectory_round_trips_jsonl_and_parquet(tmp_path):
    traj = _trajectory()
    trajectories = [traj, replace(traj, seed=8, steering=None)]

    jsonl_path = tmp_path / "trajectories.jsonl"
    parquet_path = tmp_path / "trajectories.parquet"
    write_jsonl(jsonl_path, trajectories)
    write_parquet(parquet_path, trajectories)

    assert read_jsonl(jsonl_path) == trajectories
    assert read_parquet(parquet_path) == trajectories
