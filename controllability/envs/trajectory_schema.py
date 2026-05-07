from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class SteeringMetadata:
    controller: Literal["none", "one_shot", "open_loop", "closed_loop"]
    coordinate: int | None
    sign: Literal["+", "-"] | None
    edit_norm: float
    trust_region_violations: int
    applied_edits: list[dict[str, Any]]


@dataclass(frozen=True)
class Trajectory:
    env_id: str
    model_id: str
    prompt_id: str
    seed: int
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    step_metadata: list[dict[str, Any]]
    reward: float
    success: bool
    steering: SteeringMetadata | None
    git_hash: str
    prime_rl_git_hash: str
    timestamp_utc: str
    wall_clock_seconds: float

    @property
    def trajectory_id(self) -> str:
        steering = self.steering.controller if self.steering else "none"
        return f"{self.env_id}:{self.model_id}:{self.prompt_id}:{self.seed}:{steering}"


def trajectory_to_record(traj: Trajectory) -> dict[str, Any]:
    record = dataclasses.asdict(traj)
    record["trajectory_id"] = traj.trajectory_id
    return record


def trajectory_from_record(record: dict[str, Any]) -> Trajectory:
    payload = dict(record)
    payload.pop("trajectory_id", None)
    steering = payload.get("steering")
    if steering is not None and not isinstance(steering, SteeringMetadata):
        payload["steering"] = SteeringMetadata(**steering)
    return Trajectory(**payload)


def write_jsonl(path: Path, trajectories: list[Trajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for traj in trajectories:
            json.dump(trajectory_to_record(traj), f, sort_keys=True)
            f.write("\n")


def read_jsonl(path: Path) -> list[Trajectory]:
    with path.open() as f:
        return [trajectory_from_record(json.loads(line)) for line in f if line.strip()]


def write_parquet(path: Path, trajectories: list[Trajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [trajectory_to_record(traj) for traj in trajectories]
    pq.write_table(pa.Table.from_pylist(records), path)


def read_parquet(path: Path) -> list[Trajectory]:
    records = pq.read_table(path).to_pylist()
    return [trajectory_from_record(record) for record in records]


def none_steering_metadata() -> SteeringMetadata:
    return SteeringMetadata(
        controller="none",
        coordinate=None,
        sign=None,
        edit_norm=0.0,
        trust_region_violations=0,
        applied_edits=[],
    )
