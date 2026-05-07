from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from controllability.envs.trajectory_schema import Trajectory, trajectory_to_record, write_jsonl


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    top_k: int | None = None
    min_p: float | None = None
    max_completion_tokens: int | None = 128
    min_tokens: int | None = None
    seed: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def to_sampling_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if self.temperature is not None:
            args["temperature"] = self.temperature
        if self.top_p is not None:
            args["top_p"] = self.top_p
        if self.max_completion_tokens is not None:
            args["max_completion_tokens"] = self.max_completion_tokens
        if self.seed is not None:
            args["seed"] = self.seed

        extra_body = dict(self.extra_body)
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if self.min_tokens is not None:
            extra_body["min_tokens"] = self.min_tokens
        if extra_body:
            args["extra_body"] = extra_body
        return args


@dataclass(frozen=True)
class SteeringSpec:
    controller: Literal["none", "one_shot", "open_loop", "closed_loop"]
    coordinate: int | None = None
    sign: Literal["+", "-"] | None = None
    edit_norm: float = 0.0
    patch_layer: int | None = None
    patch_position: int = -1
    readout_layer: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_failed_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            json.dump(row, f, sort_keys=True)
            f.write("\n")


def write_rollout_outputs(output_dir: Path, trajectories: list[Trajectory], failed: list[dict[str, Any]]) -> None:
    write_jsonl(output_dir / "trajectories.jsonl", trajectories)
    write_failed_rows(output_dir / "failed.jsonl", failed)
    with (output_dir / "trajectories.records.json").open("w") as f:
        json.dump([trajectory_to_record(traj) for traj in trajectories], f, indent=2, sort_keys=True)
        f.write("\n")
