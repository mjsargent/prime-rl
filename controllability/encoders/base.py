from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from controllability.envs.trajectory_schema import Trajectory


@dataclass(frozen=True)
class EncodedTrajectory:
    trajectory_id: str
    env_id: str
    behavior_features: np.ndarray
    quality: float
    coherence: float
    raw_features: dict
    encoder_version: str


class BehavioralEncoder(Protocol):
    env_id: str
    feature_dim: int
    encoder_version: str

    def encode(self, traj: Trajectory) -> EncodedTrajectory: ...

    def encode_batch(self, trajs: list[Trajectory]) -> list[EncodedTrajectory]: ...


class FixedFeatureEncoder:
    encoder_version = "fixed_feature_v1"
    feature_dim = 16

    def __init__(self, env_id: str):
        self.env_id = env_id

    def encode(self, traj: Trajectory) -> EncodedTrajectory:
        text = "\n".join(str(message.get("content", "")) for message in traj.messages)
        assistant_text = "\n".join(
            str(message.get("content", "")) for message in traj.messages if message.get("role") == "assistant"
        )
        features = np.array(
            [
                traj.reward,
                float(traj.success),
                float(len(traj.messages)),
                float(len(traj.tool_calls)),
                float(len(traj.step_metadata)),
                float(len(text)),
                float(len(assistant_text)),
                float(sum(ch.isdigit() for ch in text)),
                float(text.count("\n")),
                float(text.count("```")),
                float(text.lower().count("therefore")),
                float(text.lower().count("because")),
                float(text.lower().count("error")),
                float(text.lower().count("tool")),
                traj.wall_clock_seconds,
                1.0,
            ],
            dtype=np.float32,
        )
        scale = np.maximum(np.abs(features).max(), 1.0)
        features = features / scale
        return EncodedTrajectory(
            trajectory_id=traj.trajectory_id,
            env_id=traj.env_id,
            behavior_features=features,
            quality=float(traj.reward),
            coherence=1.0 if traj.messages else 0.0,
            raw_features={
                "message_count": len(traj.messages),
                "tool_call_count": len(traj.tool_calls),
                "character_count": len(text),
            },
            encoder_version=self.encoder_version,
        )

    def encode_batch(self, trajs: list[Trajectory]) -> list[EncodedTrajectory]:
        return [self.encode(traj) for traj in trajs]
