from __future__ import annotations

from controllability.encoders.base import BehavioralEncoder, FixedFeatureEncoder
from controllability.encoders.sentence_encoder import SentenceTrajectoryEncoder
from controllability.encoders.task_aware_encoder import TaskAwareBehavioralEncoderV3


def get_encoder(
    env_id: str,
    *,
    encoder_name: str = "fixed_feature_v1",
    encoder_config: dict | None = None,
) -> BehavioralEncoder:
    encoder_config = encoder_config or {}
    if encoder_name in {"fixed_feature_v1", "fixed"}:
        return FixedFeatureEncoder(env_id=env_id)
    if encoder_name in {"sentence_e5_structured_v1", "sentence_e5"}:
        return SentenceTrajectoryEncoder(env_id=env_id, **encoder_config)
    if encoder_name in {"behavioral_encoder_v3", "task_aware_v3"}:
        return TaskAwareBehavioralEncoderV3(env_id=env_id, **encoder_config)
    raise ValueError(f"Unknown behavioral encoder: {encoder_name}")
