from __future__ import annotations

from controllability.encoders.base import BehavioralEncoder, FixedFeatureEncoder


def get_encoder(env_id: str) -> BehavioralEncoder:
    return FixedFeatureEncoder(env_id=env_id)
