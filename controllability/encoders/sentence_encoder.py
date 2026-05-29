from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import torch

from controllability.encoders.base import EncodedTrajectory, FixedFeatureEncoder
from controllability.envs.trajectory_schema import Trajectory


def _trajectory_text(traj: Trajectory) -> str:
    sections: list[str] = []
    for message in traj.messages:
        role = str(message.get("role", "unknown"))
        content = message.get("content", "")
        sections.append(f"{role}: {content}")
    if traj.tool_calls:
        sections.append("tool_calls: " + json.dumps(traj.tool_calls, sort_keys=True))
    if traj.step_metadata:
        sections.append("metadata: " + json.dumps(traj.step_metadata, sort_keys=True))
    return "\n".join(sections)


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


@lru_cache(maxsize=4)
def _load_sentence_model(model_id: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32)
    model.eval()
    model.to(device)
    return tokenizer, model


class SentenceTrajectoryEncoder(FixedFeatureEncoder):
    encoder_version = "sentence_e5_structured_v1"

    def __init__(
        self,
        env_id: str,
        *,
        model_id: str = "intfloat/e5-large-v2",
        device: str | None = None,
        max_length: int = 512,
        normalize: bool = True,
    ):
        super().__init__(env_id)
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)
        self.normalize = bool(normalize)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        tokenizer, model = _load_sentence_model(self.model_id, self.device)
        prompts = [f"query: {text}" for text in texts]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"]).float()
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    def _structured_features(self, traj: Trajectory) -> np.ndarray:
        fixed = super().encode(traj)
        return fixed.behavior_features.astype(np.float32)

    def encode(self, traj: Trajectory) -> EncodedTrajectory:
        return self.encode_batch([traj])[0]

    def encode_batch(self, trajs: list[Trajectory]) -> list[EncodedTrajectory]:
        texts = [_trajectory_text(traj) for traj in trajs]
        embeddings = self._embed_texts(texts)
        encoded: list[EncodedTrajectory] = []
        for traj, embedding in zip(trajs, embeddings, strict=True):
            structured = self._structured_features(traj)
            features = np.concatenate([embedding, structured], axis=0).astype(np.float32)
            encoded.append(
                EncodedTrajectory(
                    trajectory_id=traj.trajectory_id,
                    env_id=traj.env_id,
                    behavior_features=features,
                    quality=float(traj.reward),
                    coherence=1.0 if traj.messages else 0.0,
                    raw_features={
                        "sentence_model_id": self.model_id,
                        "sentence_embedding_dim": int(embedding.shape[0]),
                        "structured_feature_dim": int(structured.shape[0]),
                        "trajectory_characters": len(_trajectory_text(traj)),
                    },
                    encoder_version=self.encoder_version,
                )
            )
        return encoded
