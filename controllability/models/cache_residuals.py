from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from controllability.envs.trajectory_schema import read_jsonl
from controllability.models.residual_patcher import get_hidden_state


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _find_layers(model) -> list[torch.nn.Module]:
    for root_name, attr_name in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
        root = getattr(model, root_name, None)
        layers = getattr(root, attr_name, None) if root is not None else None
        if layers is not None:
            return list(layers)
    raise ValueError(f"Could not locate transformer layers for {model.__class__.__name__}")


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    clean = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user", "assistant", "tool"} and isinstance(content, str) and content:
            clean.append({"role": role, "content": content})
    return clean


def _tokenize(tokenizer, messages: list[dict[str, Any]], max_length: int) -> list[int]:
    clean = _clean_messages(messages)
    if clean:
        try:
            tokens = tokenizer.apply_chat_template(clean, tokenize=True, add_generation_prompt=False)
        except Exception:
            tokens = tokenizer.encode("\n".join(m["content"] for m in clean))
    else:
        tokens = tokenizer.encode("")
    if hasattr(tokens, "ids"):
        tokens = tokens.ids
    if isinstance(tokens, dict) or hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    if isinstance(tokens, str):
        tokens = tokenizer.encode(tokens)
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.flatten().tolist()
    return list(tokens)[-max_length:]


def cache_from_config(config: dict[str, Any]) -> None:
    trajectory_path = Path(config["trajectory_path"])
    model_id = config["model_id"]
    layer = int(config["layer"])
    max_states = int(config.get("max_states", 50_000))
    tokens_per_trajectory = int(config.get("tokens_per_trajectory", 12))
    max_length = int(config.get("max_length", 512))
    batch_size = int(config.get("batch_size", 4))
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    output_dir = Path(
        config.get("output_dir")
        or f"runs/activations/{_safe_name(config['env_id'])}_{_safe_name(model_id)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"residuals_l{layer}.npy"
    manifest_path = output_dir / f"residuals_l{layer}_manifest.json"

    trajectories = read_jsonl(trajectory_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=config.get("device_map", "auto"),
        trust_remote_code=True,
    )
    model.eval()
    layers = _find_layers(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output) -> None:
        captured["hidden"] = get_hidden_state(output)

    handle = layers[layer].register_forward_hook(hook)
    residual_chunks: list[np.ndarray] = []
    total_states = 0
    try:
        for start in tqdm(range(0, len(trajectories), batch_size), desc="cache residuals"):
            batch = trajectories[start : start + batch_size]
            token_lists = [_tokenize(tokenizer, traj.messages, max_length=max_length) for traj in batch]
            encoded = tokenizer.pad(
                {"input_ids": token_lists},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                model(**encoded, use_cache=False)
            hidden = captured["hidden"].detach().cpu()
            attention = encoded["attention_mask"].detach().cpu()
            for row_idx, mask in enumerate(attention):
                valid_positions = torch.nonzero(mask, as_tuple=False).flatten()
                if len(valid_positions) == 0:
                    continue
                selected = valid_positions[-tokens_per_trajectory:]
                arr = hidden[row_idx, selected, :].float().numpy().astype(np.float16)
                residual_chunks.append(arr)
                total_states += arr.shape[0]
                if total_states >= max_states:
                    break
            if total_states >= max_states:
                break
    finally:
        handle.remove()

    residuals = np.concatenate(residual_chunks, axis=0)[:max_states]
    np.save(output_path, residuals)
    with manifest_path.open("w") as f:
        json.dump(
            {
                "env_id": config["env_id"],
                "model_id": model_id,
                "trajectory_path": str(trajectory_path),
                "layer": layer,
                "output_path": str(output_path),
                "shape": list(residuals.shape),
                "dtype": str(residuals.dtype),
                "tokens_per_trajectory": tokens_per_trajectory,
                "max_length": max_length,
                "git_hash": _git_hash(),
                "timestamp_utc": _utc_now(),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cache_from_config(_load_config(args.config))


if __name__ == "__main__":
    main()
