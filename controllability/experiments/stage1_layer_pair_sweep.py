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
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from controllability.envs.trajectory_schema import read_jsonl
from controllability.models.residual_patcher import get_hidden_state, replace_hidden_state
from controllability.reports.audit_table import append_audit_row


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _find_layers(model) -> list[torch.nn.Module]:
    root = getattr(model, "model", None)
    layers = getattr(root, "layers", None) if root is not None else None
    if layers is not None:
        return list(layers)
    root = getattr(model, "transformer", None)
    layers = getattr(root, "h", None) if root is not None else None
    if layers is not None:
        return list(layers)
    raise ValueError(f"Could not locate transformer layers for {model.__class__.__name__}")


def _hidden_size(config) -> int:
    value = getattr(config, "hidden_size", None)
    if value is None:
        value = getattr(config, "n_embd", None)
    if value is None:
        raise ValueError(f"Could not infer hidden size from {config.__class__.__name__}")
    return int(value)


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
            tokens = tokenizer.encode("\n".join(message["content"] for message in clean))
    else:
        tokens = tokenizer.encode("")
    if hasattr(tokens, "ids"):
        tokens = tokens.ids
    if isinstance(tokens, dict) or hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    if isinstance(tokens, str):
        tokens = tokenizer.encode(tokens)
    if isinstance(tokens, Tensor):
        tokens = tokens.flatten().tolist()
    return list(tokens)[-max_length:]


def _resolve_dtype(value: str) -> torch.dtype:
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {value!r}")


def _fit_pca(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0)
    centered = x - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    usable = min(dim, vh.shape[0])
    return mean, vh[:usable], singular_values[:usable]


def _effective_rank(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    probs = values / values.sum()
    return float(np.exp(-(probs * np.log(np.maximum(probs, 1e-12))).sum()))


def _stable_rank(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.max(initial=0.0) <= 0:
        return 0.0
    return float(values.sum() / values.max())


def _capture_layer_states(
    model,
    tokenizer,
    layers: list[torch.nn.Module],
    layer_ids: list[int],
    token_lists: list[list[int]],
    *,
    positions_per_sequence: int,
    batch_size: int,
) -> dict[int, np.ndarray]:
    captures: dict[int, list[np.ndarray]] = {layer_id: [] for layer_id in layer_ids}
    captured: dict[int, Tensor] = {}
    handles = []

    for layer_id in layer_ids:
        def hook(_module, _inputs, output, *, sink_layer=layer_id) -> None:
            captured[sink_layer] = get_hidden_state(output).detach()

        handles.append(layers[layer_id].register_forward_hook(hook))

    try:
        for start in tqdm(range(0, len(token_lists), batch_size), desc="capture layer states"):
            batch_tokens = token_lists[start : start + batch_size]
            encoded = tokenizer.pad(
                {"input_ids": batch_tokens},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(model.device)
            captured.clear()
            with torch.no_grad():
                model(**encoded, use_cache=False)
            attention = encoded["attention_mask"].detach().cpu()
            for layer_id in layer_ids:
                hidden = captured[layer_id].detach().cpu()
                rows = []
                for row_idx, mask in enumerate(attention):
                    valid_positions = torch.nonzero(mask, as_tuple=False).flatten()
                    if len(valid_positions) == 0:
                        continue
                    selected = valid_positions[-positions_per_sequence:]
                    rows.append(hidden[row_idx, selected, :].float().numpy())
                captures[layer_id].append(np.concatenate(rows, axis=0))
    finally:
        for handle in handles:
            handle.remove()

    return {layer_id: np.concatenate(chunks, axis=0) for layer_id, chunks in captures.items()}


def _metric_basis(layer_states: np.ndarray, positions_per_sequence: int, metric: str) -> np.ndarray | None:
    if metric == "identity":
        return None
    if metric == "cov_h":
        return layer_states.astype(np.float32) - layer_states.mean(axis=0, keepdims=True)
    if metric != "cov_delta_h":
        raise ValueError(f"Unsupported residual metric {metric!r}")
    deltas = []
    usable = len(layer_states) - (len(layer_states) % positions_per_sequence)
    grouped = layer_states[:usable].reshape(-1, positions_per_sequence, layer_states.shape[-1])
    for group in grouped:
        if len(group) > 1:
            deltas.append(np.diff(group, axis=0))
    if not deltas:
        return np.empty((0, layer_states.shape[-1]), dtype=np.float32)
    basis = np.concatenate(deltas, axis=0).astype(np.float32)
    return basis - basis.mean(axis=0, keepdims=True)


def _jacobian_rows(
    model,
    layers: list[torch.nn.Module],
    token_ids: list[int],
    *,
    patch_layer: int,
    readout_layer: int,
    chart_components: np.ndarray,
) -> np.ndarray:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    hidden_dim = _hidden_size(model.config)
    delta = torch.zeros(hidden_dim, device=model.device, dtype=torch.float32, requires_grad=True)
    captured: dict[str, Tensor] = {}

    def patch_hook(_module, _inputs, output):
        hidden = get_hidden_state(output)
        patched = hidden.clone()
        patch_delta = delta.to(device=patched.device, dtype=patched.dtype)
        patched[:, -1, :] = patched[:, -1, :] + patch_delta
        return replace_hidden_state(output, patched)

    def readout_hook(_module, _inputs, output) -> None:
        captured["readout"] = get_hidden_state(output)

    patch_handle = layers[patch_layer].register_forward_hook(patch_hook)
    readout_handle = layers[readout_layer].register_forward_hook(readout_hook)
    try:
        model(input_ids=input_ids, use_cache=False)
        readout = captured["readout"][:, -1, :].float()
        components = torch.as_tensor(chart_components, device=readout.device, dtype=torch.float32)
        chart_values = readout @ components.T
        rows = []
        for idx in range(chart_values.shape[-1]):
            grad = torch.autograd.grad(
                chart_values[0, idx],
                delta,
                retain_graph=idx < chart_values.shape[-1] - 1,
            )[0]
            rows.append(grad.detach().float().cpu().numpy())
        return np.stack(rows, axis=0)
    finally:
        patch_handle.remove()
        readout_handle.remove()
        del captured
        torch.cuda.empty_cache()


def _controllability_matrix(jacobian_rows: np.ndarray, metric_basis: np.ndarray | None) -> np.ndarray:
    if metric_basis is None:
        return jacobian_rows @ jacobian_rows.T
    if len(metric_basis) == 0:
        return np.zeros((jacobian_rows.shape[0], jacobian_rows.shape[0]), dtype=np.float32)
    projected = jacobian_rows @ metric_basis.T
    return (projected @ projected.T) / max(len(metric_basis) - 1, 1)


def run_sweep(config_path: Path) -> int:
    config = _load_config(config_path)
    timestamp = _utc_now()
    env_id = config["env_id"]
    model_id = config["model_id"]
    run_id = config.get("run_id") or f"stage1_layer_pair_sweep_{_safe_name(model_id)}_{_safe_name(env_id)}_{timestamp.replace(':', '').replace('-', '')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trajectories = read_jsonl(Path(config["trajectory_path"]))
    sample_trajectories = int(config.get("sample_trajectories", 64))
    jacobian_states = int(config.get("jacobian_states", 4))
    positions_per_sequence = int(config.get("positions_per_sequence", 8))
    max_length = int(config.get("max_length", 256))
    chart_dim = int(config.get("chart_dim", 8))
    batch_size = int(config.get("batch_size", 4))
    seed = int(config.get("seed", 42))
    metric = config.get("residual_metric", "cov_delta_h")

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(trajectories), size=min(sample_trajectories, len(trajectories)), replace=False))
    selected_trajs = [trajectories[int(idx)] for idx in selected]

    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_resolve_dtype(config.get("dtype", "bfloat16")),
        trust_remote_code=True,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    layers = _find_layers(model)
    last_layer = len(layers) - 1

    requested_pairs = config.get("layer_pairs")
    if requested_pairs is None:
        patch_layers = config.get("patch_layers", [4, 8, 12, 16, 20])
        requested_pairs = []
        for patch_layer in patch_layers:
            for readout_layer in (patch_layer + 2, patch_layer + 4, last_layer):
                if readout_layer <= last_layer and readout_layer > patch_layer:
                    requested_pairs.append([patch_layer, readout_layer])
    layer_pairs = [(int(left), int(right)) for left, right in requested_pairs]
    layer_ids = sorted({layer_id for pair in layer_pairs for layer_id in pair})

    token_lists = [_tokenize(tokenizer, traj.messages, max_length=max_length) for traj in selected_trajs]
    layer_states = _capture_layer_states(
        model,
        tokenizer,
        layers,
        layer_ids,
        token_lists,
        positions_per_sequence=positions_per_sequence,
        batch_size=batch_size,
    )
    chart_components = {layer_id: _fit_pca(states, chart_dim)[1] for layer_id, states in layer_states.items()}
    metric_bases = {layer_id: _metric_basis(states, positions_per_sequence, metric) for layer_id, states in layer_states.items()}

    jacobian_token_lists = token_lists[: min(jacobian_states, len(token_lists))]
    rows = []
    for patch_layer, readout_layer in tqdm(layer_pairs, desc="sweep layer pairs"):
        ranks = []
        stable_ranks = []
        eigenvalues_by_state = []
        for token_ids in jacobian_token_lists:
            jacobian = _jacobian_rows(
                model,
                layers,
                token_ids,
                patch_layer=patch_layer,
                readout_layer=readout_layer,
                chart_components=chart_components[readout_layer],
            )
            matrix = _controllability_matrix(jacobian, metric_bases[patch_layer])
            eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
            eigenvalues = np.clip(eigenvalues, 0.0, None)[::-1]
            ranks.append(_effective_rank(eigenvalues))
            stable_ranks.append(_stable_rank(eigenvalues))
            eigenvalues_by_state.append(eigenvalues.tolist())
        rows.append(
            {
                "patch_layer": patch_layer,
                "readout_layer": readout_layer,
                "effective_rank_mean": float(np.mean(ranks)),
                "effective_rank_median": float(np.median(ranks)),
                "effective_rank_max": float(np.max(ranks)),
                "stable_rank_mean": float(np.mean(stable_ranks)),
                "stable_rank_median": float(np.median(stable_ranks)),
                "state_effective_ranks": ranks,
                "state_stable_ranks": stable_ranks,
                "state_eigenvalues": eigenvalues_by_state,
            }
        )

    best = max(rows, key=lambda row: row["effective_rank_median"]) if rows else {}
    summary = {
        "stage": "stage1_layer_pair_sweep",
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "config": config,
        "inputs": {"trajectories_from": config["trajectory_path"], "model_checkpoint": model_id},
        "metrics": {
            "best_patch_layer": best.get("patch_layer"),
            "best_readout_layer": best.get("readout_layer"),
            "best_effective_rank_median": best.get("effective_rank_median", 0.0),
            "best_stable_rank_median": best.get("stable_rank_median", 0.0),
        },
        "gate": "diagnostic",
        "gate_reason": "diagnostic layer-pair suffix-Jacobian rank sweep; not a Stage 1 pass/fail gate",
        "next_stage_inputs": {},
    }
    _write_json(run_dir / "layer_pair_rank_sweep.json", {"rows": rows})
    _write_json(run_dir / "summary.json", summary)
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp,
            "stage": "stage1_layer_pair_sweep",
            "run_id": run_id,
            "gate_status": "diagnostic",
            "key_metrics": json.dumps(summary["metrics"], sort_keys=True),
            "evidence_path": str(run_dir / "summary.json"),
            "prime_rl_modifications": "none",
        },
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_sweep(args.config))


if __name__ == "__main__":
    main()
