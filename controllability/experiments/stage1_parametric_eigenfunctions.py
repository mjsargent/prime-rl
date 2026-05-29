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
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from controllability.envs.trajectory_schema import read_jsonl
from controllability.experiments.stage1_confound_diagnostics import _natural_k, _subspace_overlap
from controllability.experiments.stage1_layer_pair_sweep import (
    _capture_layer_states,
    _controllability_matrix,
    _effective_rank,
    _find_layers,
    _fit_pca,
    _jacobian_rows,
    _metric_basis,
    _resolve_dtype,
    _safe_name,
    _stable_rank,
    _tokenize,
)
from controllability.reports.audit_table import append_audit_row


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _last_position_states(layer_states: np.ndarray, positions_per_sequence: int, count: int) -> np.ndarray:
    usable = len(layer_states) - (len(layer_states) % positions_per_sequence)
    grouped = layer_states[:usable].reshape(-1, positions_per_sequence, layer_states.shape[-1])
    return grouped[:count, -1, :].astype(np.float32)


def _normalize_metric_matrices(matrices: np.ndarray, ridge: float) -> np.ndarray:
    normalized = []
    dim = matrices.shape[-1]
    eye = np.eye(dim, dtype=np.float32)
    for matrix in matrices:
        matrix = ((matrix + matrix.T) / 2.0).astype(np.float32)
        scale = float(np.trace(matrix) / max(dim, 1))
        scale = max(scale, 1e-8)
        normalized.append(matrix / scale + ridge * eye)
    return np.stack(normalized, axis=0).astype(np.float32)


class EigenfunctionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _output_covariance(values: Tensor) -> Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(values.shape[0] - 1, 1)


def _gradient_tensor(outputs: Tensor, inputs: Tensor) -> Tensor:
    grads = []
    for idx in range(outputs.shape[1]):
        grad = torch.autograd.grad(
            outputs[:, idx].sum(),
            inputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(grad)
    return torch.stack(grads, dim=1)


def _rayleigh_energies(outputs: Tensor, gradients: Tensor, metrics: Tensor) -> Tensor:
    centered = outputs - outputs.mean(dim=0, keepdim=True)
    variances = torch.clamp(torch.diag(centered.T @ centered / max(outputs.shape[0] - 1, 1)), min=1e-8)
    energies = torch.einsum("nkd,ndh,nkh->k", gradients, metrics, gradients) / outputs.shape[0]
    return energies / variances


def _train_parametric_eigenfunctions(
    z: np.ndarray,
    metrics: np.ndarray,
    *,
    output_dim: int,
    hidden_dim: int,
    steps: int,
    learning_rate: float,
    ortho_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[EigenfunctionMLP, dict[str, Any]]:
    torch.manual_seed(seed)
    z_mean = z.mean(axis=0, keepdims=True)
    z_std = z.std(axis=0, keepdims=True) + 1e-6
    z_normalized = (z - z_mean) / z_std

    z_tensor = torch.as_tensor(z_normalized, dtype=torch.float32, device=device)
    metric_tensor = torch.as_tensor(metrics, dtype=torch.float32, device=device)
    model = EigenfunctionMLP(z.shape[1], hidden_dim, output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    eye = torch.eye(output_dim, dtype=torch.float32, device=device)
    history = []

    for step in tqdm(range(steps), desc="train parametric eigenfunctions"):
        inputs = z_tensor.detach().clone().requires_grad_(True)
        outputs = model(inputs)
        gradients = _gradient_tensor(outputs, inputs)
        energies = _rayleigh_energies(outputs, gradients, metric_tensor)
        covariance = _output_covariance(outputs)
        mean_loss = outputs.mean(dim=0).square().mean()
        ortho_loss = (covariance - eye).square().mean()
        loss = energies.mean() + ortho_weight * ortho_loss + mean_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(steps // 5, 1) == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "energy_mean": float(energies.mean().detach().cpu()),
                    "ortho_loss": float(ortho_loss.detach().cpu()),
                    "mean_loss": float(mean_loss.detach().cpu()),
                }
            )

    model.z_mean = torch.as_tensor(z_mean, dtype=torch.float32, device=device)  # type: ignore[attr-defined]
    model.z_std = torch.as_tensor(z_std, dtype=torch.float32, device=device)  # type: ignore[attr-defined]
    return model, {"training_history": history}


def _evaluate_parametric_eigenfunctions(
    model: EigenfunctionMLP,
    z: np.ndarray,
    metrics: np.ndarray,
    *,
    device: torch.device,
    natural_k_max: int,
) -> dict[str, Any]:
    z_tensor = torch.as_tensor(z, dtype=torch.float32, device=device)
    z_normalized = (z_tensor - model.z_mean) / model.z_std  # type: ignore[attr-defined]
    metric_tensor = torch.as_tensor(metrics, dtype=torch.float32, device=device)
    inputs = z_normalized.detach().clone().requires_grad_(True)
    outputs = model(inputs)
    gradients = _gradient_tensor(outputs, inputs)
    energies = _rayleigh_energies(outputs, gradients, metric_tensor)
    covariance = _output_covariance(outputs)
    order = torch.argsort(energies)
    sorted_outputs = outputs[:, order].detach().cpu().numpy()
    sorted_energies = energies[order].detach().cpu().numpy()
    gaps = [float(sorted_energies[idx] - sorted_energies[idx - 1]) for idx in range(1, len(sorted_energies))]
    natural_k, _ = _natural_k(sorted_energies.tolist(), natural_k_max)
    covariance_np = covariance.detach().cpu().numpy()
    return {
        "rayleigh_energies": sorted_energies.tolist(),
        "energy_effective_rank_inverse": _effective_rank(1.0 / np.maximum(sorted_energies, 1e-8)),
        "energy_stable_rank_inverse": _stable_rank(1.0 / np.maximum(sorted_energies, 1e-8)),
        "natural_k_by_energy_gap": min(natural_k, len(sorted_energies)),
        "energy_gaps": gaps,
        "output_covariance_diag": np.diag(covariance_np).tolist(),
        "output_covariance_max_abs_offdiag": float(
            np.max(np.abs(covariance_np - np.diag(np.diag(covariance_np))))
        ),
        "ordered_outputs": sorted_outputs,
    }


def _principal_component_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float | None:
    if k <= 0:
        return None
    if a.shape[0] != b.shape[0]:
        return None
    return _subspace_overlap(a, b, min(k, a.shape[1], b.shape[1]))


def _collect_matrices(
    model,
    layers,
    token_lists: list[list[int]],
    layer_states: dict[int, np.ndarray],
    *,
    patch_layer: int,
    readout_layer: int,
    chart_dim: int,
    jacobian_states: int,
    residual_metric: str,
    positions_per_sequence: int,
    metric_ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    readout_states = layer_states[readout_layer]
    mean, components, _ = _fit_pca(readout_states, chart_dim)
    z_states = _last_position_states(readout_states, positions_per_sequence, jacobian_states)
    z = (z_states - mean) @ components.T
    metric_basis = _metric_basis(layer_states[patch_layer], positions_per_sequence, residual_metric)
    matrices = []
    for token_ids in tqdm(token_lists[:jacobian_states], desc=f"jacobians l{patch_layer}->r{readout_layer}"):
        jacobian = _jacobian_rows(
            model,
            layers,
            token_ids,
            patch_layer=patch_layer,
            readout_layer=readout_layer,
            chart_components=components,
        )
        matrices.append(_controllability_matrix(jacobian, metric_basis))
    return z.astype(np.float32), _normalize_metric_matrices(np.stack(matrices, axis=0), metric_ridge)


def _analyze_pair(
    model,
    layers,
    token_lists: list[list[int]],
    layer_states: dict[int, np.ndarray],
    pair: tuple[int, int],
    config: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    patch_layer, readout_layer = pair
    chart_dim = int(config.get("chart_dim", 32))
    output_dim = int(config.get("eigenfunction_count", 8))
    jacobian_states = int(config.get("jacobian_states", 64))
    residual_metric = str(config.get("residual_metric", "cov_h"))
    z, matrices = _collect_matrices(
        model,
        layers,
        token_lists,
        layer_states,
        patch_layer=patch_layer,
        readout_layer=readout_layer,
        chart_dim=chart_dim,
        jacobian_states=jacobian_states,
        residual_metric=residual_metric,
        positions_per_sequence=int(config.get("positions_per_sequence", 8)),
        metric_ridge=float(config.get("metric_ridge", 1e-4)),
    )
    train_count = max(output_dim + 2, int(round(len(z) * float(config.get("train_fraction", 0.75)))))
    train_count = min(train_count, len(z) - 1)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(z))
    train_idx = np.sort(permutation[:train_count])
    test_idx = np.sort(permutation[train_count:])

    trained, training = _train_parametric_eigenfunctions(
        z[train_idx],
        matrices[train_idx],
        output_dim=output_dim,
        hidden_dim=int(config.get("hidden_dim", 64)),
        steps=int(config.get("train_steps", 600)),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        ortho_weight=float(config.get("ortho_weight", 10.0)),
        seed=seed,
        device=device,
    )
    train_eval = _evaluate_parametric_eigenfunctions(
        trained,
        z[train_idx],
        matrices[train_idx],
        device=device,
        natural_k_max=int(config.get("natural_k_max", output_dim - 1)),
    )
    test_eval = _evaluate_parametric_eigenfunctions(
        trained,
        z[test_idx],
        matrices[test_idx],
        device=device,
        natural_k_max=int(config.get("natural_k_max", output_dim - 1)),
    )
    overlap_k = min(output_dim, len(train_idx), len(test_idx))
    output_overlap = _principal_component_overlap(
        train_eval.pop("ordered_outputs"),
        test_eval.pop("ordered_outputs"),
        overlap_k,
    )
    return {
        "patch_layer": patch_layer,
        "readout_layer": readout_layer,
        "chart_dim": chart_dim,
        "eigenfunction_count": output_dim,
        "residual_metric": residual_metric,
        "jacobian_states": int(len(z)),
        "train_states": int(len(train_idx)),
        "test_states": int(len(test_idx)),
        "train": train_eval,
        "test": test_eval,
        "train_test_output_subspace_overlap": output_overlap,
        **training,
    }


def run_parametric(config_path: Path) -> int:
    config = _load_config(config_path)
    timestamp = _utc_now()
    env_id = config["env_id"]
    model_id = config["model_id"]
    run_id = config.get("run_id") or (
        f"stage1_parametric_eigenfunctions_{_safe_name(model_id)}_"
        f"{_safe_name(env_id)}_{timestamp.replace(':', '').replace('-', '')}"
    )
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trajectories = read_jsonl(Path(config["trajectory_path"]))
    sample_trajectories = int(config.get("sample_trajectories", 128))
    positions_per_sequence = int(config.get("positions_per_sequence", 8))
    max_length = int(config.get("max_length", 512))
    batch_size = int(config.get("batch_size", 4))
    seed = int(config.get("seed", 42))

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

    layer_pairs = [(int(left), int(right)) for left, right in config["layer_pairs"]]
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

    pair_results = []
    for pair in layer_pairs:
        result = _analyze_pair(
            model,
            layers,
            token_lists,
            layer_states,
            pair,
            config,
            device=device,
            seed=seed + pair[0] * 100 + pair[1],
        )
        pair_results.append(result)
        _write_json(run_dir / f"parametric_l{pair[0]}_r{pair[1]}.json", result)

    best = min(pair_results, key=lambda row: np.mean(row["test"]["rayleigh_energies"]), default={})
    metrics = {
        "best_patch_layer": best.get("patch_layer"),
        "best_readout_layer": best.get("readout_layer"),
        "best_test_mean_rayleigh_energy": float(np.mean(best.get("test", {}).get("rayleigh_energies", [0.0]))),
        "best_test_inverse_energy_effective_rank": best.get("test", {}).get("energy_effective_rank_inverse"),
        "best_test_natural_k_by_energy_gap": best.get("test", {}).get("natural_k_by_energy_gap"),
        "best_train_test_output_subspace_overlap": best.get("train_test_output_subspace_overlap"),
    }
    summary = {
        "stage": "stage1_parametric_eigenfunctions",
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "config": config,
        "inputs": {
            "trajectories_from": config["trajectory_path"],
            "model_checkpoint": model_id,
        },
        "metrics": metrics,
        "formulation_b_parametric_eigenfunctions": pair_results,
        "gate": "diagnostic",
        "gate_reason": "parametric neural eigenfunction diagnostic only; does not reopen the stopped Stage 1 gate",
        "next_stage_inputs": {},
    }
    _write_json(run_dir / "summary.json", summary)
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp,
            "stage": "stage1_parametric_eigenfunctions",
            "run_id": run_id,
            "gate_status": "diagnostic",
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": str(run_dir / "summary.json"),
            "prime_rl_modifications": "none",
        },
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_parametric(args.config))


if __name__ == "__main__":
    main()
