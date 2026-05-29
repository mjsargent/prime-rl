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


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def _regularized_covariance(z: np.ndarray, ridge: float) -> np.ndarray:
    centered = z - z.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    scale = float(np.trace(cov) / max(cov.shape[0], 1))
    return _symmetrize(cov + ridge * max(scale, 1e-12) * np.eye(cov.shape[0], dtype=np.float64))


def _generalized_spectrum(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = _symmetrize(np.asarray(numerator, dtype=np.float64))
    denominator = _symmetrize(np.asarray(denominator, dtype=np.float64))
    scale = float(np.trace(denominator) / max(denominator.shape[0], 1))
    denominator = denominator + ridge * max(scale, 1e-12) * np.eye(denominator.shape[0])
    denom_evals, denom_evecs = np.linalg.eigh(denominator)
    keep = denom_evals > max(denom_evals.max(initial=0.0) * 1e-8, 1e-12)
    if not np.any(keep):
        return np.zeros(numerator.shape[0], dtype=float), np.eye(numerator.shape[0], dtype=float)
    inv_sqrt = (denom_evecs[:, keep] / np.sqrt(denom_evals[keep])) @ denom_evecs[:, keep].T
    whitened = _symmetrize(inv_sqrt @ numerator @ inv_sqrt)
    evals, whitened_evecs = np.linalg.eigh(whitened)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    evecs = inv_sqrt @ whitened_evecs[:, order]
    return evals.astype(float), evecs.astype(float)


def _matrix_rank_summary(matrix: np.ndarray) -> dict[str, Any]:
    eigvals = np.clip(np.linalg.eigvalsh(_symmetrize(matrix)), 0.0, None)[::-1]
    return {
        "effective_rank": _effective_rank(eigvals),
        "stable_rank": _stable_rank(eigvals),
        "top_eigenvalues": eigvals[: min(16, len(eigvals))].tolist(),
    }


def _average_controllability(
    model,
    layers,
    token_lists: list[list[int]],
    layer_states: dict[int, np.ndarray],
    layer_pairs: list[tuple[int, int]],
    *,
    chart_dim: int,
    jacobian_states: int,
    residual_metrics: list[str],
    positions_per_sequence: int,
    ridge: float,
) -> list[dict[str, Any]]:
    metric_bases = {
        (layer_id, metric): _metric_basis(states, positions_per_sequence, metric)
        for layer_id, states in layer_states.items()
        for metric in residual_metrics
    }
    chart_means = {}
    chart_components = {}
    chart_covariances = {}
    for layer_id, states in layer_states.items():
        mean, components, _ = _fit_pca(states, chart_dim)
        chart_means[layer_id] = mean
        chart_components[layer_id] = components
        z = (states - mean) @ components.T
        chart_covariances[layer_id] = _regularized_covariance(z, ridge)

    results = []
    jacobian_token_lists = token_lists[: min(jacobian_states, len(token_lists))]
    for patch_layer, readout_layer in tqdm(layer_pairs, desc="average controllability"):
        jacobians = []
        for token_ids in jacobian_token_lists:
            jacobians.append(
                _jacobian_rows(
                    model,
                    layers,
                    token_ids,
                    patch_layer=patch_layer,
                    readout_layer=readout_layer,
                    chart_components=chart_components[readout_layer],
                )
            )
        for metric in residual_metrics:
            matrices = [
                _controllability_matrix(jacobian, metric_bases[(patch_layer, metric)])
                for jacobian in jacobians
            ]
            m_bar = _symmetrize(np.mean(np.stack(matrices, axis=0), axis=0))
            generalized_evals, _ = _generalized_spectrum(
                m_bar,
                chart_covariances[readout_layer],
                ridge=ridge,
            )
            local_eigenvalues = [
                np.clip(np.linalg.eigvalsh(_symmetrize(matrix)), 0.0, None)[::-1]
                for matrix in matrices
            ]
            local_effective_ranks = [_effective_rank(values) for values in local_eigenvalues]
            local_stable_ranks = [_stable_rank(values) for values in local_eigenvalues]
            results.append(
                {
                    "patch_layer": patch_layer,
                    "readout_layer": readout_layer,
                    "residual_metric": metric,
                    "chart_dim": chart_dim,
                    "jacobian_states": len(jacobians),
                    "mean_controllability": _matrix_rank_summary(m_bar),
                    "generalized_effective_rank": _effective_rank(generalized_evals),
                    "generalized_stable_rank": _stable_rank(generalized_evals),
                    "generalized_top_eigenvalues": generalized_evals[: min(16, len(generalized_evals))].tolist(),
                    "local_effective_rank_median": float(np.median(local_effective_ranks)),
                    "local_stable_rank_median": float(np.median(local_stable_ranks)),
                }
            )
    return results


def _pca_features(
    patch_states: np.ndarray,
    readout_states: np.ndarray,
    *,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([patch_states, readout_states], axis=0).astype(np.float32)
    mean, components, _ = _fit_pca(combined, feature_dim)
    x = (patch_states - mean) @ components.T
    y = (readout_states - mean) @ components.T
    return x.astype(np.float64), y.astype(np.float64)


def _fit_dmd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    ridge: float,
) -> dict[str, Any]:
    xtx = x_train.T @ x_train
    scale = float(np.trace(xtx) / max(xtx.shape[0], 1))
    operator = np.linalg.solve(
        xtx + ridge * max(scale, 1e-12) * np.eye(xtx.shape[0]),
        x_train.T @ y_train,
    )
    pred = x_test @ operator
    centered = y_test - y_test.mean(axis=0, keepdims=True)
    sse = float(np.sum((y_test - pred) ** 2))
    sst = float(np.sum(centered**2))
    r2 = 1.0 - sse / max(sst, 1e-12)
    singular_values = np.linalg.svd(operator, compute_uv=False)
    eigenvalues = np.linalg.eigvals(operator)
    abs_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    return {
        "test_r2": float(r2),
        "operator_effective_rank": _effective_rank(singular_values),
        "operator_stable_rank": _stable_rank(singular_values),
        "spectral_radius": float(abs_eigenvalues[0]) if len(abs_eigenvalues) else 0.0,
        "persistent_modes_abs_ge_0_95": int(np.sum(abs_eigenvalues >= 0.95)),
        "persistent_modes_abs_ge_0_90": int(np.sum(abs_eigenvalues >= 0.90)),
        "persistent_modes_abs_ge_0_80": int(np.sum(abs_eigenvalues >= 0.80)),
        "top_abs_eigenvalues": abs_eigenvalues[: min(16, len(abs_eigenvalues))].tolist(),
        "top_singular_values": singular_values[: min(16, len(singular_values))].tolist(),
    }


def _koopman_dmd(
    layer_states: dict[int, np.ndarray],
    layer_pairs: list[tuple[int, int]],
    *,
    feature_dims: list[int],
    ridge: float,
    test_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    results = []
    rng = np.random.default_rng(seed)
    for patch_layer, readout_layer in tqdm(layer_pairs, desc="koopman dmd"):
        patch_states = layer_states[patch_layer]
        readout_states = layer_states[readout_layer]
        if len(patch_states) != len(readout_states):
            raise ValueError(f"Layer states for ({patch_layer}, {readout_layer}) are not aligned")
        permutation = rng.permutation(len(patch_states))
        test_size = max(1, int(round(len(permutation) * test_fraction)))
        test_idx = np.sort(permutation[:test_size])
        train_idx = np.sort(permutation[test_size:])
        for feature_dim in feature_dims:
            x, y = _pca_features(
                patch_states,
                readout_states,
                feature_dim=feature_dim,
            )
            result = _fit_dmd(
                x[train_idx],
                y[train_idx],
                x[test_idx],
                y[test_idx],
                ridge=ridge,
            )
            result.update(
                {
                    "patch_layer": patch_layer,
                    "readout_layer": readout_layer,
                    "feature_dim": feature_dim,
                    "train_states": int(len(train_idx)),
                    "test_states": int(len(test_idx)),
                }
            )
            results.append(result)
    return results


def run_graph_free(config_path: Path) -> int:
    config = _load_config(config_path)
    timestamp = _utc_now()
    env_id = config["env_id"]
    model_id = config["model_id"]
    run_id = config.get("run_id") or (
        f"stage1_graph_free_spectra_{_safe_name(model_id)}_"
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

    ridge = float(config.get("ridge", 1e-4))
    average_results = _average_controllability(
        model,
        layers,
        token_lists,
        layer_states,
        layer_pairs,
        chart_dim=int(config.get("chart_dim", 32)),
        jacobian_states=int(config.get("jacobian_states", 8)),
        residual_metrics=[str(value) for value in config.get("residual_metrics", ["identity", "cov_h", "cov_delta_h"])],
        positions_per_sequence=positions_per_sequence,
        ridge=ridge,
    )
    dmd_results = _koopman_dmd(
        layer_states,
        layer_pairs,
        feature_dims=[int(value) for value in config.get("dmd_feature_dims", [16, 32, 64, 128])],
        ridge=ridge,
        test_fraction=float(config.get("dmd_test_fraction", 0.25)),
        seed=seed + 10_000,
    )

    best_average = max(average_results, key=lambda row: row["generalized_effective_rank"], default={})
    best_dmd = max(dmd_results, key=lambda row: row["test_r2"], default={})
    metrics = {
        "best_average_patch_layer": best_average.get("patch_layer"),
        "best_average_readout_layer": best_average.get("readout_layer"),
        "best_average_metric": best_average.get("residual_metric"),
        "best_average_generalized_effective_rank": best_average.get("generalized_effective_rank"),
        "best_average_generalized_stable_rank": best_average.get("generalized_stable_rank"),
        "best_dmd_patch_layer": best_dmd.get("patch_layer"),
        "best_dmd_readout_layer": best_dmd.get("readout_layer"),
        "best_dmd_feature_dim": best_dmd.get("feature_dim"),
        "best_dmd_test_r2": best_dmd.get("test_r2"),
        "best_dmd_persistent_modes_abs_ge_0_90": best_dmd.get("persistent_modes_abs_ge_0_90"),
    }
    summary = {
        "stage": "stage1_graph_free_spectra",
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
        "formulation_a_average_controllability": average_results,
        "formulation_c_koopman_dmd": dmd_results,
        "gate": "diagnostic",
        "gate_reason": "graph-free diagnostic only; does not reopen the stopped Stage 1 gate",
        "next_stage_inputs": {},
    }
    _write_json(run_dir / "average_controllability_spectrum.json", {"rows": average_results})
    _write_json(run_dir / "koopman_dmd_spectrum.json", {"rows": dmd_results})
    _write_json(run_dir / "summary.json", summary)
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp,
            "stage": "stage1_graph_free_spectra",
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
    raise SystemExit(run_graph_free(args.config))


if __name__ == "__main__":
    main()
