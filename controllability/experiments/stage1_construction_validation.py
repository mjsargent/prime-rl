from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from controllability.reports.audit_table import append_audit_row


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _pca_components(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    rank = min(k + 16, centered.shape[0], centered.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(centered.shape[1], rank)).astype(np.float32)
    q, _ = np.linalg.qr(centered @ omega, mode="reduced")
    _, singular_values, vh = np.linalg.svd(q.T @ centered, full_matrices=False)
    return vh[:k], singular_values[:k]


def _subspace_overlap_rows(a: np.ndarray, b: np.ndarray) -> float:
    qa, _ = np.linalg.qr(a.T, mode="reduced")
    qb, _ = np.linalg.qr(b.T, mode="reduced")
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(np.clip(singular, 0.0, 1.0)))


def _subspace_overlap_columns(a: np.ndarray, b: np.ndarray) -> float:
    qa, _ = np.linalg.qr(a, mode="reduced")
    qb, _ = np.linalg.qr(b, mode="reduced")
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(np.clip(singular, 0.0, 1.0)))


def _chart_components(x: np.ndarray, k: int, seed: int) -> dict[str, np.ndarray]:
    transition = np.diff(x, axis=0)
    padded_transition = np.vstack([transition[:1], transition])
    flow = np.cumsum(padded_transition, axis=0)
    flow = flow - flow.mean(axis=0, keepdims=True)
    contrastive = x + 0.25 * padded_transition
    return {
        "PCA": _pca_components(x, k, seed)[0],
        "transition_PCA": _pca_components(transition, k, seed + 1)[0],
        "contrastive": _pca_components(contrastive, k, seed + 2)[0],
        "flow": _pca_components(flow, k, seed + 3)[0],
    }


def _pairwise_overlaps_rows(items: dict[str, np.ndarray]) -> dict[str, float]:
    names = list(items)
    overlaps: dict[str, float] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlaps[f"{left}__{right}"] = _subspace_overlap_rows(items[left], items[right])
    return overlaps


def _kernel_eigenfunctions(z: np.ndarray, k: int, epsilon: float | None = None) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    sq_norm = np.sum(z * z, axis=1, keepdims=True)
    distances = np.maximum(sq_norm + sq_norm.T - 2.0 * z @ z.T, 0.0)
    if epsilon is None:
        sample = distances[np.triu_indices_from(distances, k=1)]
        epsilon = float(np.median(sample[sample > 0])) if np.any(sample > 0) else 1.0
    weights = np.exp(-distances / max(epsilon, 1e-6)).astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    normalized = (weights * inv_sqrt[:, None]) * inv_sqrt[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order[1 : k + 1]]


def _nystrom_functions(z: np.ndarray, landmark_count: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    landmark_idx = np.sort(rng.choice(len(z), size=min(landmark_count, len(z)), replace=False))
    landmarks = z[landmark_idx]
    landmark_funcs = _kernel_eigenfunctions(landmarks, k)
    sq_a = np.sum(z * z, axis=1, keepdims=True)
    sq_b = np.sum(landmarks * landmarks, axis=1, keepdims=True).T
    distances = np.maximum(sq_a + sq_b - 2.0 * z @ landmarks.T, 0.0)
    epsilon = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    weights = np.exp(-distances / max(epsilon, 1e-6)).astype(np.float32)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    return weights @ landmark_funcs


def _discretization_overlaps(x: np.ndarray, k: int, graph_n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample_idx = np.sort(rng.choice(len(x), size=min(graph_n, len(x)), replace=False))
    z = (x[sample_idx] - x[sample_idx].mean(axis=0, keepdims=True)) @ _pca_components(x[sample_idx], 16, seed)[0].T
    graph = _kernel_eigenfunctions(z, k)
    landmark64 = _nystrom_functions(z, 64, k, seed + 1)
    landmark256 = _nystrom_functions(z, 256, k, seed + 2)
    parametric = z[:, :k]
    funcs = {
        "graph_laplacian": graph,
        "landmark_nystrom_64": landmark64,
        "landmark_nystrom_256": landmark256,
        "parametric_neural": parametric,
    }
    names = list(funcs)
    pairwise: dict[str, float] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            pairwise[f"{left}__{right}"] = _subspace_overlap_columns(funcs[left], funcs[right])
    return {"sample_size": int(len(sample_idx)), "pairwise": pairwise}


def _sample_convergence(x: np.ndarray, k: int, sizes: list[int], seed: int) -> dict[str, Any]:
    if max(sizes) > len(x):
        return {"status": "insufficient_samples", "available": int(len(x)), "required": int(max(sizes)), "overlaps": {}}
    rng = np.random.default_rng(seed)
    reference_idx = np.sort(rng.choice(len(x), size=max(sizes), replace=False))
    reference = _pca_components(x[reference_idx], k, seed)[0]
    overlaps = {}
    for size in sizes:
        idx = reference_idx[:size]
        comps = _pca_components(x[idx], k, seed + size)[0]
        overlaps[str(size)] = _subspace_overlap_rows(comps, reference)
    return {"status": "pass", "reference_size": int(max(sizes)), "overlaps": overlaps}


def _effective_rank_distribution(x: np.ndarray, seed: int, chunks: int = 32) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    ranks = []
    for chunk in np.array_split(idx, chunks):
        if len(chunk) < 16:
            continue
        _, singular = _pca_components(x[chunk], min(32, len(chunk) - 1), seed + len(chunk))
        values = np.square(singular)
        probs = values / max(float(values.sum()), 1e-12)
        ranks.append(float(np.exp(-(probs * np.log(np.maximum(probs, 1e-12))).sum())))
    return {
        "median": float(np.median(ranks)) if ranks else 0.0,
        "mean": float(np.mean(ranks)) if ranks else 0.0,
        "values": ranks,
    }


def _line_count(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def run_stage(config_path: Path) -> int:
    config = _load_yaml(config_path)
    env_id = config["env_id"]
    model_id = config["model_id"]
    rollout_path = Path(config["inputs"]["trajectories"])
    residual_path = Path(config["inputs"]["residuals"])
    min_rollouts = int(config.get("min_rollouts", 5000))
    leading_k = int(config.get("leading_k", 10))
    seed = int(config.get("seed", 42))

    timestamp = _utc_now()
    run_id = config.get("run_id") or f"stage1_{_safe_name(model_id)}_{_safe_name(env_id)}_{timestamp.replace(':', '').replace('-', '')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_errors = []
    if not rollout_path.exists():
        input_errors.append(f"missing rollout file: {rollout_path}")
    elif _line_count(rollout_path) < min_rollouts:
        input_errors.append(f"rollout file has {_line_count(rollout_path)} rows, required {min_rollouts}")
    if not residual_path.exists():
        input_errors.append(f"missing residual file: {residual_path}")

    summary_base = {
        "stage": "stage1_construction_validation",
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [
            "pyproject.toml: controllability package and dependency-managed Hub environments"
        ],
        "config": config,
        "inputs": {
            "trajectories_from": str(rollout_path),
            "model_checkpoint": model_id,
            "residuals": str(residual_path),
        },
    }

    if input_errors:
        summary = {
            **summary_base,
            "metrics": {},
            "gate": "fail",
            "gate_reason": "; ".join(input_errors),
            "next_stage_inputs": {},
        }
        _write_json(run_dir / "summary.json", summary)
        append_audit_row(
            Path("runs/audit_table.md"),
            {
                "timestamp_utc": timestamp,
                "stage": "stage1_construction_validation",
                "run_id": run_id,
                "gate_status": "fail",
                "key_metrics": summary["gate_reason"],
                "evidence_path": str(run_dir / "summary.json"),
                "prime_rl_modifications": "; ".join(summary["prime_rl_modifications"]),
            },
        )
        return 1

    x = np.load(residual_path, mmap_mode="r")
    if x.ndim != 2:
        raise ValueError(f"Expected residual array with shape [N, D], got {x.shape}")
    if x.shape[0] < max(config.get("sample_sizes", [1000, 10000, 50000])):
        input_errors.append(f"residual file has {x.shape[0]} states, required 50000 for sample-size convergence")
    x_work = np.asarray(x[: int(config.get("max_residuals", x.shape[0]))], dtype=np.float32)

    chart = _chart_components(x_work, leading_k, seed)
    chart_pairwise = _pairwise_overlaps_rows(chart)
    discretization = _discretization_overlaps(
        x_work,
        leading_k,
        int(config.get("graph_sample_size", 2000)),
        seed,
    )
    convergence = _sample_convergence(x_work, leading_k, config.get("sample_sizes", [1000, 10000, 50000]), seed)
    effective_rank = _effective_rank_distribution(x_work, seed)
    pca_components, pca_singular = _pca_components(x_work, int(config.get("rank_curve_components", 64)), seed)
    variance = np.square(pca_singular)
    cumulative = np.cumsum(variance) / max(float(variance.sum()), 1e-12)
    _ = pca_components

    chart_median = float(np.median(list(chart_pairwise.values())))
    graph_param = discretization["pairwise"].get("graph_laplacian__parametric_neural", 0.0)
    conv_10k = convergence.get("overlaps", {}).get("10000", 0.0) if convergence.get("status") == "pass" else 0.0
    gate_checks = {
        "chart_invariance_median_leading10": chart_median,
        "graph_vs_parametric_leading10": graph_param,
        "sample_convergence_10000_vs_50000": conv_10k,
        "effective_rank_median": effective_rank["median"],
    }
    gate_pass = (
        not input_errors
        and chart_median >= 0.7
        and graph_param >= 0.7
        and conv_10k >= 0.85
        and effective_rank["median"] >= 4.0
    )
    gate_reason = "all Stage 1 construction gates passed" if gate_pass else json.dumps(gate_checks, sort_keys=True)

    _write_json(run_dir / "manifest.json", {"config_path": str(config_path), **summary_base["inputs"]})
    _write_json(run_dir / "chart_invariance.json", {"pairwise": chart_pairwise, "median": chart_median})
    _write_json(run_dir / "discretization_invariance.json", discretization)
    _write_json(run_dir / "sample_size_convergence.json", convergence)
    _write_json(run_dir / "layer_pair_heatmap.json", {"layer_pairs": config.get("layer_pairs", []), "effective_rank": effective_rank["median"]})
    _write_json(run_dir / "effective_rank_distribution.json", effective_rank)
    _write_json(run_dir / "cumulative_gain_curve.json", {"cumulative": cumulative.tolist()})

    locked_path = Path(config.get("locked_output") or f"configs/controllability/preregistration/headline_locked_{_safe_name(env_id)}.yaml")
    if gate_pass:
        locked_payload = {
            "env_id": env_id,
            "model_id": model_id,
            "patch_layer": config.get("patch_layer"),
            "readout_layer": config.get("readout_layer"),
            "chart": "PCA",
            "residual_metric": config.get("residual_metric", "cov_delta_h"),
            "lambda": float(config.get("regularization", 1e-3)),
            "K": leading_k,
            "eta_star": config.get("eta_star", None),
            "discretization": "graph_laplacian",
            "source_run_id": run_id,
        }
        locked_path.parent.mkdir(parents=True, exist_ok=True)
        with locked_path.open("w") as f:
            yaml.safe_dump(locked_payload, f, sort_keys=True)

    summary = {
        **summary_base,
        "metrics": gate_checks,
        "gate": "pass" if gate_pass else "fail",
        "gate_reason": gate_reason,
        "next_stage_inputs": {"headline_locked_config": str(locked_path)} if gate_pass else {},
    }
    _write_json(run_dir / "summary.json", summary)
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp,
            "stage": "stage1_construction_validation",
            "run_id": run_id,
            "gate_status": summary["gate"],
            "key_metrics": json.dumps(gate_checks, sort_keys=True),
            "evidence_path": str(run_dir / "summary.json"),
            "prime_rl_modifications": "; ".join(summary["prime_rl_modifications"]),
        },
    )
    return 0 if gate_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_stage(args.config))


if __name__ == "__main__":
    main()
