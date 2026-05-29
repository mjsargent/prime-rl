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


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


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


def _randomized_pca(
    x: np.ndarray,
    dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    rank = min(dim + 16, centered.shape[0], centered.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(centered.shape[1], rank)).astype(np.float32)
    q, _ = np.linalg.qr(centered @ omega, mode="reduced")
    _, singular_values, vh = np.linalg.svd(q.T @ centered, full_matrices=False)
    total_variance = float(np.sum(centered * centered, dtype=np.float64))
    return mean.squeeze(0), vh[:dim], singular_values[:dim], total_variance


def _variance_summary(singular_values: np.ndarray, total_variance: float, checkpoints: list[int]) -> dict[str, Any]:
    variances = np.square(np.asarray(singular_values, dtype=np.float64))
    denom = max(total_variance, 1e-12)
    cumulative = np.cumsum(variances) / denom
    usable_checkpoints = [point for point in checkpoints if point <= len(cumulative)]
    return {
        "effective_rank_top_components": _effective_rank(variances),
        "stable_rank_top_components": _stable_rank(variances),
        "cumulative_explained_variance": {
            str(point): float(cumulative[point - 1]) for point in usable_checkpoints
        },
        "top_variance_fractions": (variances[: min(16, len(variances))] / denom).tolist(),
    }


def _pairwise_sq_distances(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    sq_norm = np.sum(z * z, axis=1, keepdims=True)
    return np.maximum(sq_norm + sq_norm.T - 2.0 * z @ z.T, 0.0)


def _normalized_eigendecomposition(weights: np.ndarray, eig_count: int) -> tuple[np.ndarray, np.ndarray]:
    degree = weights.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    normalized = (weights * inv_sqrt[:, None]) * inv_sqrt[None, :]
    evals, evecs = np.linalg.eigh(normalized)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    nontrivial = min(eig_count, len(evals) - 1)
    laplacian_mu = 1.0 - evals[1 : nontrivial + 1]
    return laplacian_mu.astype(float), evecs[:, 1 : nontrivial + 1].astype(np.float32)


def _kernel_eigendecomposition(
    z: np.ndarray,
    eig_count: int,
    bandwidth_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    distances = _pairwise_sq_distances(z)
    sample = distances[np.triu_indices_from(distances, k=1)]
    base_epsilon = float(np.median(sample[sample > 0])) if np.any(sample > 0) else 1.0
    epsilon = max(base_epsilon * float(bandwidth_scale), 1e-6)
    weights = np.exp(-distances / epsilon).astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    mu, evecs = _normalized_eigendecomposition(weights, eig_count)
    return mu, evecs, epsilon


def _knn_eigendecomposition(z: np.ndarray, eig_count: int, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    distances = _pairwise_sq_distances(z)
    sample = distances[np.triu_indices_from(distances, k=1)]
    epsilon = float(np.median(sample[sample > 0])) if np.any(sample > 0) else 1.0
    weights = np.zeros_like(distances, dtype=np.float32)
    k = min(int(neighbors) + 1, distances.shape[0])
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    for row_idx, cols in enumerate(nearest):
        cols = cols[cols != row_idx]
        weights[row_idx, cols] = np.exp(-distances[row_idx, cols] / max(epsilon, 1e-6))
    weights = np.maximum(weights, weights.T)
    np.fill_diagonal(weights, 0.0)
    return _normalized_eigendecomposition(weights, eig_count)


def _subspace_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    qa, _ = np.linalg.qr(a[:, :k], mode="reduced")
    qb, _ = np.linalg.qr(b[:, :k], mode="reduced")
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(np.clip(singular, 0.0, 1.0)))


def _natural_k(mu: list[float], max_k: int) -> tuple[int, list[float]]:
    usable = min(max_k + 1, len(mu))
    if usable < 2:
        return 1, []
    gaps = [float(mu[idx] - mu[idx - 1]) for idx in range(1, usable)]
    return int(np.argmax(gaps) + 1), gaps


def _sample_indices(population: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size > population:
        raise ValueError(f"Requested {sample_size} samples from population {population}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(population, size=sample_size, replace=False))


def _load_sample(residuals: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(residuals[indices], dtype=np.float32)


def _manifest_for(residual_path: Path) -> dict[str, Any]:
    manifest = residual_path.with_name(f"{residual_path.stem}_manifest.json")
    if not manifest.exists():
        return {}
    with manifest.open() as f:
        return json.load(f)


def _transition_sample(
    residuals: np.ndarray,
    *,
    tokens_per_trajectory: int,
    max_transitions: int,
    seed: int,
) -> np.ndarray:
    usable_states = residuals.shape[0] - (residuals.shape[0] % tokens_per_trajectory)
    if usable_states <= tokens_per_trajectory:
        return np.empty((0, residuals.shape[1]), dtype=np.float32)
    group_count = usable_states // tokens_per_trajectory
    transitions_per_group = tokens_per_trajectory - 1
    group_sample = min(group_count, max(1, int(np.ceil(max_transitions / transitions_per_group))))
    rng = np.random.default_rng(seed)
    group_ids = np.sort(rng.choice(group_count, size=group_sample, replace=False))
    chunks = []
    for group_id in group_ids:
        start = int(group_id) * tokens_per_trajectory
        group = np.asarray(residuals[start : start + tokens_per_trajectory], dtype=np.float32)
        chunks.append(np.diff(group, axis=0))
    return np.concatenate(chunks, axis=0)[:max_transitions]


def _local_rank_summary(rank_sweep_path: Path | None) -> dict[str, Any]:
    if rank_sweep_path is None or not rank_sweep_path.exists():
        return {"status": "not_available"}
    with rank_sweep_path.open() as f:
        payload = json.load(f)
    rows = payload.get("rows", [])
    best = max(rows, key=lambda row: row.get("effective_rank_median", 0.0), default={})
    return {
        "status": "available",
        "source": str(rank_sweep_path),
        "best_patch_layer": best.get("patch_layer"),
        "best_readout_layer": best.get("readout_layer"),
        "best_effective_rank_median": best.get("effective_rank_median"),
        "best_stable_rank_median": best.get("stable_rank_median"),
        "rows": [
            {
                "patch_layer": row.get("patch_layer"),
                "readout_layer": row.get("readout_layer"),
                "effective_rank_median": row.get("effective_rank_median"),
                "stable_rank_median": row.get("stable_rank_median"),
            }
            for row in rows
        ],
    }


def _analyze_pair(pair: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    residual_path = Path(pair["residuals"])
    residuals = np.load(residual_path, mmap_mode="r")
    if residuals.ndim != 2:
        raise ValueError(f"Expected [states, hidden] residuals at {residual_path}, got {residuals.shape}")

    seed = int(config.get("seed", 42)) + int(pair["patch_layer"]) * 100 + int(pair["readout_layer"])
    chart_dims = [int(value) for value in config.get("chart_dims", [32, 64, 128, 256])]
    sample_sizes = [int(value) for value in config.get("sample_sizes", [100_000, 200_000])]
    max_chart_dim = max(chart_dims)
    anchor_size = int(config.get("anchor_size", 2000))
    eig_count = int(config.get("eig_count", 16))
    natural_k_max = int(config.get("natural_k_max", 15))
    bandwidth_scales = [float(value) for value in config.get("bandwidth_scales", [0.5, 1.0, 2.0, 4.0])]
    knn_values = [int(value) for value in config.get("knn_values", [15, 30, 60])]
    max_pca_fit_states = int(config.get("max_pca_fit_states", max(sample_sizes)))
    transition_states = int(config.get("transition_pca_states", 50_000))

    if max(sample_sizes) > residuals.shape[0]:
        raise ValueError(f"{residual_path} has {residuals.shape[0]} states, required {max(sample_sizes)}")
    if anchor_size > min(sample_sizes):
        raise ValueError("anchor_size must be <= smallest sample size")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(residuals.shape[0])
    anchor_idx = np.sort(permutation[:anchor_size])
    anchor = _load_sample(residuals, anchor_idx)
    checkpoints = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    eigenspaces: dict[tuple[int, int], np.ndarray] = {}
    spectra: dict[tuple[int, int], list[float]] = {}
    z_anchors: dict[tuple[int, int], np.ndarray] = {}
    pca_summaries: dict[str, Any] = {}
    fit_sizes: dict[str, int] = {}
    for sample_size in sample_sizes:
        fit_size = min(sample_size, max_pca_fit_states)
        sample_idx = np.sort(permutation[:fit_size])
        sample = _load_sample(residuals, sample_idx)
        mean, components, singular_values, total_variance = _randomized_pca(
            sample,
            max_chart_dim,
            seed + sample_size,
        )
        fit_sizes[str(sample_size)] = fit_size
        pca_summaries[str(sample_size)] = _variance_summary(singular_values, total_variance, checkpoints)
        centered_anchor = anchor - mean
        for dim in chart_dims:
            z_anchor = centered_anchor @ components[:dim].T
            z_anchors[(sample_size, dim)] = z_anchor
            mu, evecs, epsilon = _kernel_eigendecomposition(z_anchor, eig_count, bandwidth_scale=1.0)
            spectra[(sample_size, dim)] = mu.tolist()
            eigenspaces[(sample_size, dim)] = evecs
            if sample_size == max(sample_sizes):
                pca_summaries[str(sample_size)][f"kernel_epsilon_d{dim}"] = epsilon

    reference_size = max(sample_sizes)
    non_reference = [size for size in sample_sizes if size != reference_size]
    dimension_results = []
    for dim in chart_dims:
        reference_mu = spectra[(reference_size, dim)]
        natural_k, gaps = _natural_k(reference_mu, natural_k_max)
        natural_k = min(natural_k, eigenspaces[(reference_size, dim)].shape[1])
        leading_10 = min(10, eigenspaces[(reference_size, dim)].shape[1])
        overlaps_natural = {}
        overlaps_10 = {}
        for sample_size in sample_sizes:
            overlaps_natural[str(sample_size)] = _subspace_overlap(
                eigenspaces[(sample_size, dim)],
                eigenspaces[(reference_size, dim)],
                natural_k,
            )
            overlaps_10[str(sample_size)] = _subspace_overlap(
                eigenspaces[(sample_size, dim)],
                eigenspaces[(reference_size, dim)],
                leading_10,
            )
        dimension_results.append(
            {
                "chart_dim": dim,
                "natural_k": natural_k,
                "laplacian_mu_reference": reference_mu,
                "eigengaps_reference": gaps,
                "overlaps_natural_k": overlaps_natural,
                "overlaps_leading_10": overlaps_10,
                "natural_k_min_nonreference_overlap": float(
                    min(overlaps_natural[str(size)] for size in non_reference)
                )
                if non_reference
                else 1.0,
                "leading_10_min_nonreference_overlap": float(
                    min(overlaps_10[str(size)] for size in non_reference)
                )
                if non_reference
                else 1.0,
            }
        )

    reference_dim = int(config.get("bandwidth_chart_dim", min(64, max_chart_dim)))
    if reference_dim not in chart_dims:
        reference_dim = chart_dims[0]
    reference_eigenspace = eigenspaces[(reference_size, reference_dim)]
    reference_mu = spectra[(reference_size, reference_dim)]
    reference_k, _ = _natural_k(reference_mu, natural_k_max)
    reference_k = min(reference_k, reference_eigenspace.shape[1])
    z_anchor = z_anchors[(reference_size, reference_dim)]
    bandwidth_results = []
    for scale in bandwidth_scales:
        mu, evecs, epsilon = _kernel_eigendecomposition(z_anchor, eig_count, bandwidth_scale=scale)
        natural_k, gaps = _natural_k(mu.tolist(), natural_k_max)
        natural_k = min(natural_k, evecs.shape[1])
        bandwidth_results.append(
            {
                "bandwidth_scale": scale,
                "epsilon": epsilon,
                "natural_k": natural_k,
                "eigengaps": gaps,
                "overlap_with_median_bandwidth_at_reference_k": _subspace_overlap(
                    evecs,
                    reference_eigenspace,
                    min(reference_k, evecs.shape[1]),
                ),
            }
        )

    knn_results = []
    for neighbors in knn_values:
        knn_spaces = {}
        knn_spectra = {}
        for sample_size in sample_sizes:
            mu, evecs = _knn_eigendecomposition(z_anchors[(sample_size, reference_dim)], eig_count, neighbors)
            knn_spaces[sample_size] = evecs
            knn_spectra[sample_size] = mu.tolist()
        reference_knn_mu = knn_spectra[reference_size]
        reference_knn_space = knn_spaces[reference_size]
        mu = np.asarray(reference_knn_mu, dtype=float)
        evecs = reference_knn_space
        natural_k, gaps = _natural_k(mu.tolist(), natural_k_max)
        natural_k = min(natural_k, evecs.shape[1])
        leading_10 = min(10, evecs.shape[1])
        overlaps_natural = {}
        overlaps_10 = {}
        for sample_size in sample_sizes:
            overlaps_natural[str(sample_size)] = _subspace_overlap(
                knn_spaces[sample_size],
                reference_knn_space,
                natural_k,
            )
            overlaps_10[str(sample_size)] = _subspace_overlap(
                knn_spaces[sample_size],
                reference_knn_space,
                leading_10,
            )
        knn_results.append(
            {
                "neighbors": neighbors,
                "natural_k": natural_k,
                "eigengaps": gaps,
                "overlap_with_kernel_at_reference_k": _subspace_overlap(
                    evecs,
                    reference_eigenspace,
                    min(reference_k, evecs.shape[1]),
                ),
                "overlaps_natural_k": overlaps_natural,
                "overlaps_leading_10": overlaps_10,
            }
        )

    manifest = _manifest_for(residual_path)
    tokens_per_trajectory = int(manifest.get("tokens_per_trajectory", config.get("tokens_per_trajectory", 48)))
    transition = _transition_sample(
        residuals,
        tokens_per_trajectory=tokens_per_trajectory,
        max_transitions=transition_states,
        seed=seed + 17,
    )
    transition_summary = {"status": "not_available"}
    if len(transition) > 0:
        _, _, singular_values, total_variance = _randomized_pca(
            transition,
            max_chart_dim,
            seed + 23,
        )
        transition_summary = {
            "status": "available",
            "tokens_per_trajectory": tokens_per_trajectory,
            "num_transitions": int(len(transition)),
            **_variance_summary(singular_values, total_variance, checkpoints),
        }

    return {
        "patch_layer": int(pair["patch_layer"]),
        "readout_layer": int(pair["readout_layer"]),
        "residual_path": str(residual_path),
        "residual_shape": list(residuals.shape),
        "chart_fit_sizes": fit_sizes,
        "chart_dimension_sweep": dimension_results,
        "pca_collapse": pca_summaries,
        "transition_pca_collapse": transition_summary,
        "bandwidth_chart_dim": reference_dim,
        "bandwidth_sweep": bandwidth_results,
        "knn_sweep": knn_results,
    }


def _summarize_confound_status(pair_results: list[dict[str, Any]], local_rank: dict[str, Any]) -> dict[str, Any]:
    max_natural_k = 0
    natural_k_by_dim = []
    max_pca_effective_rank = 0.0
    max_transition_effective_rank = 0.0
    bandwidth_natural_ks = []
    knn_natural_ks = []
    for pair in pair_results:
        for result in pair["chart_dimension_sweep"]:
            max_natural_k = max(max_natural_k, int(result["natural_k"]))
            natural_k_by_dim.append(
                {
                    "patch_layer": pair["patch_layer"],
                    "readout_layer": pair["readout_layer"],
                    "chart_dim": result["chart_dim"],
                    "natural_k": result["natural_k"],
                    "natural_k_min_nonreference_overlap": result["natural_k_min_nonreference_overlap"],
                    "leading_10_min_nonreference_overlap": result["leading_10_min_nonreference_overlap"],
                }
            )
        for pca in pair["pca_collapse"].values():
            max_pca_effective_rank = max(max_pca_effective_rank, float(pca["effective_rank_top_components"]))
        transition = pair["transition_pca_collapse"]
        if transition.get("status") == "available":
            max_transition_effective_rank = max(
                max_transition_effective_rank,
                float(transition["effective_rank_top_components"]),
            )
        bandwidth_natural_ks.extend(int(item["natural_k"]) for item in pair["bandwidth_sweep"])
        knn_natural_ks.extend(int(item["natural_k"]) for item in pair["knn_sweep"])

    return {
        "chart_dimension_ceiling": {
            "status": "not_supported" if max_natural_k < 4 else "possible",
            "max_natural_k_seen": max_natural_k,
            "natural_k_by_dim": natural_k_by_dim,
            "interpretation": "If natural K does not grow with d up to 256, low K is unlikely to be caused by a hard chart-dimension ceiling.",
        },
        "chart_collapse": {
            "status": "possible" if max_pca_effective_rank < 8.0 else "not_supported",
            "max_pca_effective_rank_top_components": max_pca_effective_rank,
            "max_transition_pca_effective_rank_top_components": max_transition_effective_rank,
        },
        "graph_bandwidth": {
            "status": "possible" if max(bandwidth_natural_ks or [0]) >= 4 else "not_supported",
            "natural_ks_seen": bandwidth_natural_ks,
        },
        "knn_graph_type": {
            "status": "possible" if max(knn_natural_ks or [0]) >= 4 else "not_supported",
            "natural_ks_seen": knn_natural_ks,
        },
        "local_rank_from_saved_sweep": local_rank,
        "not_resolved_by_this_run": [
            "residual_metric_sweep over C_l = I, Cov(h_l), Cov(delta h_l): requires recomputing Jacobian matrices, not just residual eigengaps",
            "lambda sweep in G_tau = (M_tau + lambda I)^-1: current Stage 1 eigengap diagnostic is a residual-chart proxy and does not build the true anisotropic controllability graph",
            "patch-position diagnostic: requires fresh HF Jacobian/steering runs at alternate token positions",
            "LayerNorm across-state Jacobian consistency: requires finite-difference and singular-spectrum checks on stored states",
            "mean-vs-local E[M_tau] diagnostic: saved layer-pair sweeps contain local eigenvalues but not full M_tau matrices",
            "rollout-distribution heterogeneity and open-ended environments: require new rollout caches",
        ],
    }


def run_diagnostics(config_path: Path) -> int:
    config = _load_config(config_path)
    timestamp = _utc_now()
    env_id = config["env_id"]
    model_id = config["model_id"]
    run_id = config.get("run_id") or (
        f"stage1_confound_diagnostics_{_safe_name(model_id)}_"
        f"{_safe_name(env_id)}_{timestamp.replace(':', '').replace('-', '')}"
    )
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pair_results = []
    for pair in config["layer_pairs"]:
        result = _analyze_pair(pair, config)
        pair_results.append(result)
        _write_json(run_dir / f"confounds_l{pair['patch_layer']}_r{pair['readout_layer']}.json", result)

    local_rank = _local_rank_summary(Path(config["rank_sweep_path"]) if config.get("rank_sweep_path") else None)
    confound_status = _summarize_confound_status(pair_results, local_rank)
    metrics = {
        "max_natural_k_seen": confound_status["chart_dimension_ceiling"]["max_natural_k_seen"],
        "chart_dimension_ceiling_status": confound_status["chart_dimension_ceiling"]["status"],
        "chart_collapse_status": confound_status["chart_collapse"]["status"],
        "graph_bandwidth_status": confound_status["graph_bandwidth"]["status"],
        "knn_graph_type_status": confound_status["knn_graph_type"]["status"],
        "local_rank_best_effective_rank_median": local_rank.get("best_effective_rank_median"),
    }
    summary = {
        "stage": "stage1_confound_diagnostics",
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "config": config,
        "inputs": {
            "model_checkpoint": model_id,
            "rank_sweep": config.get("rank_sweep_path"),
        },
        "metrics": metrics,
        "confound_status": confound_status,
        "gate": "diagnostic",
        "gate_reason": "confound diagnostics only; does not reopen the stopped Stage 1 gate",
        "next_stage_inputs": {},
    }
    _write_json(run_dir / "confound_status.json", confound_status)
    _write_json(run_dir / "summary.json", summary)
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp,
            "stage": "stage1_confound_diagnostics",
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
    raise SystemExit(run_diagnostics(args.config))


if __name__ == "__main__":
    main()
