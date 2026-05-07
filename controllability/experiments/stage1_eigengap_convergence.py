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


def _randomized_pca(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    rank = min(k + 16, centered.shape[0], centered.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(centered.shape[1], rank)).astype(np.float32)
    q, _ = np.linalg.qr(centered @ omega, mode="reduced")
    _, _, vh = np.linalg.svd(q.T @ centered, full_matrices=False)
    return mean.squeeze(0), vh[:k]


def _kernel_eigendecomposition(z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.float32)
    sq_norm = np.sum(z * z, axis=1, keepdims=True)
    distances = np.maximum(sq_norm + sq_norm.T - 2.0 * z @ z.T, 0.0)
    sample = distances[np.triu_indices_from(distances, k=1)]
    epsilon = float(np.median(sample[sample > 0])) if np.any(sample > 0) else 1.0
    weights = np.exp(-distances / max(epsilon, 1e-6)).astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    normalized = (weights * inv_sqrt[:, None]) * inv_sqrt[None, :]
    evals, evecs = np.linalg.eigh(normalized)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    nontrivial = min(k, len(evals) - 1)
    laplacian_mu = 1.0 - evals[1 : nontrivial + 1]
    return laplacian_mu.astype(float), evecs[:, 1 : nontrivial + 1].astype(np.float32)


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


def _is_monotone(values: list[float], tolerance: float = 1e-3) -> bool:
    return all(right + tolerance >= left for left, right in zip(values, values[1:], strict=False))


def _analyze_pair(
    residual_path: Path,
    *,
    sample_sizes: list[int],
    anchor_size: int,
    chart_dim: int,
    eig_count: int,
    seed: int,
    natural_k_max: int,
) -> dict[str, Any]:
    residuals = np.load(residual_path, mmap_mode="r")
    if residuals.ndim != 2:
        raise ValueError(f"Expected residual array with shape [N, D], got {residuals.shape}")
    max_sample = max(sample_sizes)
    if residuals.shape[0] < max_sample:
        raise ValueError(f"{residual_path} has {residuals.shape[0]} states, required {max_sample}")
    if anchor_size > min(sample_sizes):
        raise ValueError("anchor_size must be <= smallest sample size")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(residuals.shape[0])
    anchor_idx = np.sort(permutation[:anchor_size])
    anchor = np.asarray(residuals[anchor_idx], dtype=np.float32)

    eigenspaces: dict[int, np.ndarray] = {}
    spectra: dict[int, list[float]] = {}
    for sample_size in sample_sizes:
        sample_idx = np.sort(permutation[:sample_size])
        sample = np.asarray(residuals[sample_idx], dtype=np.float32)
        mean, components = _randomized_pca(sample, chart_dim, seed + sample_size)
        z_anchor = (anchor - mean) @ components.T
        mu, evecs = _kernel_eigendecomposition(z_anchor, eig_count)
        spectra[sample_size] = mu.tolist()
        eigenspaces[sample_size] = evecs

    reference_size = max(sample_sizes)
    reference_mu = spectra[reference_size]
    natural_k, gaps = _natural_k(reference_mu, natural_k_max)
    leading_10 = min(10, eigenspaces[reference_size].shape[1])
    natural_k = min(natural_k, eigenspaces[reference_size].shape[1])

    overlaps_natural = {}
    overlaps_10 = {}
    for sample_size in sample_sizes:
        overlaps_natural[str(sample_size)] = _subspace_overlap(
            eigenspaces[sample_size],
            eigenspaces[reference_size],
            natural_k,
        )
        overlaps_10[str(sample_size)] = _subspace_overlap(
            eigenspaces[sample_size],
            eigenspaces[reference_size],
            leading_10,
        )

    non_reference = [size for size in sample_sizes if size != reference_size]
    natural_values = [overlaps_natural[str(size)] for size in non_reference]
    leading_10_values = [overlaps_10[str(size)] for size in non_reference]
    return {
        "residual_path": str(residual_path),
        "sample_sizes": sample_sizes,
        "reference_size": reference_size,
        "anchor_size": anchor_size,
        "chart_dim": chart_dim,
        "eig_count": eig_count,
        "laplacian_mu_reference": reference_mu,
        "eigengaps_reference": gaps,
        "natural_k": natural_k,
        "leading_10_k": leading_10,
        "overlaps_natural_k": overlaps_natural,
        "overlaps_leading_10": overlaps_10,
        "monotone_natural_k": _is_monotone(natural_values),
        "monotone_leading_10": _is_monotone(leading_10_values),
        "natural_k_100k_vs_200k": overlaps_natural.get("100000"),
        "leading_10_100k_vs_200k": overlaps_10.get("100000"),
    }


def run_diagnostic(config_path: Path) -> int:
    config = _load_config(config_path)
    timestamp = _utc_now()
    env_id = config["env_id"]
    model_id = config["model_id"]
    run_id = config.get("run_id") or f"stage1_eigengap_convergence_{_safe_name(model_id)}_{_safe_name(env_id)}_{timestamp.replace(':', '').replace('-', '')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    sample_sizes = [int(size) for size in config.get("sample_sizes", [10_000, 50_000, 100_000, 200_000])]
    anchor_size = int(config.get("anchor_size", 2000))
    chart_dim = int(config.get("chart_dim", 32))
    eig_count = int(config.get("eig_count", 16))
    seed = int(config.get("seed", 42))
    natural_k_max = int(config.get("natural_k_max", 15))

    pair_results = []
    for pair in config["layer_pairs"]:
        result = _analyze_pair(
            Path(pair["residuals"]),
            sample_sizes=sample_sizes,
            anchor_size=anchor_size,
            chart_dim=chart_dim,
            eig_count=eig_count,
            seed=seed + int(pair["patch_layer"]) * 100 + int(pair["readout_layer"]),
            natural_k_max=natural_k_max,
        )
        result["patch_layer"] = int(pair["patch_layer"])
        result["readout_layer"] = int(pair["readout_layer"])
        pair_results.append(result)
        _write_json(run_dir / f"eigengap_l{pair['patch_layer']}_r{pair['readout_layer']}.json", result)

    stable_pairs = [
        result
        for result in pair_results
        if (result["natural_k_100k_vs_200k"] or 0.0) >= float(config.get("natural_k_overlap_threshold", 0.85))
    ]
    best = max(pair_results, key=lambda result: result["natural_k_100k_vs_200k"] or 0.0)
    metrics = {
        "best_patch_layer": best["patch_layer"],
        "best_readout_layer": best["readout_layer"],
        "best_natural_k": best["natural_k"],
        "best_natural_k_100k_vs_200k": best["natural_k_100k_vs_200k"],
        "best_leading_10_100k_vs_200k": best["leading_10_100k_vs_200k"],
        "num_stable_pairs_at_natural_k": len(stable_pairs),
    }
    summary = {
        "stage": "stage1_eigengap_convergence",
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "config": config,
        "inputs": {"model_checkpoint": model_id},
        "metrics": metrics,
        "gate": "diagnostic_pass" if stable_pairs else "diagnostic_fail",
        "gate_reason": "diagnostic only: natural-K convergence at 100k vs 200k",
        "next_stage_inputs": {},
    }
    _write_json(run_dir / "summary.json", summary)
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": timestamp,
            "stage": "stage1_eigengap_convergence",
            "run_id": run_id,
            "gate_status": summary["gate"],
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
    raise SystemExit(run_diagnostic(args.config))


if __name__ == "__main__":
    main()
