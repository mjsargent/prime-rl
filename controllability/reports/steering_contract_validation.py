from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from controllability.experiments.stage4_trajectory_generation import (
    _direction_and_delta,
    _fit_stage4_basis,
    _generate_with_delta,
    _load_base_prompts,
    _tokenize_for_generation,
)
from controllability.models.frozen_model import FrozenModel
from controllability.reports.audit_table import append_audit_row


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _safe_id(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").replace(".", "_")


def _chart_residual(frozen: FrozenModel, basis: dict[str, Any], residual: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(basis["chart_mean"], dtype=torch.float32, device=frozen.device)
    components = torch.as_tensor(basis["chart_components"], dtype=torch.float32, device=frozen.device)
    return (residual.float() - mean) @ components.T


def _base_generate(frozen: FrozenModel, token_ids: list[int], *, max_new_tokens: int) -> list[int]:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=frozen.device)
    generated = frozen.model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=frozen.tokenizer.eos_token_id,
    )
    return generated[0].detach().cpu().tolist()


def _cosine_and_r2(predicted: torch.Tensor, measured: torch.Tensor) -> tuple[float, float]:
    predicted_flat = predicted.float().flatten()
    measured_flat = measured.float().flatten()
    denom = float((measured_flat @ measured_flat).detach().cpu())
    if denom <= 0:
        return 0.0, 0.0
    error = measured_flat - predicted_flat
    cosine = float(F.cosine_similarity(predicted_flat, measured_flat, dim=0).detach().cpu())
    r2 = 1.0 - float((error @ error).detach().cpu()) / max(denom, 1e-12)
    return cosine, r2


def _displacement(
    frozen: FrozenModel,
    basis: dict[str, Any],
    token_ids: list[int],
    *,
    patch_layer: int,
    readout_layer: int,
    delta: torch.Tensor,
) -> torch.Tensor:
    baseline = frozen.get_residual(token_ids, layer=readout_layer, position=-1).float()
    patched = frozen.patched_forward(
        token_ids,
        patch_layer=patch_layer,
        patch_position=-1,
        delta=delta,
        readout_layer=readout_layer,
        readout_position=-1,
    ).float()
    return _chart_residual(frozen, basis, patched) - _chart_residual(frozen, basis, baseline)


def _eta_by_coordinate(config: dict[str, Any]) -> dict[int, float]:
    rows = _load_json(config["stage3_aggregated_path"])["rows"]
    return {
        int(row["coordinate"]): float(row["eta_star_mean"])
        for row in rows
        if row["formulation"] == config["formulation"]
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.median(tensor).item())


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, q).item())


def _validation_cells(
    config: dict[str, Any],
    prompts: list[tuple[str, list[dict[str, str]]]],
) -> list[tuple[str, list[dict[str, str]], int, str]]:
    coordinates = [int(item) for item in config.get("contract_coordinates", config.get("coordinates", []))]
    signs = [str(item) for item in config.get("contract_signs", config.get("signs", ["+"]))]
    max_cells = int(config.get("contract_max_cells", 12))
    cells: list[tuple[str, list[dict[str, str]], int, str]] = []
    for prompt_id, prompt in prompts:
        for coordinate in coordinates:
            for sign in signs:
                cells.append((prompt_id, prompt, coordinate, sign))
                if len(cells) >= max_cells:
                    return cells
    return cells


def _run_zero_edit_check(
    frozen: FrozenModel,
    config: dict[str, Any],
    prompts: list[tuple[str, list[dict[str, str]]]],
) -> dict[str, Any]:
    max_new_tokens = int(config.get("contract_zero_edit_tokens", 32))
    rows = []
    for prompt_id, prompt in prompts[: int(config.get("contract_zero_edit_prompts", 3))]:
        token_ids = _tokenize_for_generation(frozen.tokenizer, prompt, int(config.get("max_prefix_tokens", 512)))
        zero = torch.zeros(frozen.model.config.hidden_size, device=frozen.device)
        base = _base_generate(frozen, token_ids, max_new_tokens=max_new_tokens)
        patched = _generate_with_delta(
            frozen,
            token_ids,
            patch_layer=int(config["patch_layer"]),
            delta=zero,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            apply_once=True,
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "token_match": base == patched,
                "base_new_tokens": len(base) - len(token_ids),
                "patched_new_tokens": len(patched) - len(token_ids),
            }
        )
    return {
        "rows": rows,
        "pass": bool(rows) and all(row["token_match"] for row in rows),
        "checked_prompts": len(rows),
    }


def _run_linearity_checks(
    frozen: FrozenModel,
    basis: dict[str, Any],
    config: dict[str, Any],
    prompts: list[tuple[str, list[dict[str, str]]]],
    eta_by_coordinate: dict[int, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    patch_layer = int(config["patch_layer"])
    readout_layer = int(config["readout_layer"])
    probe_norm = float(config.get("linearization_probe_norm", 1e-3))
    reference_norm_candidates = [
        float(item)
        for item in config.get(
            "contract_reference_norm_candidates",
            [float(config.get("contract_small_edit_norm", probe_norm)), 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
        )
    ]
    min_displacement_norm = float(config.get("contract_min_displacement_norm", 1e-6))
    multipliers = [float(item) for item in config.get("contract_eta_multipliers", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])]
    cells = _validation_cells(config, prompts)
    small_rows = []
    sweep_rows: list[dict[str, Any]] = []
    for prompt_id, prompt, coordinate, sign in cells:
        token_ids = _tokenize_for_generation(frozen.tokenizer, prompt, int(config.get("max_prefix_tokens", 512)))
        small_norm = reference_norm_candidates[-1]
        disp_small = torch.zeros(int(config.get("chart_dim", 32)), dtype=torch.float32, device=frozen.device)
        disp_double = torch.zeros_like(disp_small)
        for candidate_norm in reference_norm_candidates:
            delta_small, _direction, _grad_hidden = _direction_and_delta(
                frozen,
                basis,
                token_ids,
                coordinate=coordinate,
                sign=sign,
                edit_norm=candidate_norm,
                patch_layer=patch_layer,
                readout_layer=readout_layer,
            )
            delta_double, _direction, _grad_hidden = _direction_and_delta(
                frozen,
                basis,
                token_ids,
                coordinate=coordinate,
                sign=sign,
                edit_norm=2.0 * candidate_norm,
                patch_layer=patch_layer,
                readout_layer=readout_layer,
            )
            candidate_disp = _displacement(
                frozen,
                basis,
                token_ids,
                patch_layer=patch_layer,
                readout_layer=readout_layer,
                delta=delta_small,
            )
            candidate_double = _displacement(
                frozen,
                basis,
                token_ids,
                patch_layer=patch_layer,
                readout_layer=readout_layer,
                delta=delta_double,
            )
            if (
                float(candidate_disp.norm().detach().cpu()) >= min_displacement_norm
                and float(candidate_double.norm().detach().cpu()) >= min_displacement_norm
            ):
                small_norm = candidate_norm
                disp_small = candidate_disp
                disp_double = candidate_double
                break
        cosine, r2 = _cosine_and_r2(2.0 * disp_small, disp_double)
        small_rows.append(
            {
                "prompt_id": prompt_id,
                "coordinate": coordinate,
                "sign": sign,
                "edit_norm": small_norm,
                "reference_norm_candidates": reference_norm_candidates,
                "cosine": cosine,
                "r2": r2,
                "small_displacement_norm": float(disp_small.norm().detach().cpu()),
                "double_displacement_norm": float(disp_double.norm().detach().cpu()),
            }
        )
        eta_star = eta_by_coordinate[coordinate]
        for multiplier in multipliers:
            edit_norm = eta_star * multiplier
            delta_eta, _direction, _grad_hidden = _direction_and_delta(
                frozen,
                basis,
                token_ids,
                coordinate=coordinate,
                sign=sign,
                edit_norm=edit_norm,
                patch_layer=patch_layer,
                readout_layer=readout_layer,
            )
            disp_eta = _displacement(
                frozen,
                basis,
                token_ids,
                patch_layer=patch_layer,
                readout_layer=readout_layer,
                delta=delta_eta,
            )
            cosine_eta, r2_eta = _cosine_and_r2(disp_small * (edit_norm / small_norm), disp_eta)
            sweep_rows.append(
                {
                    "prompt_id": prompt_id,
                    "coordinate": coordinate,
                    "sign": sign,
                    "eta_multiplier": multiplier,
                    "eta_star": eta_star,
                    "edit_norm": edit_norm,
                    "cosine": cosine_eta,
                    "r2": r2_eta,
                    "measured_norm": float(disp_eta.norm().detach().cpu()),
                }
            )
    small_cosines = [float(row["cosine"]) for row in small_rows]
    small_r2 = [float(row["r2"]) for row in small_rows]
    small_summary = {
        "rows": small_rows,
        "checked_cells": len(small_rows),
        "median_cosine": _median(small_cosines),
        "p10_cosine": _quantile(small_cosines, 0.1),
        "median_r2": _median(small_r2),
        "pass": bool(small_rows) and _median(small_cosines) >= float(config.get("contract_small_edit_cosine_floor", 0.9)),
    }
    by_multiplier = []
    selected_multiplier = 0.0
    for multiplier in multipliers:
        rows = [row for row in sweep_rows if float(row["eta_multiplier"]) == multiplier]
        cosines = [float(row["cosine"]) for row in rows]
        r2_values = [float(row["r2"]) for row in rows]
        row = {
            "eta_multiplier": multiplier,
            "checked_cells": len(rows),
            "median_cosine": _median(cosines),
            "p10_cosine": _quantile(cosines, 0.1),
            "median_r2": _median(r2_values),
        }
        row["pass"] = bool(rows) and row["median_cosine"] >= float(config.get("contract_eta_cosine_floor", 0.9))
        if row["pass"]:
            selected_multiplier = multiplier
        by_multiplier.append(row)
    eta_summary = {
        "rows": sweep_rows,
        "by_multiplier": by_multiplier,
        "selected_eta_multiplier": selected_multiplier,
        "selected_eta_by_coordinate": {
            str(coordinate): eta * selected_multiplier for coordinate, eta in sorted(eta_by_coordinate.items())
        },
        "pass": selected_multiplier > 0.0,
    }
    return small_summary, eta_summary


def _closed_loop_semantics(config: dict[str, Any], eta_by_coordinate: dict[int, float]) -> dict[str, Any]:
    max_new_tokens = int(config.get("max_completion_tokens", 512))
    chunk = int(config.get("closed_loop_chunk_tokens", 8))
    max_chunks = max(1, math.ceil(max_new_tokens / max(chunk, 1)))
    budget_mode = str(config.get("closed_loop_budget_mode", "l2_per_turn"))
    rows = []
    for coordinate, eta in sorted(eta_by_coordinate.items()):
        if budget_mode == "l2_per_turn":
            chunk_norm = eta / math.sqrt(max_chunks)
        elif budget_mode == "per_chunk":
            chunk_norm = eta
        else:
            chunk_norm = float("nan")
        cumulative_l2 = math.sqrt(max_chunks * chunk_norm * chunk_norm) if math.isfinite(chunk_norm) else float("nan")
        rows.append(
            {
                "coordinate": coordinate,
                "eta_star": eta,
                "max_chunks": max_chunks,
                "chunk_edit_norm": chunk_norm,
                "worst_case_cumulative_l2": cumulative_l2,
                "cumulative_within_eta": bool(math.isfinite(cumulative_l2) and cumulative_l2 <= eta * (1.0 + 1e-6)),
            }
        )
    return {
        "closed_loop_budget_mode": budget_mode,
        "closed_loop_chunk_tokens": chunk,
        "max_new_tokens": max_new_tokens,
        "rows": rows,
        "pass": budget_mode == "l2_per_turn" and bool(rows) and all(row["cumulative_within_eta"] for row in rows),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    run_id = str(
        config.get(
            "run_id",
            f"steering_contract_validation_{_safe_id(config['env_id'])}_{config['formulation']}_{_utc_now().replace(':', '').replace('-', '')}",
        )
    )
    run_dir = Path("runs") / run_id
    frozen = FrozenModel(
        config["model_id"],
        dtype=str(config.get("dtype", "bfloat16")),
        device_map=config.get("device_map", "auto"),
    )
    prompts = _load_base_prompts(Path(config["trajectories_path"]), n_prompts=int(config.get("contract_n_prompts", 3)))
    basis = _fit_stage4_basis(frozen, config)
    etas = _eta_by_coordinate(config)
    selected_coordinates = {int(item) for item in config.get("contract_coordinates", config.get("coordinates", []))}
    if selected_coordinates:
        etas = {coordinate: eta for coordinate, eta in etas.items() if coordinate in selected_coordinates}

    zero_edit = _run_zero_edit_check(frozen, config, prompts)
    small_edit, eta_sweep = _run_linearity_checks(frozen, basis, config, prompts, etas)
    closed_loop = _closed_loop_semantics(config, etas)
    metrics = {
        "zero_edit_pass": zero_edit["pass"],
        "small_edit_median_cosine": small_edit["median_cosine"],
        "small_edit_p10_cosine": small_edit["p10_cosine"],
        "small_edit_median_r2": small_edit["median_r2"],
        "eta_sweep_selected_multiplier": eta_sweep["selected_eta_multiplier"],
        "closed_loop_semantics_pass": closed_loop["pass"],
    }
    gate = "pass" if zero_edit["pass"] and small_edit["pass"] and eta_sweep["pass"] and closed_loop["pass"] else "fail"
    summary = {
        "stage": "steering_contract_validation",
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": config,
        "inputs": {
            "config_path": str(config_path),
            "trajectories_path": config["trajectories_path"],
            "stage3_aggregated_path": config["stage3_aggregated_path"],
        },
        "metrics": metrics,
        "zero_edit_fidelity": zero_edit,
        "small_edit_finite_difference": small_edit,
        "eta_sweep": eta_sweep,
        "closed_loop_semantics": closed_loop,
        "gate": gate,
        "gate_reason": "all four steering-contract fixes passed"
        if gate == "pass"
        else "one or more steering-contract fixes failed; do not run Phase 2 scale-up",
        "next_stage_inputs": {
            "selected_eta_by_coordinate": f"runs/{run_id}/selected_eta_by_coordinate.json"
        }
        if gate == "pass"
        else {},
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "zero_edit_fidelity.json", zero_edit)
    _write_json(run_dir / "small_edit_finite_difference.json", small_edit)
    _write_json(run_dir / "eta_sweep.json", eta_sweep)
    _write_json(run_dir / "closed_loop_semantics.json", closed_loop)
    _write_json(run_dir / "selected_eta_by_coordinate.json", eta_sweep["selected_eta_by_coordinate"])
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": summary["timestamp_utc"],
            "stage": "steering_contract_validation",
            "run_id": run_id,
            "gate_status": gate,
            "key_metrics": json.dumps(metrics, sort_keys=True),
            "evidence_path": f"runs/{run_id}/summary.json",
            "prime_rl_modifications": "none",
        },
    )
    if gate != "pass":
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
