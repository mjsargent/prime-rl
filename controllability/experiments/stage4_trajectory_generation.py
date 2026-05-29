from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch import Tensor

from controllability.encoders.encoder_registry import get_encoder
from controllability.envs.trajectory_schema import SteeringMetadata, Trajectory, trajectory_to_record
from controllability.experiments.stage1_graph_free_spectra import (
    _generalized_spectrum,
    _regularized_covariance,
)
from controllability.experiments.stage1_layer_pair_sweep import (
    _capture_layer_states,
    _controllability_matrix,
    _fit_pca,
    _jacobian_rows,
    _metric_basis,
    _tokenize,
)
from controllability.experiments.stage1_parametric_eigenfunctions import (
    _gradient_tensor,
    _normalize_metric_matrices,
    _rayleigh_energies,
    _train_parametric_eigenfunctions,
)
from controllability.models.frozen_model import FrozenModel
from controllability.models.residual_patcher import residual_patch_hook
from controllability.reports.audit_table import append_audit_row

Controller = Literal["none", "one_shot", "open_loop", "closed_loop"]


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _safe_id(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").replace(".", "_")


def _clean_prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    prompt: list[dict[str, str]] = []
    seen_first_assistant = False
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            seen_first_assistant = True
            break
        if seen_first_assistant or role not in {"system", "user"} or not isinstance(content, str):
            continue
        item = {"role": role, "content": content}
        if not prompt or prompt[-1] != item:
            prompt.append(item)
    return prompt


def _load_base_prompts(path: Path, *, n_prompts: int) -> list[tuple[str, list[dict[str, str]]]]:
    prompts: dict[str, list[dict[str, str]]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_id = str(record["prompt_id"])
            prompts.setdefault(prompt_id, _clean_prompt_messages(record["messages"]))
            if len(prompts) >= n_prompts:
                break
    return list(prompts.items())


def _tokenize_for_generation(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
) -> list[int]:
    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
            enable_thinking=enable_thinking,
        )
    except Exception:
        try:
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=tools,
            )
        except Exception:
            tokens = tokenizer.encode("\n".join(message["content"] for message in messages))
    if isinstance(tokens, dict) or hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    if isinstance(tokens, Tensor):
        tokens = tokens.flatten().tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    if isinstance(tokens, str):
        tokens = tokenizer.encode(tokens)
    return list(tokens)[-max_length:]


def _sample_training_tokens(tokenizer, trajectories_path: Path, *, sample_size: int, max_length: int) -> list[list[int]]:
    messages = []
    with trajectories_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            messages.append(_clean_prompt_messages(record["messages"]))
            if len(messages) >= sample_size:
                break
    return [_tokenize(tokenizer, prompt, max_length) for prompt in messages if prompt]


def _fit_stage4_basis(frozen: FrozenModel, config: dict[str, Any]) -> dict[str, Any]:
    patch_layer = int(config["patch_layer"])
    readout_layer = int(config["readout_layer"])
    chart_dim = int(config["chart_dim"])
    formulation = str(config.get("formulation", "parametric_b"))
    positions_per_sequence = int(config.get("positions_per_sequence", 8))
    sample_tokens = _sample_training_tokens(
        frozen.tokenizer,
        Path(config["trajectories_path"]),
        sample_size=int(config.get("basis_sample_trajectories", 64)),
        max_length=int(config.get("max_prefix_tokens", 512)),
    )
    layer_states = _capture_layer_states(
        frozen.model,
        frozen.tokenizer,
        frozen.layers,
        [patch_layer, readout_layer],
        sample_tokens,
        positions_per_sequence=positions_per_sequence,
        batch_size=int(config.get("basis_batch_size", 2)),
    )
    if formulation == "koopman_dmd_c":
        chart_states = np.concatenate([layer_states[patch_layer], layer_states[readout_layer]], axis=0)
    else:
        chart_states = layer_states[readout_layer]
    mean, components, _ = _fit_pca(chart_states, chart_dim)
    metric_basis = _metric_basis(layer_states[patch_layer], positions_per_sequence, str(config.get("residual_metric", "cov_h")))
    jacobian_tokens = sample_tokens[: int(config.get("basis_jacobian_states", 16))]
    matrices = []
    z_rows = []
    for token_ids in jacobian_tokens:
        jacobian = _jacobian_rows(
            frozen.model,
            frozen.layers,
            token_ids,
            patch_layer=patch_layer,
            readout_layer=readout_layer,
            chart_components=components,
        )
        matrices.append(_controllability_matrix(jacobian, metric_basis))
        residual = frozen.get_residual(token_ids, layer=readout_layer, position=-1).float().cpu().numpy()[0]
        z_rows.append((residual - mean) @ components.T)
    z = np.asarray(z_rows, dtype=np.float32)

    if formulation == "average_controllability_a":
        m_bar = np.mean(np.stack(matrices, axis=0), axis=0)
        z_all = (layer_states[readout_layer] - mean) @ components.T
        covariance = _regularized_covariance(z_all, float(config.get("metric_ridge", 1e-4)))
        _evals, evecs = _generalized_spectrum(
            m_bar,
            covariance,
            ridge=float(config.get("metric_ridge", 1e-4)),
        )
        return {
            "basis_kind": "linear_chart",
            "basis_source": "average_controllability_generalized_modes",
            "linear_gradients": evecs[:, : int(config["selected_k"])].T.astype(np.float32),
            "chart_mean": mean.astype(np.float32),
            "chart_components": components.astype(np.float32),
            "basis_states": int(len(z)),
        }

    if formulation == "koopman_dmd_c":
        patch_z = (layer_states[patch_layer] - mean) @ components.T
        readout_z = (layer_states[readout_layer] - mean) @ components.T
        xtx = patch_z.T @ patch_z
        scale = float(np.trace(xtx) / max(xtx.shape[0], 1))
        ridge = float(config.get("metric_ridge", 1e-4)) * max(scale, 1e-12)
        operator = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), patch_z.T @ readout_z)
        output_modes, _singular_values, _input_modes = np.linalg.svd(operator, full_matrices=False)
        return {
            "basis_kind": "linear_chart",
            "basis_source": "koopman_dmd_output_singular_modes",
            "linear_gradients": output_modes[:, : int(config["selected_k"])].T.astype(np.float32),
            "chart_mean": mean.astype(np.float32),
            "chart_components": components.astype(np.float32),
            "basis_states": int(len(z)),
        }

    if formulation == "original_graph":
        return {
            "basis_kind": "linear_chart",
            "basis_source": "graph_negative_control_pca_proxy_no_offsample_graph_basis",
            "linear_gradients": np.eye(chart_dim, dtype=np.float32)[: int(config["selected_k"])],
            "chart_mean": mean.astype(np.float32),
            "chart_components": components.astype(np.float32),
            "basis_states": int(len(z)),
        }

    if formulation != "parametric_b":
        raise ValueError(f"Unsupported Stage 4 formulation: {formulation}")

    metrics = _normalize_metric_matrices(np.stack(matrices, axis=0), float(config.get("metric_ridge", 1e-4)))
    device = frozen.device
    model, _history = _train_parametric_eigenfunctions(
        z,
        metrics,
        output_dim=int(config["selected_k"]),
        hidden_dim=int(config.get("hidden_dim", 64)),
        steps=int(config.get("basis_train_steps", 250)),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        ortho_weight=float(config.get("ortho_weight", 10.0)),
        seed=int(config.get("seed", 42)),
        device=device,
    )
    inputs = torch.as_tensor(z, dtype=torch.float32, device=device)
    normalized = ((inputs - model.z_mean) / model.z_std).detach().clone().requires_grad_(True)  # type: ignore[attr-defined]
    outputs = model(normalized)
    gradients = _gradient_tensor(outputs, normalized)
    metric_tensor = torch.as_tensor(metrics, dtype=torch.float32, device=device)
    energies = _rayleigh_energies(outputs, gradients, metric_tensor).detach().cpu().numpy()
    order = np.argsort(energies).astype(int).tolist()
    return {
        "basis_kind": "parametric_eigenfunction",
        "basis_source": "parametric_neural_eigenfunctions",
        "eigen_model": model,
        "coordinate_order": order,
        "chart_mean": mean.astype(np.float32),
        "chart_components": components.astype(np.float32),
        "basis_energies": energies.tolist(),
        "basis_states": int(len(z)),
    }


def _layer_device(layer: torch.nn.Module, *, fallback: torch.device) -> torch.device:
    try:
        return next(layer.parameters()).device
    except StopIteration:
        return fallback


def _grad_hidden_for_coordinate(
    frozen: FrozenModel,
    basis: dict[str, Any],
    token_ids: list[int],
    *,
    coordinate: int,
    readout_layer: int,
) -> Tensor:
    if basis["basis_kind"] == "linear_chart":
        component_device = _layer_device(frozen.layers[readout_layer], fallback=frozen.device)
        components = torch.as_tensor(basis["chart_components"], dtype=torch.float32, device=component_device)
        gradients = torch.as_tensor(basis["linear_gradients"], dtype=torch.float32, device=component_device)
        grad_z = gradients[int(coordinate)].reshape(1, -1)
        return (grad_z @ components).detach()
    residual = frozen.get_residual(token_ids, layer=readout_layer, position=-1).float()
    components = torch.as_tensor(basis["chart_components"], dtype=torch.float32, device=residual.device)
    mean = torch.as_tensor(basis["chart_mean"], dtype=torch.float32, device=residual.device)
    z = (residual - mean) @ components.T
    model = basis["eigen_model"]
    normalized = ((z - model.z_mean) / model.z_std).detach().clone().requires_grad_(True)  # type: ignore[attr-defined]
    output = model(normalized)
    raw_coordinate = int(basis["coordinate_order"][coordinate])
    grad_norm = torch.autograd.grad(output[0, raw_coordinate], normalized)[0]
    grad_z = grad_norm / model.z_std  # type: ignore[attr-defined]
    grad_hidden = grad_z @ components
    return grad_hidden.reshape_as(residual).detach()


def _direction_and_delta(
    frozen: FrozenModel,
    basis: dict[str, Any],
    token_ids: list[int],
    *,
    coordinate: int,
    sign: str,
    edit_norm: float,
    patch_layer: int,
    readout_layer: int,
) -> tuple[Tensor, Tensor, Tensor]:
    grad_hidden = _grad_hidden_for_coordinate(frozen, basis, token_ids, coordinate=coordinate, readout_layer=readout_layer)
    raw_direction = frozen.vjp_to_residual(
        token_ids,
        patch_layer=patch_layer,
        patch_position=-1,
        readout_layer=readout_layer,
        readout_position=-1,
        chart_grad=grad_hidden,
    ).float()
    direction = raw_direction / raw_direction.norm().clamp_min(1e-12)
    if sign == "-":
        direction = -direction
    delta = direction * float(edit_norm)
    return delta, direction, grad_hidden


def _generate_with_delta(
    frozen: FrozenModel,
    token_ids: list[int],
    *,
    patch_layer: int,
    delta: Tensor,
    max_new_tokens: int,
    temperature: float,
    apply_once: bool,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    repetition_penalty: float | None = None,
    use_cache: bool = True,
) -> list[int]:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=frozen.device)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": frozen.tokenizer.eos_token_id,
        "use_cache": bool(use_cache),
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
        if min_p is not None and float(min_p) > 0.0:
            kwargs["min_p"] = float(min_p)
    if repetition_penalty is not None and float(repetition_penalty) != 1.0:
        kwargs["repetition_penalty"] = float(repetition_penalty)
    with torch.inference_mode(), residual_patch_hook(frozen.layers[patch_layer], -1, delta, apply_once=apply_once):
        generated = frozen.model.generate(**kwargs)
    return generated[0].detach().cpu().tolist()


def _diagnose_edit(
    frozen: FrozenModel,
    token_ids: list[int],
    *,
    patch_layer: int,
    readout_layer: int,
    delta: Tensor,
    direction: Tensor,
    grad_hidden: Tensor,
    edit_norm: float,
    eta_star: float,
) -> dict[str, Any]:
    baseline = frozen.get_residual(token_ids, layer=readout_layer, position=-1).float()
    patched = frozen.patched_forward(
        token_ids,
        patch_layer=patch_layer,
        patch_position=-1,
        delta=delta,
        readout_layer=readout_layer,
        readout_position=-1,
    ).float()
    measured = float(((patched - baseline) * grad_hidden).sum().detach().cpu())
    predicted = float((delta.float() * direction.float()).sum().detach().cpu())
    return {
        "edit_norm": float(edit_norm),
        "eta_star": float(eta_star),
        "trust_region_violation": bool(edit_norm > eta_star),
        "predicted_displacement": predicted,
        "measured_displacement": measured,
    }


def _decode_completion(frozen: FrozenModel, prefix_tokens: list[int], generated_tokens: list[int]) -> tuple[str, int]:
    new_tokens = generated_tokens[len(prefix_tokens) :]
    return frozen.tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens)


def _coherence(completion: str, max_new_tokens: int, generated_length: int) -> bool:
    text = completion.strip()
    return bool(text) and "\ufffd" not in text and generated_length <= max_new_tokens


def _make_trajectory(
    *,
    env_id: str,
    model_id: str,
    prompt_id: str,
    seed: int,
    prompt_messages: list[dict[str, str]],
    completion: str,
    controller: Controller,
    coordinate: int | None,
    sign: str | None,
    edit_norm: float,
    diagnostics: list[dict[str, Any]],
    generated_length: int,
    wall_clock_seconds: float,
) -> Trajectory:
    git_hash = _git_hash()
    messages = [dict(message) for message in prompt_messages]
    messages.append({"role": "assistant", "content": completion})
    return Trajectory(
        env_id=env_id,
        model_id=model_id,
        prompt_id=prompt_id,
        seed=seed,
        messages=messages,
        tool_calls=[],
        step_metadata=[
            {
                "controller": controller,
                "coordinate": coordinate,
                "generated_tokens": generated_length,
                "diagnostics": diagnostics,
            }
        ],
        reward=0.0,
        success=False,
        steering=SteeringMetadata(
            controller=controller,
            coordinate=coordinate,
            sign=sign if sign in {"+", "-"} else None,
            edit_norm=float(edit_norm),
            trust_region_violations=sum(1 for item in diagnostics if item["trust_region_violation"]),
            applied_edits=diagnostics,
        ),
        git_hash=git_hash,
        prime_rl_git_hash=git_hash,
        timestamp_utc=_utc_now(),
        wall_clock_seconds=wall_clock_seconds,
    )


def _jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = _load_base_prompts(Path(config["trajectories_path"]), n_prompts=int(config["n_prompts"]))
    jobs = []
    job_id = 0
    for prompt_id, prompt_messages in prompts:
        for seed in config["seeds"]:
            jobs.append(
                {
                    "job_id": f"job_{job_id:06d}",
                    "prompt_id": prompt_id,
                    "prompt_messages": prompt_messages,
                    "seed": int(seed),
                    "controller": "none",
                    "coordinate": None,
                    "sign": None,
                    "edit_norm": 0.0,
                    "eta_multiplier": 0.0,
                }
            )
            job_id += 1
            for coordinate in config["coordinates"]:
                for controller in config["controllers"]:
                    for sign in config["signs"]:
                        for eta_multiplier in config["eta_multipliers"]:
                            jobs.append(
                                {
                                    "job_id": f"job_{job_id:06d}",
                                    "prompt_id": prompt_id,
                                    "prompt_messages": prompt_messages,
                                    "seed": int(seed),
                                    "controller": controller,
                                    "coordinate": int(coordinate),
                                    "sign": sign,
                                    "eta_multiplier": float(eta_multiplier),
                                }
                            )
                            job_id += 1
    return jobs


def _init_jobs_db(path: Path, run_id: str, jobs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            "create table if not exists stage4_jobs (run_id text, job_id text, status text, payload text, updated_utc text, primary key(run_id, job_id))"
        )
        for job in jobs:
            con.execute(
                "insert or replace into stage4_jobs values (?, ?, ?, ?, ?)",
                (run_id, job["job_id"], "pending", json.dumps(job, sort_keys=True), _utc_now()),
            )


def _update_job(path: Path, run_id: str, job_id: str, status: str) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            "update stage4_jobs set status=?, updated_utc=? where run_id=? and job_id=?",
            (status, _utc_now(), run_id, job_id),
        )


def _cache_encoded(cache_path: Path, records: list[dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_path) as con:
        con.execute(
            "create table if not exists encoded_trajectories (trajectory_id text primary key, encoded_json text, updated_utc text)"
        )
        for record in records:
            con.execute(
                "insert or replace into encoded_trajectories values (?, ?, ?)",
                (record["trajectory_id"], json.dumps(record, sort_keys=True), _utc_now()),
            )


def _write_encoded(
    run_dir: Path,
    trajectories: list[Trajectory],
    *,
    encoder_name: str = "fixed_feature_v1",
    encoder_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    def completion_tokens(traj: Trajectory) -> int:
        total = 0
        for step in traj.step_metadata:
            if not isinstance(step, dict):
                continue
            if "generated_tokens" in step:
                total += int(step["generated_tokens"])
                continue
            usage = step.get("usage") or {}
            if isinstance(usage, dict):
                total += int(usage.get("completion_tokens", 0))
        return total

    records = []
    by_env: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        by_env.setdefault(traj.env_id, []).append(traj)
    for env_id, env_trajs in by_env.items():
        encoder = get_encoder(env_id, encoder_name=encoder_name, encoder_config=encoder_config)
        for traj, encoded in zip(env_trajs, encoder.encode_batch(env_trajs), strict=True):
            steering = traj.steering
            records.append(
                {
                    "trajectory_id": encoded.trajectory_id,
                    "env_id": traj.env_id,
                    "model_id": traj.model_id,
                    "controller": steering.controller if steering else "none",
                    "coordinate": steering.coordinate if steering else None,
                    "sign": steering.sign if steering else None,
                    "prompt_id": traj.prompt_id,
                    "seed": traj.seed,
                    "behavior_features": encoded.behavior_features.tolist(),
                    "quality": encoded.quality,
                    "coherence": float(_coherence(str(traj.messages[-1]["content"]), 10**9, completion_tokens(traj))),
                    "encoder_version": encoded.encoder_version,
                }
            )
    pq.write_table(pa.Table.from_pylist(records), run_dir / "encoded.parquet")
    return records


def _cell_key(row: dict[str, Any]) -> str:
    controller = row.get("controller")
    coordinate = row.get("coordinate")
    sign = row.get("sign")
    if controller == "none":
        return "none"
    return f"{controller}|k={coordinate}|sign={sign}"


def _expected_steering_cell_keys(config: dict[str, Any]) -> set[str]:
    controllers = config.get("required_controllers", config.get("controllers", []))
    signs = config.get("required_signs", config.get("signs", []))
    coordinates = config.get("required_coordinates", config.get("coordinates", []))
    return {
        f"{controller}|k={int(coordinate)}|sign={sign}"
        for controller in controllers
        for coordinate in coordinates
        for sign in signs
    }


def _cell_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = _cell_key(row)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _completeness_report(config: dict[str, Any], jobs: list[dict[str, Any]], trajectories: list[Trajectory], failed: list[dict[str, Any]]) -> dict[str, Any]:
    expected = _expected_steering_cell_keys(config)
    configured = {
        _cell_key(job)
        for job in jobs
        if job.get("controller") != "none"
    }
    trajectory_rows = []
    for traj in trajectories:
        steering = traj.steering
        trajectory_rows.append(
            {
                "controller": steering.controller if steering else "none",
                "coordinate": steering.coordinate if steering else None,
                "sign": steering.sign if steering else None,
            }
        )
    failed_rows = [
        {
            "controller": row.get("controller"),
            "coordinate": row.get("coordinate"),
            "sign": row.get("sign"),
        }
        for row in failed
    ]
    produced = {_cell_key(row) for row in trajectory_rows if row.get("controller") != "none"}
    failed_keys = {_cell_key(row) for row in failed_rows if row.get("controller") != "none"}
    return {
        "require_complete_steering_grid": bool(config.get("require_complete_steering_grid", False)),
        "required_controllers": list(config.get("required_controllers", config.get("controllers", []))),
        "required_signs": list(config.get("required_signs", config.get("signs", []))),
        "required_coordinates": [int(value) for value in config.get("required_coordinates", config.get("coordinates", []))],
        "expected_steering_cells": sorted(expected),
        "configured_steering_cells": sorted(configured),
        "produced_steering_cells": sorted(produced),
        "failed_steering_cells": sorted(failed_keys),
        "missing_configured_cells": sorted(expected - configured),
        "missing_produced_cells": sorted(expected - produced),
        "extra_configured_cells": sorted(configured - expected),
        "job_cell_counts": _cell_counts(jobs),
        "trajectory_cell_counts": _cell_counts(trajectory_rows),
        "failed_cell_counts": _cell_counts(failed_rows),
        "configured_complete": expected <= configured,
        "produced_complete": expected <= produced,
    }


def _aggregate_diagnostics(run_dir: Path, diagnostics: list[dict[str, Any]], trajectories: list[Trajectory]) -> dict[str, Any]:
    by_norm: dict[float, list[dict[str, Any]]] = {}
    for item in diagnostics:
        by_norm.setdefault(float(item["edit_norm"]), []).append(item)
    trust_rows = [
        {
            "edit_norm": norm,
            "num_edits": len(items),
            "violation_rate": sum(1 for item in items if item["trust_region_violation"]) / len(items),
        }
        for norm, items in sorted(by_norm.items())
        if norm > 0
    ]
    length_rows = []
    coherence_rows = []
    for traj in trajectories:
        meta = traj.step_metadata[0]
        steering = traj.steering
        controller = steering.controller if steering else "none"
        edit_norm = steering.edit_norm if steering else 0.0
        coherent = _coherence(str(traj.messages[-1]["content"]), 10**9, int(meta["generated_tokens"]))
        length_rows.append(
            {
                "controller": controller,
                "edit_norm": edit_norm,
                "generated_tokens": int(meta["generated_tokens"]),
            }
        )
        coherence_rows.append({"controller": controller, "edit_norm": edit_norm, "coherent": bool(coherent)})

    coherence_by_norm: dict[float, list[bool]] = {}
    for row in coherence_rows:
        coherence_by_norm.setdefault(float(row["edit_norm"]), []).append(bool(row["coherent"]))
    coherence_summary = [
        {
            "edit_norm": norm,
            "num_trajectories": len(values),
            "coherence_rate": sum(values) / len(values),
        }
        for norm, values in sorted(coherence_by_norm.items())
    ]

    _write_json(run_dir / "trust_region_violation_rate_by_edit_norm.json", {"rows": trust_rows})
    _write_json(run_dir / "trajectory_length_distribution_by_controller.json", {"rows": length_rows})
    _write_json(run_dir / "coherence_rate_by_edit_norm.json", {"rows": coherence_summary})
    pq.write_table(pa.Table.from_pylist(diagnostics), run_dir / "realized_vs_predicted_displacement.parquet")
    return {
        "trust_region_violation_rate_mean": float(np.mean([row["violation_rate"] for row in trust_rows])) if trust_rows else 0.0,
        "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in length_rows])) if length_rows else 0.0,
        "coherence_rate": float(np.mean([row["coherent"] for row in coherence_rows])) if coherence_rows else 0.0,
        "num_displacement_diagnostics": len(diagnostics),
    }


def run_stage4(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    run_id = config.get("run_id") or f"stage4_v2_{_safe_id(config['model_id'])}_{_safe_id(config['env_id'])}_{_utc_now().replace(':', '')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectories").mkdir(exist_ok=True)
    worker_id = f"{run_id}_worker_0"
    heartbeat = Path("runs/heartbeats") / f"{worker_id}.json"
    jobs_db = Path(config.get("jobs_db", "runs/jobs.db"))

    stage3 = _load_json(config["stage3_aggregated_path"])["rows"]
    formulation = str(config.get("formulation", "parametric_b"))
    eta_by_coordinate = {
        int(row["coordinate"]): float(row["eta_star_mean"])
        for row in stage3
        if row["formulation"] == formulation
    }
    jobs = _jobs(config)
    _init_jobs_db(jobs_db, run_id, jobs)

    frozen = FrozenModel(
        config["model_id"],
        dtype=str(config.get("dtype", "bfloat16")),
        device_map=config.get("device_map", "auto"),
    )
    basis = _fit_stage4_basis(frozen, config)
    trajectories: list[Trajectory] = []
    failed: list[dict[str, Any]] = []
    displacement_diagnostics: list[dict[str, Any]] = []
    max_new_tokens = int(config.get("max_new_tokens", 32))
    temperature = float(config.get("temperature", 0.0))
    top_p = float(config["top_p"]) if "top_p" in config else None
    top_k = int(config["top_k"]) if "top_k" in config else None
    min_p = float(config["min_p"]) if "min_p" in config else None
    repetition_penalty = float(config["repetition_penalty"]) if "repetition_penalty" in config else None
    checkpoint_interval = int(config.get("checkpoint_interval", 100))

    started = time.perf_counter()
    for idx, job in enumerate(jobs, start=1):
        _write_json(heartbeat, {"worker_id": worker_id, "run_id": run_id, "job_id": job["job_id"], "timestamp_utc": _utc_now()})
        _update_job(jobs_db, run_id, job["job_id"], "running")
        try:
            torch.manual_seed(int(job["seed"]))
            prefix_tokens = _tokenize_for_generation(
                frozen.tokenizer,
                job["prompt_messages"],
                int(config.get("max_prefix_tokens", 512)),
                enable_thinking=bool(config.get("enable_thinking", False)),
            )
            controller: Controller = job["controller"]
            diagnostics = []
            generated = prefix_tokens
            edit_norm = 0.0
            if controller == "none":
                generated = _generate_with_delta(
                    frozen,
                    prefix_tokens,
                    patch_layer=int(config["patch_layer"]),
                    delta=torch.zeros(frozen.model.config.hidden_size, device=frozen.device),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    apply_once=True,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                )
            else:
                eta_star = eta_by_coordinate[int(job["coordinate"])]
                edit_norm = float(job["eta_multiplier"]) * eta_star
                delta, direction, grad_hidden = _direction_and_delta(
                    frozen,
                    basis,
                    prefix_tokens,
                    coordinate=int(job["coordinate"]),
                    sign=str(job["sign"]),
                    edit_norm=edit_norm,
                    patch_layer=int(config["patch_layer"]),
                    readout_layer=int(config["readout_layer"]),
                )
                diag = _diagnose_edit(
                    frozen,
                    prefix_tokens,
                    patch_layer=int(config["patch_layer"]),
                    readout_layer=int(config["readout_layer"]),
                    delta=delta,
                    direction=direction,
                    grad_hidden=grad_hidden,
                    edit_norm=edit_norm,
                    eta_star=eta_star,
                )
                diagnostics.append({**diag, "job_id": job["job_id"], "controller": controller, "coordinate": job["coordinate"], "sign": job["sign"]})
                if controller == "one_shot":
                    generated = _generate_with_delta(
                        frozen,
                        prefix_tokens,
                        patch_layer=int(config["patch_layer"]),
                        delta=delta,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        apply_once=True,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                    )
                elif controller == "open_loop":
                    generated = _generate_with_delta(
                        frozen,
                        prefix_tokens,
                        patch_layer=int(config["patch_layer"]),
                        delta=delta,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        apply_once=False,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                    )
                elif controller == "closed_loop":
                    current = list(prefix_tokens)
                    remaining = max_new_tokens
                    chunk = int(config.get("closed_loop_chunk_tokens", 8))
                    max_chunks = max(1, math.ceil(max_new_tokens / max(chunk, 1)))
                    budget_mode = str(config.get("closed_loop_budget_mode", "l2_per_turn"))
                    if budget_mode == "l2_per_turn":
                        chunk_edit_norm = edit_norm / math.sqrt(max_chunks)
                    elif budget_mode == "per_chunk":
                        chunk_edit_norm = edit_norm
                    else:
                        raise ValueError(f"unknown closed_loop_budget_mode: {budget_mode}")
                    while remaining > 0:
                        delta, direction, grad_hidden = _direction_and_delta(
                            frozen,
                            basis,
                            current,
                            coordinate=int(job["coordinate"]),
                            sign=str(job["sign"]),
                            edit_norm=chunk_edit_norm,
                            patch_layer=int(config["patch_layer"]),
                            readout_layer=int(config["readout_layer"]),
                        )
                        current = _generate_with_delta(
                            frozen,
                            current,
                            patch_layer=int(config["patch_layer"]),
                            delta=delta,
                            max_new_tokens=min(chunk, remaining),
                            temperature=temperature,
                            apply_once=True,
                            top_p=top_p,
                            top_k=top_k,
                            min_p=min_p,
                            repetition_penalty=repetition_penalty,
                        )
                        remaining = max_new_tokens - (len(current) - len(prefix_tokens))
                        if current and current[-1] == frozen.tokenizer.eos_token_id:
                            break
                    generated = current
                else:
                    raise ValueError(f"unknown controller {controller}")
            completion, generated_length = _decode_completion(frozen, prefix_tokens, generated)
            traj = _make_trajectory(
                env_id=config["env_id"],
                model_id=config["model_id"],
                prompt_id=f"{job['prompt_id']}:{job['job_id']}",
                seed=int(job["seed"]),
                prompt_messages=job["prompt_messages"],
                completion=completion,
                controller=controller,
                coordinate=job["coordinate"],
                sign=job["sign"],
                edit_norm=edit_norm,
                diagnostics=diagnostics,
                generated_length=generated_length,
                wall_clock_seconds=time.perf_counter() - started,
            )
            trajectories.append(traj)
            displacement_diagnostics.extend(diagnostics)
            _append_jsonl(run_dir / "trajectories" / f"{job['job_id']}.jsonl", trajectory_to_record(traj))
            _update_job(jobs_db, run_id, job["job_id"], "done")
        except Exception as exc:
            row = {
                "job_id": job["job_id"],
                "env_id": config["env_id"],
                "model_id": config["model_id"],
                "prompt_id": job["prompt_id"],
                "seed": job["seed"],
                "controller": job["controller"],
                "coordinate": job["coordinate"],
                "sign": job["sign"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failed.append(row)
            _append_jsonl(run_dir / "failed_jobs.jsonl", row)
            _update_job(jobs_db, run_id, job["job_id"], "failed")
        if idx % checkpoint_interval == 0:
            _write_json(run_dir / f"checkpoint_{idx:06d}.json", {"completed": len(trajectories), "failed": len(failed)})

    (run_dir / "failed_jobs.jsonl").touch(exist_ok=True)
    encoded_records = _write_encoded(
        run_dir,
        trajectories,
        encoder_name=str(config.get("encoder_name", "fixed_feature_v1")),
        encoder_config=config.get("encoder_config") or {},
    )
    _cache_encoded(Path(config["encoder_cache_path"]), encoded_records)
    diagnostics_metrics = _aggregate_diagnostics(run_dir, displacement_diagnostics, trajectories)
    completeness = _completeness_report(config, jobs, trajectories, failed)
    _write_json(run_dir / "controller_sign_coordinate_completeness.json", completeness)
    summary_metrics = {
        "num_jobs": len(jobs),
        "num_trajectories": len(trajectories),
        "num_failed": len(failed),
        "controllers": sorted({job["controller"] for job in jobs}),
        "signs": sorted({str(job["sign"]) for job in jobs if job["sign"] is not None}),
        "coordinates": [int(v) for v in config["coordinates"]],
        "formulation": formulation,
        "basis_source": str(basis.get("basis_source", "unknown")),
        "encoder_name": str(config.get("encoder_name", "fixed_feature_v1")),
        "configured_steering_grid_complete": bool(completeness["configured_complete"]),
        "produced_steering_grid_complete": bool(completeness["produced_complete"]),
        "missing_configured_steering_cells": completeness["missing_configured_cells"],
        "missing_produced_steering_cells": completeness["missing_produced_cells"],
        **diagnostics_metrics,
    }
    all_accounted = len(trajectories) + len(failed) == len(jobs)
    no_failures = len(failed) == 0
    complete_required_grid = (not completeness["require_complete_steering_grid"]) or (
        completeness["configured_complete"] and completeness["produced_complete"]
    )
    gate = "pass" if all_accounted and no_failures and complete_required_grid else "fail"
    summary = {
        "stage": "stage4_trajectory_generation",
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "git_hash": _git_hash(),
        "prime_rl_git_hash": _git_hash(),
        "prime_rl_modifications": [],
        "config": config,
        "inputs": {
            "stage1_locked_config": config["stage1_locked_config"],
            "stage2_baselines": config["stage2_baselines"],
            "stage3_aggregated": config["stage3_aggregated_path"],
        },
        "metrics": summary_metrics,
        "gate": gate,
        "gate_reason": "all Stage 4 v2 job cells produced trajectories"
        if gate == "pass"
        else "Stage 4 v2 requires all cells to produce trajectories and, when requested, the complete controller/sign/coordinate grid",
        "next_stage_inputs": {
            "encoded": f"runs/{run_id}/encoded.parquet",
            "trajectories_dir": f"runs/{run_id}/trajectories",
            "failed_jobs": f"runs/{run_id}/failed_jobs.jsonl",
            "completeness": f"runs/{run_id}/controller_sign_coordinate_completeness.json",
        },
        "limitations": [
            "Stage 4 v2 uses HF patched decoding from environment prompts; it does not yet execute tool-use or verifier-scored environment interaction.",
            "Reward and success are unset for generated steered trajectories; coherence is a text-level completion sanity signal.",
            "The original_graph formulation uses a PCA-proxy off-sample basis because the failed graph estimator did not produce reusable off-sample eigenfunctions."
            if formulation == "original_graph"
            else "No formulation-specific limitation beyond the reduced HF patched-decoding pilot.",
        ],
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "manifest.json", {"config_path": str(config_path), "outputs": sorted(p.name for p in run_dir.iterdir())})
    append_audit_row(
        Path("runs/audit_table.md"),
        {
            "timestamp_utc": summary["timestamp_utc"],
            "stage": "stage4_trajectory_generation",
            "run_id": run_id,
            "gate_status": gate,
            "key_metrics": json.dumps(summary_metrics, sort_keys=True),
            "evidence_path": f"runs/{run_id}/summary.json",
            "prime_rl_modifications": "none",
        },
    )
    if gate != "pass":
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_stage4(args.config)


if __name__ == "__main__":
    main()
