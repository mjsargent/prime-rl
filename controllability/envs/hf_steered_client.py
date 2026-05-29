from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from verifiers.clients.client import Client
from verifiers.types import Response, ResponseMessage, SamplingArgs, Tool, ToolCall, Usage

from controllability.experiments.stage4_trajectory_generation import (
    _direction_and_delta,
    _generate_with_delta,
    _tokenize_for_generation,
)
from controllability.models.frozen_model import FrozenModel


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    if hasattr(message, "model_dump"):
        data = message.model_dump()
        return dict(data)
    role = getattr(message, "role", "user")
    content = getattr(message, "content", "")
    return {"role": role, "content": content}


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(?P<payload>.*?)\s*</tool_call>", re.DOTALL)


def _parse_tool_calls_from_text(content: str) -> tuple[str, list[ToolCall]]:
    tool_calls: list[ToolCall] = []

    def collect(match: re.Match[str]) -> str:
        payload = match.group("payload").strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return match.group(0)
        if not isinstance(parsed, dict):
            return match.group(0)
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if not isinstance(name, str):
            return match.group(0)
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
        tool_calls.append(ToolCall(id=f"call_{uuid.uuid4().hex}", name=name, arguments=arguments))
        return ""

    remaining = _TOOL_CALL_PATTERN.sub(collect, content).strip()
    return remaining, tool_calls


class HFSteeredChatClient(Client[FrozenModel, list[dict[str, Any]], Response, dict[str, Any]]):
    """
    Local verifiers client backed by FrozenModel patched decoding.

    This enables verifiers' rollout loop to call an HF model with residual edits
    at every assistant generation step. Qwen-style <tool_call> JSON blocks are
    converted to verifiers ToolCall objects so ToolEnv can execute them.
    """

    def __init__(
        self,
        frozen: FrozenModel,
        *,
        basis: dict[str, Any],
        patch_layer: int,
        readout_layer: int,
        coordinate: int | None,
        sign: str | None,
        controller: str,
        edit_norm: float,
        eta_star: float | None = None,
        linearization_probe_norm: float = 1e-3,
        linearization_diagnostic_scope: str = "turn",
        max_prefix_tokens: int = 512,
        closed_loop_chunk_tokens: int = 8,
        closed_loop_budget_mode: str = "l2_per_turn",
        enable_thinking: bool = False,
    ) -> None:
        super().__init__(frozen)
        self.basis = basis
        self.patch_layer = int(patch_layer)
        self.readout_layer = int(readout_layer)
        self.coordinate = coordinate
        self.sign = sign
        self.controller = controller
        self.edit_norm = float(edit_norm)
        self.eta_star = eta_star
        self.linearization_probe_norm = float(linearization_probe_norm)
        self.linearization_diagnostic_scope = linearization_diagnostic_scope
        self.max_prefix_tokens = int(max_prefix_tokens)
        self.closed_loop_chunk_tokens = int(closed_loop_chunk_tokens)
        self.closed_loop_budget_mode = closed_loop_budget_mode
        self.enable_thinking = bool(enable_thinking)
        self.diagnostics: list[dict[str, Any]] = []
        self._response_index = 0

    def setup_client(self, config):  # pragma: no cover - config path unused for local HF client
        raise NotImplementedError("HFSteeredChatClient is constructed from a FrozenModel instance, not ClientConfig")

    async def to_native_tool(self, tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    async def to_native_prompt(self, messages) -> tuple[list[dict[str, Any]], dict]:
        return [_message_to_dict(message) for message in messages], {}

    async def get_native_response(
        self,
        prompt: list[dict[str, Any]],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Response:
        max_new_tokens = int(
            sampling_args.get("max_completion_tokens")
            or sampling_args.get("max_tokens")
            or 128
        )
        temperature = float(sampling_args.get("temperature") or 0.0)
        top_p = sampling_args.get("top_p")
        top_k = sampling_args.get("top_k")
        min_p = sampling_args.get("min_p")
        repetition_penalty = sampling_args.get("repetition_penalty")
        response_index = self._response_index
        self._response_index += 1
        prefix_tokens = _tokenize_for_generation(
            self.client.tokenizer,
            prompt,
            self.max_prefix_tokens,
            tools=tools,
            enable_thinking=self.enable_thinking,
        )
        generated = list(prefix_tokens)
        if self.controller == "none" or self.coordinate is None:
            generated = _generate_with_delta(
                self.client,
                prefix_tokens,
                patch_layer=self.patch_layer,
                delta=torch.zeros(self.client.model.config.hidden_size, device=self.client.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                apply_once=True,
                top_p=float(top_p) if top_p is not None else None,
                top_k=int(top_k) if top_k is not None else None,
                min_p=float(min_p) if min_p is not None else None,
                repetition_penalty=float(repetition_penalty) if repetition_penalty is not None else None,
            )
        else:
            delta, _direction, _grad_hidden = _direction_and_delta(
                self.client,
                self.basis,
                prefix_tokens,
                coordinate=int(self.coordinate),
                sign=str(self.sign or "+"),
                edit_norm=self.edit_norm,
                patch_layer=self.patch_layer,
                readout_layer=self.readout_layer,
            )
            if self.controller == "one_shot":
                self._record_linearization(
                    prefix_tokens,
                    delta=delta,
                    response_index=response_index,
                    chunk_index=0,
                    cumulative_edit_norm=float(delta.float().norm().detach().cpu()),
                )
                generated = _generate_with_delta(
                    self.client,
                    prefix_tokens,
                    patch_layer=self.patch_layer,
                    delta=delta,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    apply_once=True,
                    top_p=float(top_p) if top_p is not None else None,
                    top_k=int(top_k) if top_k is not None else None,
                    min_p=float(min_p) if min_p is not None else None,
                    repetition_penalty=float(repetition_penalty) if repetition_penalty is not None else None,
                )
            elif self.controller == "open_loop":
                self._record_linearization(
                    prefix_tokens,
                    delta=delta,
                    response_index=response_index,
                    chunk_index=0,
                    cumulative_edit_norm=float(delta.float().norm().detach().cpu()),
                )
                generated = _generate_with_delta(
                    self.client,
                    prefix_tokens,
                    patch_layer=self.patch_layer,
                    delta=delta,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    apply_once=False,
                    top_p=float(top_p) if top_p is not None else None,
                    top_k=int(top_k) if top_k is not None else None,
                    min_p=float(min_p) if min_p is not None else None,
                    repetition_penalty=float(repetition_penalty) if repetition_penalty is not None else None,
                )
            elif self.controller == "closed_loop":
                current = list(prefix_tokens)
                remaining = max_new_tokens
                chunk_index = 0
                max_chunks = max(1, math.ceil(max_new_tokens / max(self.closed_loop_chunk_tokens, 1)))
                if self.closed_loop_budget_mode == "l2_per_turn":
                    chunk_edit_norm = self.edit_norm / math.sqrt(max_chunks)
                elif self.closed_loop_budget_mode == "per_chunk":
                    chunk_edit_norm = self.edit_norm
                else:
                    raise ValueError(f"unknown closed_loop_budget_mode: {self.closed_loop_budget_mode}")
                cumulative_edit_norm_sq = 0.0
                while remaining > 0:
                    delta, _direction, _grad_hidden = _direction_and_delta(
                        self.client,
                        self.basis,
                        current,
                        coordinate=int(self.coordinate),
                        sign=str(self.sign or "+"),
                        edit_norm=chunk_edit_norm,
                        patch_layer=self.patch_layer,
                        readout_layer=self.readout_layer,
                    )
                    cumulative_edit_norm_sq += float((delta.float() @ delta.float()).detach().cpu())
                    if self.linearization_diagnostic_scope == "all_chunks" or chunk_index == 0:
                        self._record_linearization(
                            current,
                            delta=delta,
                            response_index=response_index,
                            chunk_index=chunk_index,
                            cumulative_edit_norm=math.sqrt(cumulative_edit_norm_sq),
                        )
                    current = _generate_with_delta(
                        self.client,
                        current,
                        patch_layer=self.patch_layer,
                        delta=delta,
                        max_new_tokens=min(self.closed_loop_chunk_tokens, remaining),
                        temperature=temperature,
                        apply_once=True,
                        top_p=float(top_p) if top_p is not None else None,
                        top_k=int(top_k) if top_k is not None else None,
                        min_p=float(min_p) if min_p is not None else None,
                        repetition_penalty=float(repetition_penalty) if repetition_penalty is not None else None,
                    )
                    remaining = max_new_tokens - (len(current) - len(prefix_tokens))
                    chunk_index += 1
                    if current and current[-1] == self.client.tokenizer.eos_token_id:
                        break
                generated = current
            else:
                raise ValueError(f"unknown controller: {self.controller}")

        new_tokens = generated[len(prefix_tokens) :]
        raw_content = self.client.tokenizer.decode(new_tokens, skip_special_tokens=True)
        content, tool_calls = _parse_tool_calls_from_text(raw_content)
        finish_reason = "tool_calls" if tool_calls else ("length" if len(new_tokens) >= max_new_tokens else "stop")
        return Response(
            id=f"hf-steered-{uuid.uuid4()}",
            created=int(time.time()),
            model=model,
            usage=Usage(
                prompt_tokens=len(prefix_tokens),
                reasoning_tokens=0,
                completion_tokens=len(new_tokens),
                total_tokens=len(prefix_tokens) + len(new_tokens),
            ),
            message=ResponseMessage(
                content=content,
                reasoning_content=None,
                tool_calls=tool_calls or None,
                finish_reason=finish_reason,
                is_truncated=finish_reason == "length",
                tokens=None,
            ),
        )

    async def raise_from_native_response(self, response: Response) -> None:
        return None

    async def from_native_response(self, response: Response) -> Response:
        return response

    async def close(self) -> None:
        return None

    def _chart_residual(self, residual: torch.Tensor) -> torch.Tensor:
        components = torch.as_tensor(self.basis["chart_components"], dtype=torch.float32, device=self.client.device)
        mean = torch.as_tensor(self.basis["chart_mean"], dtype=torch.float32, device=self.client.device)
        return (residual.float() - mean) @ components.T

    def _record_linearization(
        self,
        token_ids: list[int],
        *,
        delta: torch.Tensor,
        response_index: int,
        chunk_index: int,
        cumulative_edit_norm: float,
    ) -> None:
        edit_norm = float(delta.float().norm().detach().cpu())
        if edit_norm <= 0:
            return
        probe_norm = min(self.linearization_probe_norm, max(edit_norm * 0.1, 1e-6))
        probe_delta = delta.float() / delta.float().norm().clamp_min(1e-12) * probe_norm
        baseline = self.client.get_residual(token_ids, layer=self.readout_layer, position=-1).float()
        measured_residual = self.client.patched_forward(
            token_ids,
            patch_layer=self.patch_layer,
            patch_position=-1,
            delta=delta,
            readout_layer=self.readout_layer,
            readout_position=-1,
        ).float()
        probe_residual = self.client.patched_forward(
            token_ids,
            patch_layer=self.patch_layer,
            patch_position=-1,
            delta=probe_delta,
            readout_layer=self.readout_layer,
            readout_position=-1,
        ).float()
        measured = self._chart_residual(measured_residual) - self._chart_residual(baseline)
        predicted = (self._chart_residual(probe_residual) - self._chart_residual(baseline)) * (edit_norm / probe_norm)
        measured_flat = measured.flatten()
        predicted_flat = predicted.flatten()
        denom = float((measured_flat @ measured_flat).detach().cpu())
        err = measured_flat - predicted_flat
        cosine = float(F.cosine_similarity(predicted_flat, measured_flat, dim=0).detach().cpu()) if denom > 0 else 0.0
        r2 = 1.0 - float((err @ err).detach().cpu()) / max(denom, 1e-12)
        self.diagnostics.append(
            {
                "response_index": int(response_index),
                "chunk_index": int(chunk_index),
                "controller": self.controller,
                "coordinate": self.coordinate,
                "sign": self.sign,
                "edit_norm": edit_norm,
                "cumulative_edit_norm": float(cumulative_edit_norm),
                "eta_star": self.eta_star,
                "trust_region_violation": bool(
                    self.eta_star is not None
                    and (
                        edit_norm > float(self.eta_star)
                        or float(cumulative_edit_norm) > float(self.eta_star)
                    )
                ),
                "linearization_probe_norm": probe_norm,
                "linearization_cosine": cosine,
                "linearization_r2": r2,
                "measured_norm": float(measured_flat.norm().detach().cpu()),
                "predicted_norm": float(predicted_flat.norm().detach().cpu()),
            }
        )
