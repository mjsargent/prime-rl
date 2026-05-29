from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from controllability.models.residual_patcher import get_hidden_state, replace_hidden_state, residual_patch_hook


class FrozenModel:
    """
    HF transformers backend for Jacobians, patched decoding, and steered rollouts.

    This backend is intentionally separate from prime-rl's vLLM path, which remains
    the cheap base-policy rollout path.
    """

    def __init__(self, model_id: str, dtype: str = "bfloat16", device_map: str | dict = "auto"):
        self.model_id = model_id
        self.dtype = _resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        kwargs: dict[str, Any] = {"torch_dtype": self.dtype, "trust_remote_code": True}
        if device_map != "cpu":
            kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device_map == "cpu":
            self.model.to("cpu")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.layers = _find_transformer_layers(self.model)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _tokens(self, prefix_tokens: Sequence[int] | Tensor) -> Tensor:
        if isinstance(prefix_tokens, Tensor):
            tokens = prefix_tokens
        else:
            tokens = torch.tensor([list(prefix_tokens)], dtype=torch.long)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        return tokens.to(self.device)

    def get_residual(self, prefix_tokens: Sequence[int] | Tensor, layer: int, position: int = -1) -> Tensor:
        tokens = self._tokens(prefix_tokens)
        sequence = self.get_residual_sequence(tokens, layer=layer)
        return sequence[:, position, :].detach()

    def get_residual_sequence(self, prefix_tokens: Sequence[int] | Tensor, layer: int) -> Tensor:
        tokens = self._tokens(prefix_tokens)
        captured: dict[str, Tensor] = {}

        def hook(_module, _inputs, output) -> None:
            captured["residual"] = get_hidden_state(output)

        handle = self.layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            try:
                self.model(input_ids=tokens, use_cache=False)
            finally:
                handle.remove()
        return captured["residual"].detach()

    def patched_forward(
        self,
        prefix_tokens: Sequence[int] | Tensor,
        patch_layer: int,
        patch_position: int,
        delta: Tensor,
        readout_layer: int,
        readout_position: int = -1,
    ) -> Tensor:
        tokens = self._tokens(prefix_tokens)
        captured: dict[str, Tensor] = {}

        def readout_hook(_module, _inputs, output) -> None:
            captured["readout"] = get_hidden_state(output)

        readout_handle = self.layers[readout_layer].register_forward_hook(readout_hook)
        try:
            with torch.no_grad(), residual_patch_hook(
                self.layers[patch_layer],
                patch_position,
                delta,
            ):
                self.model(input_ids=tokens, use_cache=False)
        finally:
            readout_handle.remove()
        return captured["readout"][:, readout_position, :].detach()

    def vjp_to_residual(
        self,
        prefix_tokens: Sequence[int] | Tensor,
        patch_layer: int,
        patch_position: int,
        readout_layer: int,
        readout_position: int,
        chart_grad: Tensor,
    ) -> Tensor:
        tokens = self._tokens(prefix_tokens)
        hidden_dim = _hidden_size(self.model.config)
        delta = torch.zeros(hidden_dim, device=self.device, dtype=self.dtype, requires_grad=True)
        captured: dict[str, Tensor] = {}

        def patch_hook(_module, _inputs, output):
            hidden = get_hidden_state(output)
            seq_len = hidden.shape[-2]
            pos = patch_position if patch_position >= 0 else seq_len + patch_position
            patched = hidden.clone()
            patched[..., pos, :] = patched[..., pos, :] + delta.to(device=patched.device, dtype=patched.dtype)
            return replace_hidden_state(output, patched)

        def readout_hook(_module, _inputs, output) -> None:
            captured["readout"] = get_hidden_state(output)

        patch_handle = self.layers[patch_layer].register_forward_hook(patch_hook)
        readout_handle = self.layers[readout_layer].register_forward_hook(readout_hook)
        try:
            self.model(input_ids=tokens, use_cache=False)
            readout = captured["readout"][:, readout_position, :].to(chart_grad.device)
            scalar = (readout * chart_grad.reshape_as(readout)).sum()
            grad = torch.autograd.grad(scalar, delta)[0]
        finally:
            patch_handle.remove()
            readout_handle.remove()
            self.model.zero_grad(set_to_none=True)
            captured.clear()
        return grad.detach()

    def patched_decode(
        self,
        prefix_tokens: Sequence[int] | Tensor,
        patch_layer: int,
        patch_position: int,
        delta: Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> list[int]:
        tokens = self._tokens(prefix_tokens)
        generate_kwargs: dict[str, Any] = {
            "input_ids": tokens,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        with torch.inference_mode(), residual_patch_hook(self.layers[patch_layer], patch_position, delta, apply_once=True):
            generated = self.model.generate(
                **generate_kwargs,
            )
        return generated[0].detach().cpu().tolist()


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {dtype!r}")


def _find_transformer_layers(model) -> list[torch.nn.Module]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
    ]
    for root_name, attr_name in candidates:
        root = getattr(model, root_name, None)
        layers = getattr(root, attr_name, None) if root is not None else None
        if layers is not None:
            return list(layers)
    raise ValueError(f"Could not locate transformer layers for {model.__class__.__name__}")


def _hidden_size(config) -> int:
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError(f"Could not infer hidden size from {config.__class__.__name__}")
