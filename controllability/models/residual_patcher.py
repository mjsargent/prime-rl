from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from torch import Tensor, nn


def replace_hidden_state(output: Any, hidden: Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def get_hidden_state(output: Any) -> Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


@contextmanager
def residual_patch_hook(
    layer: nn.Module,
    position: int,
    delta: Tensor,
    *,
    apply_once: bool = False,
) -> Iterator[None]:
    applied = False

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        nonlocal applied
        hidden = get_hidden_state(output)
        if apply_once and applied:
            return output
        seq_len = hidden.shape[-2]
        pos = position if position >= 0 else seq_len + position
        if pos < 0 or pos >= seq_len:
            return output
        patched = hidden.clone()
        patch_delta = delta.to(device=patched.device, dtype=patched.dtype)
        patched[..., pos, :] = patched[..., pos, :] + patch_delta
        applied = True
        return replace_hidden_state(output, patched)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def residual_capture_hook(layer: nn.Module, sink: Callable[[Tensor], None]) -> Iterator[None]:
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        sink(get_hidden_state(output))

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
