from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from controllability.models.frozen_model import FrozenModel


def probe_model(model_id: str, output_path: Path, dtype: str = "bfloat16", device_map: str = "auto") -> None:
    model = FrozenModel(model_id=model_id, dtype=dtype, device_map=device_map)
    tokens = model.tokenizer.encode("Preflight residual hook probe.", return_tensors="pt").to(model.device)
    layers = sorted({0, len(model.layers) // 2, len(model.layers) - 1})
    residuals = {str(layer): tuple(model.get_residual(tokens, layer=layer).shape) for layer in layers}
    payload = {
        "model_id": model_id,
        "num_layers": len(model.layers),
        "dtype": str(model.dtype),
        "residual_shapes": residuals,
        "cuda_available": torch.cuda.is_available(),
        "status": "pass",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()
    probe_model(args.model_id, args.output_path, dtype=args.dtype, device_map=args.device_map)


if __name__ == "__main__":
    main()
