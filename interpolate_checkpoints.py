from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linearly interpolate compatible model checkpoints."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    tuned = torch.load(args.tuned, map_location="cpu", weights_only=False)
    base_state = base["model"]
    tuned_state = tuned["model"]
    if base_state.keys() != tuned_state.keys():
        raise ValueError("checkpoint parameter keys differ")

    mixed_state = {}
    for name, base_value in base_state.items():
        tuned_value = tuned_state[name]
        if base_value.shape != tuned_value.shape or base_value.dtype != tuned_value.dtype:
            raise ValueError(f"incompatible parameter: {name}")
        if base_value.is_floating_point() or base_value.is_complex():
            mixed_state[name] = torch.lerp(base_value, tuned_value, args.alpha)
        else:
            mixed_state[name] = tuned_value.clone()

    output = {
        "model": mixed_state,
        "args": tuned["args"],
        "metrics": {},
        "format": tuned.get("format", base.get("format")),
        "interpolation": {
            "base": str(args.base),
            "tuned": str(args.tuned),
            "alpha": args.alpha,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)


if __name__ == "__main__":
    main()
