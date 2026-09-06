from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale the zero-origin residual of a source-topology checkpoint."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.scale <= 1.0:
        parser.error("--scale must be within [0, 1]")

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    keys = [
        key
        for key in checkpoint["model"]
        if key.startswith("source_topology_output.")
    ]
    if not keys:
        raise ValueError("input checkpoint has no source-topology output")
    for key in keys:
        checkpoint["model"][key].mul_(args.scale)
    checkpoint["metrics"] = None
    checkpoint["source_topology_scale"] = args.scale
    checkpoint["derived_from"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved={args.output} scale={args.scale} tensors={len(keys)}")


if __name__ == "__main__":
    main()
