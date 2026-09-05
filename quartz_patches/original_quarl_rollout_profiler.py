from __future__ import annotations

import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


class RolloutProfiler:
    """Exclusive stage timers for Quarl's original agent-side collector."""

    def __init__(self, device: torch.device, agent_id: int) -> None:
        self.device = device
        self.agent_id = agent_id
        self.enabled = bool(os.getenv("QUARL_ROLLOUT_PROFILE_OUTPUT"))
        self.sync_cuda = os.getenv("QUARL_ROLLOUT_PROFILE_SYNC_CUDA", "1") == "1"
        self.output = os.getenv("QUARL_ROLLOUT_PROFILE_OUTPUT", "")
        self.stage_seconds: dict[str, float] = defaultdict(float)
        self.stage_calls: dict[str, int] = defaultdict(int)
        self.iteration = -1

    def begin(self, iteration: int) -> None:
        if not self.enabled:
            return
        self.stage_seconds.clear()
        self.stage_calls.clear()
        self.iteration = iteration
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    @staticmethod
    def start() -> float:
        return time.perf_counter()

    def stop(self, name: str, started: float, cuda: bool = False) -> None:
        if not self.enabled:
            return
        if cuda and self.sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stage_seconds[name] += time.perf_counter() - started
        self.stage_calls[name] += 1

    def finish(
        self,
        rollout_seconds: float,
        transitions: int,
        graph_buffers: list[Any],
    ) -> None:
        if not self.enabled or self.agent_id != 0:
            return
        accounted = sum(self.stage_seconds.values())
        residual = rollout_seconds - accounted
        stage_rows = {
            name: {
                "seconds": seconds,
                "rollout_fraction": seconds / rollout_seconds,
                "microseconds_per_transition": seconds * 1e6 / transitions,
                "calls": self.stage_calls[name],
            }
            for name, seconds in sorted(self.stage_seconds.items())
        }
        input_graphs = [
            {
                "name": buffer.name,
                "input_gate_count": buffer.original_graph.gate_count,
                "input_cx_count": buffer.original_graph.cx_count,
                "input_depth": buffer.original_graph.depth,
                "best_gate_count": buffer.best_graph.gate_count,
                "buffer_size": len(buffer),
            }
            for buffer in graph_buffers
        ]
        record = {
            "schema": "original-quarl-rollout-profile-v1",
            "iteration": self.iteration,
            "agent_id": self.agent_id,
            "hostname": platform.node(),
            "pid": os.getpid(),
            "device": str(self.device),
            "gpu_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "sync_cuda_stage_boundaries": self.sync_cuda,
            "rollout_seconds": rollout_seconds,
            "transitions": transitions,
            "transitions_per_second": transitions / rollout_seconds,
            "accounted_stage_seconds": accounted,
            "accounted_rollout_fraction": accounted / rollout_seconds,
            "residual_seconds": residual,
            "residual_rollout_fraction": residual / rollout_seconds,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            ),
            "peak_cuda_reserved_bytes": (
                torch.cuda.max_memory_reserved(self.device)
                if self.device.type == "cuda"
                else 0
            ),
            "input_graphs": input_graphs,
            "stages": stage_rows,
        }
        output = Path(self.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            "QUARL_ROLLOUT_PROFILE "
            + json.dumps(
                {
                    "output": str(output),
                    "rollout_seconds": rollout_seconds,
                    "transitions": transitions,
                    "accounted_fraction": accounted / rollout_seconds,
                    "residual_fraction": residual / rollout_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
