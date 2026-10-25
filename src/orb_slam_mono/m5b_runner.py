"""Isolated full-M4 run with process memory and wall-time observations.

This wrapper does not read ground truth or change the sequence runner's tracker.
Invoke it once per fresh Python process and use a new output directory each time.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import scipy

from .sequence_demo import run_sequence


class _MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def process_memory_bytes() -> dict[str, int]:
    """Windows current-process working set, OS high-water mark, and private bytes."""
    if sys.platform != "win32":
        raise RuntimeError("M5-B process profiling requires Windows GetProcessMemoryInfo")
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_handle = ctypes.windll.kernel32.GetCurrentProcess
    get_handle.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = (ctypes.c_void_p, ctypes.POINTER(_MemoryCounters), ctypes.c_ulong)
    get_memory.restype = ctypes.c_int
    handle = get_handle()
    success = get_memory(
        handle, ctypes.byref(counters), counters.cb)
    if not success:
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return {"working_set_bytes": int(counters.WorkingSetSize),
            "os_peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "private_bytes": int(counters.PrivateUsage)}


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.05,
                 probe=process_memory_bytes):
        if interval_seconds <= 0:
            raise ValueError("Memory sample interval must be positive")
        self.interval_seconds = interval_seconds
        self.probe = probe
        self.samples: list[dict[str, float | int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self.error: BaseException | None = None

    def sample(self) -> dict[str, float | int]:
        observation = {"elapsed_seconds": perf_counter() - self._started,
                       **self.probe()}
        self.samples.append(observation)
        return observation

    def start(self) -> None:
        self._started = perf_counter()
        self.sample()
        def poll() -> None:
            try:
                while not self._stop.wait(self.interval_seconds):
                    self.sample()
            except BaseException as exc:
                self.error = exc
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()

    def finish(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.error is not None:
            raise RuntimeError("Process memory sampler failed") from self.error
        self.sample()

    def report(self) -> dict[str, Any]:
        if not self.samples:
            raise ValueError("No process memory observations")
        first, last = self.samples[0], self.samples[-1]
        os_peak = max(int(row["os_peak_working_set_bytes"]) for row in self.samples)
        sampled_peak = max(int(row["working_set_bytes"]) for row in self.samples)
        return {"method": "Windows GetProcessMemoryInfo; current process working set and OS PeakWorkingSetSize",
                "sample_interval_seconds": self.interval_seconds,
                "sample_count": len(self.samples),
                "start_working_set_bytes": first["working_set_bytes"],
                "final_working_set_bytes": last["working_set_bytes"],
                "sampled_peak_working_set_bytes": sampled_peak,
                "os_peak_working_set_bytes": os_peak,
                "peak_increase_from_start_bytes": max(0, os_peak - int(first["working_set_bytes"])),
                "start_private_bytes": first["private_bytes"],
                "final_private_bytes": last["private_bytes"],
                "sampled_peak_private_bytes": max(int(row["private_bytes"]) for row in self.samples),
                "limitations": "Working set covers the entire Python process, OpenCV/NumPy native allocations and output generation; periodic samples may miss transients, while OS peak working set is a process high-water mark. The start observation follows module imports. Private bytes are sampled, not an OS high-water mark."}


def timing_summary(rows: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    stages = ("total", "orb", "association", "initialization", "pnp", "map_update",
              "local_mapping", "bundle_adjustment")
    distributions = {}
    for stage in stages:
        values = np.asarray([r["timing_ms"][stage] for r in rows
                             if r["timing_ms"].get(stage) is not None], dtype=float)
        distributions[stage] = ({"count": 0, "median_ms": None, "p90_ms": None,
                                 "p95_ms": None, "max_ms": None, "sum_ms": None}
                                if not len(values) else
                                {"count": int(len(values)), "median_ms": float(np.median(values)),
                                 "p90_ms": float(np.percentile(values, 90)),
                                 "p95_ms": float(np.percentile(values, 95)),
                                 "max_ms": float(np.max(values)), "sum_ms": float(np.sum(values))})
    recorded = sum(float(r["timing_ms"].get("total", 0.0)) for r in rows) / 1000.0
    return {"wall_seconds": wall_seconds, "recorded_frame_processing_seconds": recorded,
            "frames_per_second_including_loading_exports": len(rows) / wall_seconds if wall_seconds else None,
            "frames_per_second_recorded_processing_only": len(rows) / recorded if recorded else None,
            "stage_runtime_ms": distributions,
            "wall_scope": "run_sequence invocation: image loading, frame processing, visual exports, maps, plots and files; excludes Python startup/imports and offline evaluation",
            "frame_scope": "tracker.process timing; excludes image loading and output visualization/export"}


def run_profiled(sequence: Path, config_path: Path, output: Path,
                 start: int, end: int, seed: int = 19,
                 sample_interval_seconds: float = 0.05) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sampler = MemorySampler(sample_interval_seconds)
    sampler.start()
    begun = perf_counter()
    try:
        summary = run_sequence("kitti", output, config, sequence, start, end, seed, "full")
    finally:
        wall = perf_counter() - begun
        sampler.finish()
    rows = [json.loads(line) for line in (output / "per_frame.jsonl").read_text(encoding="utf-8").splitlines()]
    if [r["frame_id"] for r in rows] != list(range(start, end + 1)):
        raise ValueError("Saved per-frame output lost original source IDs")
    profile = {"requested_range_inclusive": [start, end], "processed_frames": len(rows),
               "seed_argument": seed, "ablation": "full",
               "seed_policy": "Existing KITTI run_sequence policy: seed argument 19; geometry pose-stability trials use configured fixed seed. No extra OpenCV RNG reset.",
               "fresh_process_pid": os.getpid(),
               "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                               "numpy": np.__version__, "scipy": scipy.__version__,
                               "os": platform.platform(), "logical_cpus": os.cpu_count(),
                               "opencv_threads": cv2.getNumThreads(),
                               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                               "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
               "timing": timing_summary(rows, wall), "memory": sampler.report(),
               "summary": summary}
    (output / "resource_profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    (output / "memory_samples.json").write_text(json.dumps(sampler.samples) + "\n", encoding="utf-8")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=Path("data/sequences/00"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    result = run_profiled(args.sequence, args.config, args.output,
                          args.start, args.end, args.seed)
    print(json.dumps({"range": result["requested_range_inclusive"],
                      "valid_poses": result["summary"]["valid_poses"],
                      "wall_seconds": result["timing"]["wall_seconds"],
                      "peak_working_set_bytes": result["memory"]["os_peak_working_set_bytes"]}))


if __name__ == "__main__":
    main()
