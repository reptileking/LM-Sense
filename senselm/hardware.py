"""Hardware detection and job sizing.

``python -m senselm.hardware [--out docs/HARDWARE.md]`` prints the detected CPU / RAM / disk /
GPU inventory and the sizing it implies. Stages call :func:`detect` and
:func:`resolve_auto` so that config values set to ``auto`` are sized to the machine,
and the resolved values are recorded in each output's provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from senselm.common import REPO_ROOT, setup_logging


@dataclass(frozen=True)
class Hardware:
    cpu_model: str
    logical_cpus: int
    usable_cpus: int
    cpu_flags: list[str]
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    gpus: list[dict[str, Any]]

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _cpu_flags() -> list[str]:
    interesting = {"avx2", "avx512f", "avx512_bf16", "amx_bf16", "amx_int8", "avx512_vnni", "avx_vnni"}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return sorted(interesting & set(line.split(":", 1)[1].split()))
    except OSError:
        pass
    return []


def _gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gpus.append({"index": i, "name": p.name, "memory_gb": round(p.total_memory / 1e9, 1)})
            return gpus
    except Exception:  # torch missing or broken CUDA install: fall back to nvidia-smi
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for i, line in enumerate(out.strip().splitlines()):
                name, mem = [x.strip() for x in line.split(",")]
                gpus.append({"index": i, "name": name, "memory_gb": round(float(mem) / 1024, 1)})
        except (subprocess.CalledProcessError, ValueError):
            pass
    return gpus


def detect(data_dir: str | Path = REPO_ROOT / "data") -> Hardware:
    import psutil

    vm = psutil.virtual_memory()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count() or 1
    return Hardware(
        cpu_model=_cpu_model(),
        logical_cpus=os.cpu_count() or 1,
        usable_cpus=usable,
        cpu_flags=_cpu_flags(),
        ram_total_gb=round(vm.total / 1e9, 1),
        ram_available_gb=round(vm.available / 1e9, 1),
        disk_free_gb=round(shutil.disk_usage(data_dir).free / 1e9, 1),
        gpus=_gpus(),
    )


def sizing(hw: Hardware) -> dict[str, Any]:
    """Default job sizes implied by the hardware. Every value can be overridden in config."""
    if hw.has_gpu:
        mem = max(g["memory_gb"] for g in hw.gpus)
        embed_model = "BAAI/bge-large-en-v1.5" if mem >= 8 else "BAAI/bge-base-en-v1.5"
        embed_batch = 256 if mem >= 16 else 64
        device = "cuda"
    else:
        # CPU only: a 335M-parameter encoder over ~0.5M glosses is several hours on 4 cores;
        # the base model is ~3x faster. The choice is re-made empirically in align (benchmark).
        embed_model = "BAAI/bge-base-en-v1.5"
        embed_batch = 64
        device = "cpu"
    # Parquet batch size: buffered Python dict rows cost ~4 KB each; keep the buffer ~2% of RAM.
    parquet_batch_rows = int(max(10_000, min(100_000, hw.ram_available_gb * 1e9 * 0.02 / 4096)))
    return {
        "device": device,
        "torch_threads": hw.usable_cpus,
        "embed_model": embed_model,
        "embed_batch_size": embed_batch,
        "parquet_batch_rows": parquet_batch_rows,
        "duckdb_memory_limit_gb": max(1, int(hw.ram_total_gb * 0.5)),
    }


def resolve_auto(cfg_value: Any, key: str, hw: Hardware | None = None) -> Any:
    """Return ``cfg_value`` unless it is ``"auto"``, in which case use :func:`sizing`."""
    if cfg_value != "auto":
        return cfg_value
    return sizing(hw or detect())[key]


def to_markdown(hw: Hardware) -> str:
    s = sizing(hw)
    gpu = ", ".join(f"{g['name']} ({g['memory_gb']} GB)" for g in hw.gpus) or "none detected"
    lines = [
        "# Hardware report",
        "",
        "Generated by `python -m senselm.hardware --out docs/HARDWARE.md`.",
        "",
        "| Resource | Detected |",
        "|---|---|",
        f"| CPU | {hw.cpu_model} |",
        f"| Logical CPUs (usable) | {hw.logical_cpus} ({hw.usable_cpus}) |",
        f"| Relevant CPU features | {', '.join(hw.cpu_flags) or '-'} |",
        f"| RAM total / available | {hw.ram_total_gb} GB / {hw.ram_available_gb} GB |",
        f"| Free disk under data/ | {hw.disk_free_gb} GB |",
        f"| GPU | {gpu} |",
        "",
        "## Implied default sizing (all overridable in configs)",
        "",
        "| Setting | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in s.items()],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect hardware and print implied job sizing")
    ap.add_argument("--out", type=Path, default=None, help="optional markdown report path")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    setup_logging()
    hw = detect()
    if args.json:
        print(json.dumps({"hardware": asdict(hw), "sizing": sizing(hw)}, indent=2))
    else:
        print(to_markdown(hw))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(to_markdown(hw), encoding="utf-8")


if __name__ == "__main__":
    main()
