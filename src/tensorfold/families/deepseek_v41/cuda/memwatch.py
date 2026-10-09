"""Memory attribution for the two-Spark serve (cebu, 2026-10-07; TF_DS_MEMWATCH=1).

A thread samples MemAvailable every 0.5 s and logs each new low-water mark (every 0.25 GiB lower) with what holds
memory: torch reserved / allocated, the pinned host cache, this process's RSS. ``trim()`` hands back the pinned host
blocks torch caches (warm-up and load staging; torch never returns them on its own) and glibc's free arenas.
"""

from __future__ import annotations

import os
import threading
import time

import torch


def _avail() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return -1.0


def _rss() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return -1.0


def _pinned() -> float:
    try:
        s = torch.cuda.host_memory_stats()
        return float(s.get("reserved_bytes.current", s.get("allocated_bytes.current", 0)) or 0) / 2**30
    except Exception:
        return -1.0


def _host_parts() -> str:
    """cebu 2026-10-08: where the RSS is: glibc's main heap, other private anonymous memory, shared anonymous memory
    (/dev/zero mappings) and files (libraries), from /proc/self/smaps."""

    heap = anon = shm = files = 0
    name = ""
    try:
        for line in open("/proc/self/smaps"):
            if line[0] in "0123456789abcdef" and "-" in line.split(" ", 1)[0]:
                parts = line.split(None, 5)
                name = parts[5].strip() if len(parts) > 5 else ""
            elif line.startswith("Rss:"):
                kb = int(line.split()[1])
                if name == "[heap]":
                    heap += kb
                elif name.startswith("/dev/zero") or name.startswith("/SYSV") or name.startswith("/dev/shm"):
                    shm += kb
                elif name.startswith("/"):
                    files += kb
                else:
                    anon += kb
    except OSError:
        return ""
    return f" [heap {heap / 2**20:.2f}, anon {anon / 2**20:.2f}, shared {shm / 2**20:.2f}, files {files / 2**20:.2f}]"


def snapshot(rank: int, label: str) -> None:
    parts = _host_parts() if os.environ.get("TF_DS_MEMWATCH", "0") == "1" else ""
    print(f"[tensorfold-mem] rank {rank} {label}: MemAvailable {_avail():.2f} GiB, torch reserved "
          f"{torch.cuda.memory_reserved() / 2**30:.2f} allocated {torch.cuda.memory_allocated() / 2**30:.2f}, "
          f"pinned host {_pinned():.2f}, rss {_rss():.2f}{parts}", flush=True)


def trim(rank: int) -> None:
    if os.environ.get("TF_DS_MEM_TRIM", "1") == "0":
        return
    before = _avail()
    torch.cuda.synchronize()
    # cebu 2026-10-08: the warm-up leaves ~1.4 GiB reserved but unallocated in torch's cache (fragmentation); hand it
    # back once before the allocator ceiling is set (expandable_segments got the same memory but cost ~12 % prefill)
    if os.environ.get("TF_DS_EMPTY_AFTER_WARM", "1") == "1":
        torch.cuda.empty_cache()
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    print(f"[tensorfold-mem] rank {rank} trim: MemAvailable {before:.2f} -> {_avail():.2f} GiB", flush=True)


def start(rank: int) -> None:
    if os.environ.get("TF_DS_MEMWATCH", "0") != "1":
        return
    dev = torch.cuda.current_device()

    def run():
        torch.cuda.set_device(dev)
        low = 1e9
        while True:
            a = _avail()
            if a < low - 0.25:
                low = a
                snapshot(rank, f"LOW-WATER {a:.2f} GiB")
            time.sleep(0.5)

    threading.Thread(target=run, name="tf-memwatch", daemon=True).start()
