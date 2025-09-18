# This logs per-algorithm wall time, CPU time, RSS,
# Python peak memory, arch, etc., into bench_results.csv.

import csv
import functools
import os
import platform
import time
import tracemalloc

import psutil

RESULTS_CSV = os.environ.get("PIPELINE_BENCH_CSV", "bench_results.csv")


def timed(name=None):
    def deco(fn):
        label = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            proc = psutil.Process()
            cpu_start = time.process_time()
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                wall = time.perf_counter() - t0
                cpu = time.process_time() - cpu_start
                cur, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                row = {
                    "algo": label,
                    "wall_s": round(wall, 6),
                    "cpu_s": round(cpu, 6),
                    "rss_MB": round(proc.memory_info().rss / (1024**2), 2),
                    "peak_py_MB": round(peak / (1024**2), 2),
                    "hostname": os.uname().nodename,
                    "arch": platform.machine(),
                    "python": platform.python_version(),
                }
                write_row(row)

        return wrapper

    return deco


def write_row(row):
    fieldnames = [
        "algo",
        "wall_s",
        "cpu_s",
        "rss_MB",
        "peak_py_MB",
        "hostname",
        "arch",
        "python",
    ]
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)
