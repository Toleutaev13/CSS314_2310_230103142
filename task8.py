
import os
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Task 8: Parallel Tile Reduction / Compute Geometry

N = 1024
REPEATS = 3
MAX_CORES = os.cpu_count() or 1
WORKERS = MAX_CORES

rng = np.random.default_rng(42)
A = rng.random((N, N), dtype=np.float32)
B = rng.random((N, N), dtype=np.float32)
C = np.zeros((N, N), dtype=np.float32)

try:
    from threadpoolctl import threadpool_limits
    blas_control = threadpool_limits(limits=1)
except ImportError:
    blas_control = None

def reset_output():
    C.fill(0)


def non_tiled_task(row_range):
    start, end = row_range
    C[start:end, :] = A[start:end, :] @ B


def tiled_task(tile):
    row, col, block_size = tile
    row_end = min(row + block_size, N)
    col_end = min(col + block_size, N)

    C[row:row_end, col:col_end] = (
        A[row:row_end, :] @ B[:, col:col_end]
    )


def make_row_ranges(workers):
    step = (N + workers - 1) // workers
    return [
        (start, min(start + step, N))
        for start in range(0, N, step)
    ]


def make_tiles(block_size):
    return [
        (row, col, block_size)
        for row in range(0, N, block_size)
        for col in range(0, N, block_size)
    ]


def benchmark_non_tiled(executor):
    reset_output()
    executor.map(non_tiled_task, make_row_ranges(WORKERS))
    # Consume the iterator to ensure all tasks finish.
    # executor.map executes tasks as the iterator is consumed.
    return C.copy()


def run_non_tiled(executor):
    list(executor.map(non_tiled_task, make_row_ranges(WORKERS)))


def run_tiled(executor, block_size):
    list(executor.map(tiled_task, make_tiles(block_size)))


def measure(function):
    times = []

    function()  # Warm-up
    for _ in range(REPEATS):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)

    return statistics.median(times)


if blas_control is not None:
    blas_control.__enter__()

try:
    print("=" * 72)
    print("TASK 8: CPU PARALLEL MATRIX MULTIPLICATION")
    print("=" * 72)
    print(f"Matrix size: {N} x {N}")
    print(f"CPU logical cores: {MAX_CORES}")
    print(f"Worker threads: {WORKERS}")
    print(f"Repeats per configuration: {REPEATS}")
    print("Timing excludes thread-pool creation.")

    results = {}

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        configurations = [
            ("Non-tiled", lambda: run_non_tiled(executor)),
            ("Tiled 16x16", lambda: run_tiled(executor, 16)),
            ("Tiled 64x64", lambda: run_tiled(executor, 64)),
        ]

        reference = A @ B

        for name, function in configurations:
            elapsed = measure(function)
            correct = np.allclose(C, reference, rtol=1e-3, atol=1e-3)
            results[name] = elapsed

            print(f"\n{name}")
            print(f"Runtime: {elapsed:.4f} seconds")
            print(f"Correct result: {correct}")

    baseline = results["Non-tiled"]

    print("\n--- PERFORMANCE COMPARISON ---")
    print(f"{'Configuration':<16} | {'Runtime (s)':>12} | {'Speedup':>9}")
    print("-" * 44)

    for name, elapsed in results.items():
        print(f"{name:<16} | {elapsed:>12.4f} | {baseline / elapsed:>8.3f}x")

    fastest = min(results, key=results.get)
    slowest = max(results, key=results.get)

    print("\n--- CACHE LOCALITY ANALYSIS ---")
    print(f"Fastest configuration: {fastest}")
    print(f"Slowest configuration: {slowest}")
    print(
        "Tiling can improve cache locality by working on smaller matrix "
        "regions, but the best block size depends on the CPU and NumPy/BLAS."
    )
    print(
        "Cache-miss latency was not measured directly; runtime is only an "
        "indirect indicator. Hardware performance counters are required "
        "to count actual cache misses."
    )
    print("=" * 72)

finally:
    if blas_control is not None:
        blas_control.__exit__(None, None, None)
