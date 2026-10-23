
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import psutil
STUDENT_ID = "230103142"
N = 8_000

physical_cores = psutil.cpu_count(logical=False)
logical_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
workers = physical_cores or logical_cores
seed = int(STUDENT_ID)
rng = np.random.default_rng(seed)

print("=" * 65)
print("TASK 1: SPATIAL CACHE PROFILING")
print("=" * 65)
print(f"Student ID: {STUDENT_ID}")
print(f"Logical cores: {logical_cores}")
print(f"Physical cores detected: {physical_cores}")
print(f"Parallel workers: {workers}")
print(f"Matrix size: {N} x {N}")
print("Data type: float64")
print("Matrix storage order: C-order (row-major)")
print("Allocating matrix...")

matrix = rng.random((N, N), dtype=np.float64)
print(f"Matrix memory: {matrix.nbytes / (1024 ** 2):.2f} MiB")


def make_ranges(length, parts):
    step = (length + parts - 1) // parts
    return [
        (start, min(start + step, length))
        for start in range(0, length, step)
    ]


def sum_rows(row_range):
    start, end = row_range
    return matrix[start:end, :].sum(dtype=np.float64)


def sum_columns(column_range):
    start, end = column_range
    return matrix[:, start:end].sum(dtype=np.float64)


def benchmark(sum_function, ranges):
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        partial_sums = list(executor.map(sum_function, ranges))

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return sum(partial_sums), elapsed_ms


row_ranges = make_ranges(N, workers)
column_ranges = make_ranges(N, workers)

print("\nRunning Mode A: Row-Major...")
sum_a, time_a = benchmark(sum_rows, row_ranges)

print("Running Mode B: Column-Major / Strided...")
sum_b, time_b = benchmark(sum_columns, column_ranges)

penalty = time_b / time_a if time_a > 0 else float("inf")

print("\n--- RESULTS ---")
print(f"Student ID: {STUDENT_ID}")
print(f"Logical core count: {logical_cores}")
print(f"Workers used: {workers}")
print(f"Mode A (Row-Major): {time_a:.2f} ms")
print(f"Mode B (Column-Major / Strided): {time_b:.2f} ms")
print(f"Cache penalty factor (B / A): {penalty:.3f}x")
print(f"Row-major sum: {sum_a:.6f}")
print(f"Column-major sum: {sum_b:.6f}")
print(f"Sums match: {np.isclose(sum_a, sum_b, rtol=1e-12)}")

if penalty > 1:
    print("Conclusion: Column-major traversal was slower in this test.")
elif penalty < 1:
    print("Conclusion: Column-major traversal was faster in this test.")
else:
    print("Conclusion: Both traversal modes had equal measured runtime.")

print("Note: Runtime differences are consistent with traversal effects,")
print("but timing alone does not directly measure CPU cache misses.")
print("=" * 65)