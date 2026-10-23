
import os
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Task 7: Arithmetic Intensity vs. Memory Bottlenecks

N = 30_000_000
REPEATS = 3
MAX_CORES = os.cpu_count() or 1

WORKER_COUNTS = sorted(set([1, min(2, MAX_CORES), MAX_CORES]))

print("=" * 78)
print("TASK 7: ARITHMETIC INTENSITY VS. MEMORY BOTTLENECKS")
print("=" * 78)
print(f"Array size: {N:,} elements")
print(f"Data type: float32")
print(f"Logical CPU cores available: {MAX_CORES}")
print(f"Worker counts tested: {WORKER_COUNTS}")
print(f"Repeats per test: {REPEATS}")
print("Times are median kernel execution times.")
print("Thread-pool creation is excluded from timed measurements.")

# Float32 uses 4 bytes per element.
A = np.linspace(0.1, 1.0, N, dtype=np.float32)
B = np.empty_like(A)
TEMP = np.empty_like(A)

# Divide the array into approximately equal chunks.
def make_chunks(workers):
    chunk_size = (N + workers - 1) // workers
    return [
        (start, min(start + chunk_size, N))
        for start in range(0, N, chunk_size)
    ]


def run_kernel(executor, chunks, kernel):
    def process(chunk):
        start, end = chunk
        x = A[start:end]
        y = B[start:end]

        if kernel == "A":
            np.copyto(y, x)
        else:
            temp = TEMP[start:end]
            np.sin(x, out=y)
            np.cos(x, out=temp)
            np.multiply(y, temp, out=y)
            np.tan(x, out=temp)
            np.add(y, temp, out=y)

    list(executor.map(process, chunks))


def benchmark(kernel, workers):
    chunks = make_chunks(workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Warm-up before measuring.
        run_kernel(executor, chunks, kernel)

        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            run_kernel(executor, chunks, kernel)
            times.append(time.perf_counter() - start)

    return statistics.median(times)


results = {"A": {}, "B": {}}

for kernel, description in [
    ("A", "Memory-Bound: B[i] = A[i]"),
    ("B", "Compute-Bound: sin(A[i])*cos(A[i]) + tan(A[i])"),
]:
    print(f"\n--- Kernel {kernel}: {description} ---")
    print(f"{'Cores':>8} | {'Time (s)':>12} | {'Speedup':>10} | {'Efficiency':>12}")
    print("-" * 52)

    baseline = None

    for workers in WORKER_COUNTS:
        elapsed = benchmark(kernel, workers)

        if workers == 1:
            baseline = elapsed

        speedup = baseline / elapsed
        efficiency = speedup / workers

        results[kernel][workers] = {
            "time": elapsed,
            "speedup": speedup,
            "efficiency": efficiency,
        }

        print(
            f"{workers:>8} | {elapsed:>12.4f} | "
            f"{speedup:>10.3f} | {efficiency * 100:>11.2f}%"
        )

        if kernel == "A":
            # Copy reads N elements and writes N elements.
            bandwidth = (2 * N * A.itemsize) / elapsed / 1e9
            print(f"         Effective data bandwidth: {bandwidth:.2f} GB/s")

print("\n--- SCALING ANALYSIS ---")
print("Efficiency = Speedup / Cores")
print("Efficiency is shown as a percentage in the tables.")

if MAX_CORES > 2:
    a2 = results["A"][2]["speedup"]
    am = results["A"][MAX_CORES]["speedup"]
    b2 = results["B"][2]["speedup"]
    bm = results["B"][MAX_CORES]["speedup"]

    a_eff = results["A"][MAX_CORES]["efficiency"]
    b_eff = results["B"][MAX_CORES]["efficiency"]

    print(f"Kernel A speedup: 2 cores = {a2:.3f}, {MAX_CORES} cores = {am:.3f}")
    print(f"Kernel B speedup: 2 cores = {b2:.3f}, {MAX_CORES} cores = {bm:.3f}")

    if a_eff < b_eff and (am - a2) < (bm - b2):
        print(
            "Scaling conclusion: Kernel A (memory copy) shows stronger "
            "scaling saturation than Kernel B."
        )
        print(
            "This is consistent with a RAM bandwidth bottleneck, "
            "but timings alone cannot prove the exact hardware cause."
        )
    elif a_eff < b_eff:
        print(
            "Scaling conclusion: Kernel A has lower efficiency at maximum "
            "cores, consistent with possible memory bandwidth saturation."
        )
    else:
        print(
            "Scaling conclusion: The measurements do not clearly show "
            "Kernel A saturating earlier than Kernel B."
        )
        print(
            "Do not claim confirmed RAM saturation unless the measured "
            "scaling pattern supports it."
        )
else:
    print("Only one or two logical cores are available; comparison is limited.")

print("\nNote: Results depend on the CPU, RAM, operating system, and NumPy build.")
print("=" * 78)
