import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
STUDENT_ID = 230103142
TOTAL_POINTS = 40_000_000
THREADS = 4
CHUNK_SIZE = 250_000
shared_rng = np.random.default_rng(STUDENT_ID)
rng_lock = threading.Lock()


def count_points(rng, points):
    inside = 0
    remaining = points

    while remaining > 0:
        size = min(CHUNK_SIZE, remaining)

        x = rng.random(size)
        y = rng.random(size)

        inside += np.count_nonzero(x * x + y * y <= 1.0)
        remaining -= size

    return inside


def shared_worker(points):
    inside = 0
    remaining = points

    while remaining > 0:
        size = min(CHUNK_SIZE, remaining)
        with rng_lock:
            x = shared_rng.random(size)
            y = shared_rng.random(size)

        inside += np.count_nonzero(x * x + y * y <= 1.0)
        remaining -= size

    return inside


def independent_worker(args):
    thread_index, points = args
    seed = (STUDENT_ID * 10) + thread_index
    rng = np.random.default_rng(seed)

    return count_points(rng, points)


def run_shared():
    points_per_thread = TOTAL_POINTS // THREADS
    sizes = [points_per_thread] * THREADS
    sizes[-1] += TOTAL_POINTS - sum(sizes)

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        counts = list(executor.map(shared_worker, sizes))

    elapsed_ms = (time.perf_counter() - start) * 1000
    pi_estimate = 4.0 * sum(counts) / TOTAL_POINTS

    return pi_estimate, elapsed_ms


def run_independent():
    points_per_thread = TOTAL_POINTS // THREADS
    tasks = [
        (i, points_per_thread)
        for i in range(THREADS)
    ]
    tasks[-1] = (
        THREADS - 1,
        points_per_thread + TOTAL_POINTS % THREADS
    )

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        counts = list(executor.map(independent_worker, tasks))

    elapsed_ms = (time.perf_counter() - start) * 1000
    pi_estimate = 4.0 * sum(counts) / TOTAL_POINTS

    return pi_estimate, elapsed_ms


def main():
    print("=" * 68)
    print("TASK 2: PSEUDO-RANDOM GENERATOR STATE BOTTLENECK")
    print("=" * 68)
    print(f"Student ID: {STUDENT_ID}")
    print(f"Monte Carlo points: {TOTAL_POINTS:,}")
    print(f"Threads: {THREADS}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print("Starting benchmarks...")

    shared_pi, shared_ms = run_shared()
    independent_pi, independent_ms = run_independent()

    shared_error = abs(shared_pi - math.pi)
    independent_error = abs(independent_pi - math.pi)
    speedup = shared_ms / independent_ms if independent_ms > 0 else float("inf")

    print("\n--- SHARED PRNG (LOCKED) ---")
    print(f"Computed pi: {shared_pi:.10f}")
    print(f"Absolute error: {shared_error:.10f}")
    print(f"Wall time: {shared_ms:.2f} ms")

    print("\n--- INDEPENDENT PRNGs ---")
    print(f"Computed pi: {independent_pi:.10f}")
    print(f"Absolute error: {independent_error:.10f}")
    print(f"Wall time: {independent_ms:.2f} ms")

    print("\n--- COMPARISON ---")
    print(f"Speedup from independent PRNGs: {speedup:.3f}x")

    if speedup > 1:
        print("Independent PRNGs were faster in this benchmark.")
    elif speedup < 1:
        print("Independent PRNGs were slower in this benchmark.")
    else:
        print("Both methods had the same measured runtime.")

    print(
        "Note: The shared lock protects batches of random-number generation. "
        "Results also depend on NumPy, Python, and CPU thread behavior."
    )
    print("=" * 68)


if __name__ == "__main__":
    main()