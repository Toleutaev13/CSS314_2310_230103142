import os
import time
import hashlib
import statistics
from concurrent.futures import ThreadPoolExecutor
TASK_COUNT = 10_000
HASHES_PER_TASK = 500
PAYLOAD = b"A" * 1024 
REPEATS = 2

CORES = os.cpu_count() or 1
POOL_SIZES = list(dict.fromkeys([
    1,
    max(1, CORES // 2),
    CORES,
    CORES * 2,
    CORES * 4,
    CORES * 16,
    256,
]))


def cpu_task(task_id):
    digest = PAYLOAD

    for _ in range(HASHES_PER_TASK):
        digest = hashlib.sha256(PAYLOAD + digest[:0]).digest()
    return digest[0] + task_id % 256


def run_batch(pool_size):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        results = list(executor.map(cpu_task, range(TASK_COUNT)))

    elapsed_ms = (time.perf_counter() - start) * 1000
    if len(results) != TASK_COUNT:
        raise RuntimeError("Not all tasks completed")

    return elapsed_ms


def main():
    print("=" * 88)
    print("TASK 5: THREAD OVERSUBSCRIPTION THRESHOLD")
    print("=" * 88)
    print(f"Logical CPU cores: {CORES}")
    print(f"CPU-bound tasks: {TASK_COUNT:,}")
    print(f"Hashes per task: {HASHES_PER_TASK}")
    print(f"Payload size: {len(PAYLOAD):,} bytes")
    print(f"Repeats per pool size: {REPEATS}")
    print("Timing includes thread-pool creation, scheduling, and shutdown.")

    results = {}

    for pool_size in POOL_SIZES:
        times = []

        for _ in range(REPEATS):
            times.append(run_batch(pool_size))

        results[pool_size] = statistics.median(times)

    baseline = results[1]

    print("\n--- BENCHMARK RESULTS ---")
    print(
        f"{'Pool Size':>10} | {'Execution Time (ms)':>20} | "
        f"{'Speedup':>10} | {'Status'}"
    )
    print("-" * 72)

    previous_time = None
    first_worsening_size = None
    best_time = float("inf")
    best_pool = None

    for pool_size in POOL_SIZES:
        elapsed = results[pool_size]
        speedup = baseline / elapsed if elapsed > 0 else float("inf")

        if pool_size == 1:
            status = "BASELINE"
        elif previous_time is not None and elapsed > previous_time:
            status = "WORSENS vs previous"
            if first_worsening_size is None:
                first_worsening_size = pool_size
        else:
            status = "NO WORSENING vs previous"

        print(
            f"{pool_size:>10} | {elapsed:>20.2f} | "
            f"{speedup:>9.3f}x | {status}"
        )

        if elapsed < best_time:
            best_time = elapsed
            best_pool = pool_size

        previous_time = elapsed

    print("\n--- OVERSUBSCRIPTION ANALYSIS ---")
    print(f"Fastest tested pool size: {best_pool}")
    print(f"Fastest execution time: {best_time:.2f} ms")

    if first_worsening_size is not None:
        print(
            f"First tested pool size that worsened versus the preceding "
            f"tested size: {first_worsening_size}"
        )
    else:
        print(
            "No tested pool size was slower than the immediately preceding "
            "tested size."
        )

    print(
        "Interpretation: the first worsening point is an observed threshold "
        "among tested pool sizes, not proof of a specific OS context-switch "
        "cause. Python's GIL and thread-pool overhead also affect CPU-bound work."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()