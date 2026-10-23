
import os
import time
import statistics
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

STUDENT_ID = 230103142
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
REPEATS = 3
A = 2.5
B = 1.5


def parallel_chunk(args):
    x_name, y_name, start, end, a, b = args

    x_shm = shared_memory.SharedMemory(name=x_name)
    y_shm = shared_memory.SharedMemory(name=y_name)

    try:
        x = x_shm.buf.cast("d")
        y = y_shm.buf.cast("d")

        for i in range(start, end):
            y[i] = a * x[i] + b

        x.release()
        y.release()
    finally:
        x_shm.close()
        y_shm.close()


def serial_operation(x, y, n):
    start = time.perf_counter()

    for i in range(n):
        y[i] = A * x[i] + B

    return (time.perf_counter() - start) * 1000


def parallel_operation(x_shm, y_shm, n, workers):
    chunk_size = (n + workers - 1) // workers

    tasks = [
        (
            x_shm.name,
            y_shm.name,
            start,
            min(start + chunk_size, n),
            A,
            B
        )
        for start in range(0, n, chunk_size)
    ]

    start_time = time.perf_counter()

    # Pool startup and shutdown are included in the measurement.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(parallel_chunk, tasks))

    return (time.perf_counter() - start_time) * 1000


def benchmark_size(n, workers):
    x_shm = shared_memory.SharedMemory(create=True, size=n * 8)
    y_shm = shared_memory.SharedMemory(create=True, size=n * 8)

    try:
        x = x_shm.buf.cast("d")
        y = y_shm.buf.cast("d")

        for i in range(n):
            x[i] = i * 0.001

        serial_times = []
        parallel_times = []

        for _ in range(REPEATS):
            serial_times.append(serial_operation(x, y, n))
            parallel_times.append(
                parallel_operation(x_shm, y_shm, n, workers)
            )

        # Release exported views before closing shared memory.
        x.release()
        y.release()

        return (
            statistics.median(serial_times),
            statistics.median(parallel_times)
        )
    finally:
        x_shm.close()
        y_shm.close()
        x_shm.unlink()
        y_shm.unlink()


def main():
    workers = os.cpu_count() or 1

    print("Task 5: Parallel Overhead Crossover")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPU cores: {workers}")
    print(f"Workers: {workers}")
    print(f"Repeats per test: {REPEATS}")
    print(f"Operation: y[i] = {A} * x[i] + {B}")
    print("Timing includes process pool startup and shutdown.")
    print("Reported times are medians in milliseconds.\n")

    results = []

    print(
        f"{'N':>12} | {'Serial (ms)':>12} | "
        f"{'Parallel (ms)':>14} | {'Speedup':>9} | Result"
    )
    print("-" * 76)

    for n in SIZES:
        serial_ms, parallel_ms = benchmark_size(n, workers)
        speedup = serial_ms / parallel_ms

        if parallel_ms < serial_ms:
            status = "FASTER"
        elif parallel_ms > serial_ms:
            status = "SLOWER"
        else:
            status = "EQUAL"

        results.append({
            "n": n,
            "serial": serial_ms,
            "parallel": parallel_ms,
            "status": status
        })

        print(
            f"{n:>12,} | {serial_ms:>12.2f} | "
            f"{parallel_ms:>14.2f} | {speedup:>9.3f} | {status}"
        )

    print("\nCROSSOVER ANALYSIS")
    print("-" * 40)

    slower_sizes = [
        r["n"] for r in results if r["status"] == "SLOWER"
    ]
    faster_sizes = [
        r["n"] for r in results if r["status"] == "FASTER"
    ]

    if not faster_sizes:
        print("No crossover found: parallel was never faster")
        print("at the tested array sizes.")
    elif not slower_sizes:
        print("Parallel was faster at all tested array sizes.")
    else:
        last_slower = max(
            r["n"] for r in results
            if r["status"] == "SLOWER"
        )
        first_faster = min(
            r["n"] for r in results
            if r["status"] == "FASTER" and r["n"] > last_slower
        ) if any(
            r["status"] == "FASTER" and r["n"] > last_slower
            for r in results
        ) else None

        if first_faster is not None:
            print(f"Last slower tested size: N = {last_slower:,}")
            print(f"First faster tested size: N = {first_faster:,}")
            print(
                f"Observed crossover bracket: "
                f"{last_slower:,} < N <= {first_faster:,}"
            )
        else:
            print("Results are not a single slower-to-faster transition.")
            print(f"Slower sizes: {slower_sizes}")
            print(f"Faster sizes: {faster_sizes}")

    print("\nNote: the bracket is based on tested sizes.")
    print("It does not claim an exact crossover at every possible N.")


if __name__ == "__main__":
    main()
