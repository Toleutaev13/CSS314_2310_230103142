
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

STUDENT_ID = 230103142
ARRAY_SIZE = 20_000_000
P = 0.95


def partial_sum(args):
    shm_name, size, start, end = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        data = np.ndarray((size,), dtype=np.float64, buffer=shm.buf)
        return float(np.sum(data[start:end]))
    finally:
        shm.close()


def amdahl_speedup(workers):
    return 1.0 / ((1.0 - P) + P / workers)


def main():
    max_cores = os.cpu_count() or 1

    worker_counts = sorted(set([
        1, 2, 4, max_cores, max_cores * 2
    ]))

    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {os.name}")
    print(f"Logical CPU cores: {max_cores}")
    print(f"Array size: {ARRAY_SIZE}")
    print("Data type: float64")
    print("Parallel method: ProcessPoolExecutor")
    print(f"Amdahl parallel fraction: {P:.0%}")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    shm = shared_memory.SharedMemory(
        create=True, size=data.nbytes
    )

    try:
        shared_data = np.ndarray(
            data.shape, dtype=np.float64, buffer=shm.buf
        )
        shared_data[:] = data
        del shared_data
        del data

        results = []

        for workers in worker_counts:
            chunk_size = (ARRAY_SIZE + workers - 1) // workers

            tasks = [
                (
                    shm.name,
                    ARRAY_SIZE,
                    start,
                    min(start + chunk_size, ARRAY_SIZE)
                )
                for start in range(0, ARRAY_SIZE, chunk_size)
            ]

            start_time = time.perf_counter()

            with ProcessPoolExecutor(
                max_workers=workers
            ) as executor:
                partial_results = list(
                    executor.map(partial_sum, tasks)
                )

            total_sum = math_fsum(partial_results)
            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            results.append({
                "workers": workers,
                "time_ms": elapsed_ms,
                "sum": total_sum
            })

        baseline = next(
            r["time_ms"] for r in results
            if r["workers"] == 1
        )

        print(
            f"{'Cores':>10} | {'Wall Time (ms)':>15} | "
            f"{'Speedup':>10} | {'Amdahl Upper Bound':>19}"
        )
        print("-" * 65)

        for result in results:
            speedup = baseline / result["time_ms"]
            theoretical = amdahl_speedup(result["workers"])

            print(
                f"{result['workers']:>10} | "
                f"{result['time_ms']:>15.2f} | "
                f"{speedup:>10.3f} | "
                f"{theoretical:>19.3f}"
            )

        print()
        print(f"Reference sum: {results[0]['sum']:.10f}")

    finally:
        shm.close()
        shm.unlink()


def math_fsum(values):
    import math
    return math.fsum(values)


if __name__ == "__main__":
    main()
