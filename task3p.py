import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import numpy as np
STUDENT_ID = 230103142
N = 2_000_000
THREADS = 4
REPEATS = 3

rng = np.random.default_rng(STUDENT_ID)
input_array = rng.integers(1, 11, size=N, dtype=np.int64)


def serial_scan(data):
    return np.cumsum(data, dtype=np.int64)


def calculate_local_scan(chunk):
    local_output = np.cumsum(chunk, dtype=np.int64)
    chunk_sum = int(local_output[-1])
    return local_output, chunk_sum


def add_offset(args):
    local_output, offset = args
    if offset != 0:
        local_output += offset
    return local_output


def parallel_scan(data):
    chunks = [
        data[start:min(start + (N + THREADS - 1) // THREADS, N)]
        for start in range(0, N, (N + THREADS - 1) // THREADS)
    ]

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        local_results = list(executor.map(calculate_local_scan, chunks))

        local_outputs = [result[0] for result in local_results]
        chunk_sums = [result[1] for result in local_results]
        offsets = [0] * len(chunk_sums)
        running_total = 0

        for i, chunk_sum in enumerate(chunk_sums):
            offsets[i] = running_total
            running_total += chunk_sum

        adjusted = list(
            executor.map(add_offset, zip(local_outputs, offsets))
        )

    return np.concatenate(adjusted)


def measure(function):
    times = []
    output = None

    for _ in range(REPEATS):
        start = time.perf_counter()
        output = function()
        times.append((time.perf_counter() - start) * 1000)

    return output, statistics.median(times)


def main():
    print("=" * 72)
    print("TASK 3: PREFIX SCAN - STEP VS. WORK EFFICIENCY")
    print("=" * 72)
    print(f"Student ID: {STUDENT_ID}")
    print(f"Array length: {N:,}")
    print("Input values: integers from 1 to 10")
    print(f"Parallel workers: {THREADS}")
    print(f"Repeats per method: {REPEATS}")

    serial_output, serial_ms = measure(lambda: serial_scan(input_array))
    parallel_output, parallel_ms = measure(lambda: parallel_scan(input_array))

    speedup = serial_ms / parallel_ms if parallel_ms > 0 else float("inf")

    print("\n--- RESULTS ---")
    print(f"Serial Time (ms):   {serial_ms:.2f}")
    print(f"Parallel Time (ms): {parallel_ms:.2f}")
    print(f"Speedup:            {speedup:.3f}x")

    try:
        assert parallel_output[-1] == serial_output[-1]

        assert np.array_equal(parallel_output, serial_output)

        print("Verification Status: PASS")
        print(f"Serial final value:   {serial_output[-1]:,}")
        print(f"Parallel final value: {parallel_output[-1]:,}")

    except AssertionError:
        print("Verification Status: FAIL")
        print(f"Serial final value:   {serial_output[-1]:,}")
        print(f"Parallel final value: {parallel_output[-1]:,}")
        sys.exit(1)

    print("\nNote: Parallel timing includes thread-pool setup and result assembly.")
    print("For this workload, overhead can make the parallel version slower.")
    print("=" * 72)


if __name__ == "__main__":
    main()