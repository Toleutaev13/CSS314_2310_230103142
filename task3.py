
import os
import time
import platform
import threading

ITERATIONS = 50_000_000


def get_physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def unpadded_worker(shared_counts, index):
    for _ in range(ITERATIONS):
        shared_counts[index] += 1


def local_worker(shared_counts, index):
    local_count = 0

    for _ in range(ITERATIONS):
        local_count += 1

    shared_counts[index] = local_count


def run_test(worker_function, number_of_threads):
    shared_counts = [0] * number_of_threads

    threads = [
        threading.Thread(
            target=worker_function,
            args=(shared_counts, i)
        )
        for i in range(number_of_threads)
    ]

    start = time.perf_counter()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    elapsed_ms = (time.perf_counter() - start) * 1000

    expected = ITERATIONS * number_of_threads
    actual = sum(shared_counts)

    if actual != expected:
        raise RuntimeError(
            f"Incorrect result: expected {expected}, got {actual}"
        )

    return elapsed_ms


def main():
    workers = get_physical_cores()

    print("Task 3: Cache Line Contention")
    print(f"Student ID: 230103142")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Physical cores / worker threads: {workers}")
    print(f"Iterations per thread: {ITERATIONS:,}")
    print()

    print("Running unpadded shared counters...")
    unpadded_time = run_test(unpadded_worker, workers)

    print("Running local counters...")
    local_time = run_test(local_worker, workers)

    ratio = (
        unpadded_time / local_time
        if local_time > 0 else float("inf")
    )

    print()
    print(f"{'Method':<25} {'Wall Time (ms)':>16}")
    print("-" * 43)
    print(f"{'Unpadded Shared':<25} {unpadded_time:>16.2f}")
    print(f"{'Local Variables':<25} {local_time:>16.2f}")
    print(f"{'Execution Ratio':<25} {ratio:>15.2f}x")

    print()
    print(f"Total increments per run: {ITERATIONS * workers:,}")
    print("Validation: PASSED")


if __name__ == "__main__":
    main()
