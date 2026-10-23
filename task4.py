
import time
import threading
import queue
import platform

STUDENT_ID = 230103142
TOTAL_ITEMS = 1000
WORKERS = 4
CHUNK_SIZE = 1


def process_item(i):
    time.sleep(0.0001 * (i % 20))


def run_static():
    completion_times = [0.0] * WORKERS
    block_size = TOTAL_ITEMS // WORKERS

    def worker(worker_id, start, end, start_time):
        for i in range(start, end):
            process_item(i)

        completion_times[worker_id] = time.perf_counter() - start_time

    start_time = time.perf_counter()
    threads = []

    for worker_id in range(WORKERS):
        start = worker_id * block_size
        end = start + block_size

        thread = threading.Thread(
            target=worker,
            args=(worker_id, start, end, start_time)
        )
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    total_time = time.perf_counter() - start_time
    return total_time, completion_times


def run_dynamic():
    tasks = queue.Queue()
    completion_times = [0.0] * WORKERS

    for start in range(0, TOTAL_ITEMS, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, TOTAL_ITEMS)
        tasks.put((start, end))

    def worker(worker_id, start_time):
        while True:
            try:
                start, end = tasks.get_nowait()
            except queue.Empty:
                break

            try:
                for i in range(start, end):
                    process_item(i)
            finally:
                tasks.task_done()

        completion_times[worker_id] = time.perf_counter() - start_time

    start_time = time.perf_counter()
    threads = [
        threading.Thread(
            target=worker,
            args=(worker_id, start_time)
        )
        for worker_id in range(WORKERS)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    total_time = time.perf_counter() - start_time
    return total_time, completion_times


def print_results(strategy, total_time, completion_times):
    earliest = min(completion_times)
    latest = max(completion_times)
    delta = latest - earliest

    print(f"\n{strategy}")
    print("-" * 48)
    print(f"Total execution time: {total_time * 1000:.2f} ms")

    print("Thread completion times:")
    for i, completion in enumerate(completion_times):
        print(f"  Thread {i}: {completion * 1000:.2f} ms")

    print(f"Minimum completion time: {earliest * 1000:.2f} ms")
    print(f"Maximum completion time: {latest * 1000:.2f} ms")
    print(f"First-to-last completion delta: {delta * 1000:.2f} ms")


def main():
    print("Task 4: Work Scheduling Under Skew")
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"Total items: {TOTAL_ITEMS}")
    print(f"Parallel workers: {WORKERS}")
    print("Simulated cost: sleep(0.0001 * (i % 20))")
    print(f"Dynamic chunk size: {CHUNK_SIZE}")

    static_time, static_completion = run_static()
    dynamic_time, dynamic_completion = run_dynamic()

    print_results(
        "Strategy A: Static Chunk Distribution",
        static_time,
        static_completion
    )

    print_results(
        "Strategy B: Dynamic Work Queue",
        dynamic_time,
        dynamic_completion
    )

    print("\nComparison")
    print("-" * 48)
    print(f"Static total time:  {static_time * 1000:.2f} ms")
    print(f"Dynamic total time: {dynamic_time * 1000:.2f} ms")

    if dynamic_time < static_time:
        print("Result: Dynamic scheduling was faster.")
    elif static_time < dynamic_time:
        print("Result: Static scheduling was faster.")
    else:
        print("Result: Both strategies had equal execution times.")


if __name__ == "__main__":
    main()
