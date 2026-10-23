
import threading
import time

# Task 6: Synchronization Cost & Data Loss

THREADS = 4
INCREMENTS_PER_THREAD = 2_000_000
EXPECTED = THREADS * INCREMENTS_PER_THREAD

# Version 1: without synchronization
counter = [0]


def unsafe_increment():
    for i in range(INCREMENTS_PER_THREAD):
        current_value = counter[0]

        # Encourage thread switching to expose the race condition.
        if i % 1000 == 0:
            time.sleep(0)

        counter[0] = current_value + 1


start = time.perf_counter()

threads = [
    threading.Thread(target=unsafe_increment)
    for _ in range(THREADS)
]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

unsafe_time = time.perf_counter() - start
unsafe_result = counter[0]

lost_updates = EXPECTED - unsafe_result
lost_percentage = (lost_updates / EXPECTED) * 100

print("=" * 55)
print("TASK 6: SYNCHRONIZATION COST & DATA LOSS")
print("=" * 55)
print(f"Threads: {THREADS}")
print(f"Increments per thread: {INCREMENTS_PER_THREAD:,}")
print(f"Expected final value: {EXPECTED:,}")
print("\n--- WITHOUT LOCK ---")
print(f"Actual final value: {unsafe_result:,}")
print(f"Lost updates: {lost_updates:,}")
print(f"Updates lost: {lost_percentage:.2f}%")
print(f"Execution time: {unsafe_time:.4f} seconds")

# Version 2: correct synchronization using Lock
counter = [0]
lock = threading.Lock()


def safe_increment():
    for _ in range(INCREMENTS_PER_THREAD):
        with lock:
            counter[0] += 1


start = time.perf_counter()

threads = [
    threading.Thread(target=safe_increment)
    for _ in range(THREADS)
]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

safe_time = time.perf_counter() - start
safe_result = counter[0]

time_penalty = safe_time - unsafe_time
penalty_percentage = (time_penalty / unsafe_time) * 100

print("\n--- WITH LOCK ---")
print(f"Actual final value: {safe_result:,}")
print(f"Execution time: {safe_time:.4f} seconds")
print(f"Correct result: {safe_result == EXPECTED}")

print("\n--- SYNCHRONIZATION COST ---")
print(f"Time difference: {time_penalty:.4f} seconds")
print(f"Relative time penalty: {penalty_percentage:.2f}%")
print("=" * 55)
