
import os
import platform
import random
import time
import math
import statistics
import struct

STUDENT_ID = 230103142
ARRAY_SIZE = 20_000_000


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")
    print(f"CPU / Platform: {platform.processor() or platform.machine()}")
    print(f"Logical CPU cores: {os.cpu_count()}")

    rng = random.Random(STUDENT_ID)

    start_time = time.perf_counter()

    data = [rng.random() for _ in range(ARRAY_SIZE)]

    total_sum = math.fsum(data)
    mean = total_sum / ARRAY_SIZE

    squared_deviations = (
        (x - mean) ** 2 for x in data
    )
    variance = math.fsum(squared_deviations) / ARRAY_SIZE
    standard_deviation = math.sqrt(variance)

    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    print(f"Array size: {ARRAY_SIZE}")
    print("Data type: float64")
    print("Execution mode: Serial (single thread)")
    print(f"Mathematical sum: {total_sum:.10f}")
    print(f"Standard deviation: {standard_deviation:.10f}")
    print(f"Wall-clock time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()

# there is output
# Student ID: 230103142
# Operating System: Windows-10-10.0.19045-SP0
# CPU / Platform: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel
# Logical CPU cores: 8
# Array size: 20000000
# Data type: float64
# Execution mode: Serial (single thread)
# Mathematical sum: 9997708.8646199349
# Standard deviation: 0.2887170064
# Wall-clock time: 9269.01 ms