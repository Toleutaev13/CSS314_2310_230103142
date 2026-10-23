import multiprocessing as mp
from multiprocessing import shared_memory
import time
import statistics
import numpy as np
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 3


def ipc_worker(conn):
    """Receive arrays by pickle or read them through shared memory."""
    while True:
        message = conn.recv()

        if message is None:
            break

        mode = message[0]

        if mode == "pickle":
            array = message[1]
            checksum = int(np.sum(array, dtype=np.int64))

        elif mode == "shared":
            _, shm_name, shape, dtype_str = message
            shm = shared_memory.SharedMemory(name=shm_name)
            try:
                array = np.ndarray(
                    shape,
                    dtype=np.dtype(dtype_str),
                    buffer=shm.buf
                )
                checksum = int(np.sum(array, dtype=np.int64))
            finally:
                shm.close()

        else:
            raise ValueError(f"Unknown mode: {mode}")

        conn.send(checksum)

    conn.close()


def measure_pickle(conn, array):
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        conn.send(("pickle", array))
        checksum = conn.recv()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

        expected = int(np.sum(array, dtype=np.int64))
        if checksum != expected:
            raise RuntimeError("Pickle IPC checksum mismatch")

    return statistics.median(times)


def measure_shared_memory(conn, array):
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)

    try:
        shared_array = np.ndarray(
            array.shape,
            dtype=array.dtype,
            buffer=shm.buf
        )
        shared_array[:] = array

        times = []

        for _ in range(REPEATS):
            start = time.perf_counter()

            conn.send((
                "shared",
                shm.name,
                array.shape,
                array.dtype.str
            ))
            checksum = conn.recv()

            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

            expected = int(np.sum(array, dtype=np.int64))
            if checksum != expected:
                raise RuntimeError("Shared memory checksum mismatch")

        return statistics.median(times)

    finally:
        shm.close()
        shm.unlink()


def main():
    print("=" * 76)
    print("TASK 4: IPC SERIALIZATION VS. SHARED MEMORY")
    print("=" * 76)
    print(f"Repeats per test: {REPEATS}")
    print("Timing includes request, IPC transfer/access, and checksum response.")
    print("Shared-memory allocation and initial data copy are excluded from timing.")

    parent_conn, child_conn = mp.Pipe()
    process = mp.Process(target=ipc_worker, args=(child_conn,))
    process.start()
    child_conn.close()

    try:
        parent_conn.send(("pickle", np.array([1, 2, 3], dtype=np.int64)))
        parent_conn.recv()

        print(
            f"\n{'N':>12} | {'Path A Pickle (ms)':>20} | "
            f"{'Path B Shared Memory (ms)':>26} | {'A/B':>8}"
        )
        print("-" * 76)

        for n in SIZES:
            array = np.arange(n, dtype=np.int64)

            pickle_ms = measure_pickle(parent_conn, array)
            shared_ms = measure_shared_memory(parent_conn, array)
            ratio = pickle_ms / shared_ms if shared_ms > 0 else float("inf")
            print(
                f"{n:>12,} | {pickle_ms:>20.2f} | "
                f"{shared_ms:>26.2f} | {ratio:>7.2f}x"
            )
            del array
        print("\nInterpretation:")
        print("Path A serializes and transfers the full array between processes.")
        print("Path B shares the data buffer and sends only metadata over the pipe.")
        print("A/B above 1.0x means shared memory had lower measured overhead.")
        print("Note: Shared-memory setup and initial copy are not included in timing.")
    finally:
        parent_conn.send(None)
        process.join()
        parent_conn.close()
    print("=" * 76)
if __name__ == "__main__":
    mp.freeze_support()
    main()