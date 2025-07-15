from contextlib import contextmanager
import time

@contextmanager
def time_block(label):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[TIMER] {label}: {(end - start)*1000:.2f} ms")
