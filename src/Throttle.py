import time
import threading
from threading import Lock
from urllib.parse import urlparse


class Throttle:
    """Thread-safe rate limiter using sleep-outside-lock slot reservation.

    Threads acquire the lock only to compute and reserve their time slot,
    then sleep outside the lock so other threads can immediately reserve
    subsequent slots.

    Supports per-domain throttling: pass a URL to wait() and each domain
    gets its own independent rate-limit slot.
    """

    def __init__(self, min_interval: float):
        self._lock = Lock()
        self._last_time = 0.0          # global fallback
        self._domain_last_times: dict[str, float] = {}
        self._min_interval = min_interval

    def wait(self, url: str = None):
        # Each thread acquires the lock, computes its wait time and reserves a
        # slot, then releases the lock before sleeping. This lets all threads
        # schedule their slots immediately instead of blocking behind each
        # other's sleeps.
        domain = None
        if url:
            domain = urlparse(url).netloc.lower()

        with self._lock:
            now = time.time()
            if domain:
                last = self._domain_last_times.get(domain, 0.0)
            else:
                last = self._last_time

            wait_needed = self._min_interval - (now - last)
            if wait_needed > 0:
                # Reserve the next slot by advancing last_time by the interval,
                # rather than setting it to now. This way if 5 threads pile up
                # at t=10, they get slots at t=10, t=13, t=16, t=19, t=22.
                new_last = last + self._min_interval
            else:
                new_last = now
                wait_needed = 0

            if domain:
                self._domain_last_times[domain] = new_last
            else:
                self._last_time = new_last

        # Sleep outside the lock so other threads can immediately acquire it
        # and reserve their own slots.
        if wait_needed > 0:
            label = domain or "global"
            thread_name = threading.current_thread().name
            print(f"Throttle: {label} — waiting {wait_needed:.1f}s (thread: {thread_name})")
            time.sleep(wait_needed)
