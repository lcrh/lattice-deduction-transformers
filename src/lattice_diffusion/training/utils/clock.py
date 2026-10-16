"""Training wall-clock split into the compile step, monitoring, and the rest."""

from __future__ import annotations

import time
from contextlib import contextmanager


class TrainClock:
    """Call `tick()` at the top of every training step and wrap monitoring in `monitoring()`.

    The first step includes torch.compile, so it is reported separately;
    `post_compile_secs` is the loop time after it, minus monitoring.
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._first_step_end: float | None = None
        self._ticks = 0
        self._monitor = 0.0

    def tick(self) -> None:
        now = time.perf_counter()
        if self._ticks == 0:
            self._start = now
        elif self._ticks == 1:
            self._first_step_end = now
        self._ticks += 1

    @contextmanager
    def monitoring(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._monitor += time.perf_counter() - t0

    def finish(self) -> dict:
        end = time.perf_counter()
        start = self._start if self._start is not None else end
        first_end = self._first_step_end if self._first_step_end is not None else end
        return {
            "total_secs": end - start,
            "step1_compile_secs": first_end - start,
            "monitor_secs": self._monitor,
            "post_compile_secs": end - first_end - self._monitor,
        }
