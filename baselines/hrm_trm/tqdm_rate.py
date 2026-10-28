"""Steady-state seconds per step from a tqdm progress log."""

import re

_RATE = re.compile(r"(\d+)/(\d+) \[[^\]]*?([\d.]+)(s/it|it/s)\]")


def seconds_per_step(log: str, desc: str = "") -> tuple[int, int, float]:
    """The last (step, total steps, s/step) reported by tqdm in `log`.

    With `desc`, only progress lines containing it count (e.g. the training bar,
    not an evaluation bar printed after it).
    """
    if desc:
        log = "\n".join(line for line in log.replace("\r", "\n").splitlines() if desc in line)
    matches = _RATE.findall(log)
    if not matches:
        raise ValueError("no tqdm rate in the log")
    step, total, rate, unit = matches[-1]
    return int(step), int(total), float(rate) if unit == "s/it" else 1 / float(rate)
