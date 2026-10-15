"""CPU stand-ins for an eval task and for out-of-order executors."""

import random
import threading
import time

import numpy as np
import torch

from lattice_diffusion.eval import indices as index_lists
from lattice_diffusion.eval.records import EvalSpec
from lattice_diffusion.eval.runner import block_args, solve_block
from lattice_diffusion.eval.stop import SequentialStop

_TORCH_LOCK = threading.Lock()


class FakeTask:
    """Outcome of each puzzle drawn from the (block-seeded) torch RNG, like
    the real solver's sampling: ~3% wrong/timeout."""

    name = "fake"

    def __init__(self, data_dir: str = "") -> None:
        pass

    def split_size(self, split: str) -> int:
        return 5000

    def solve_block(self, model, split, indices, solve):
        r = torch.rand(len(indices)).tolist()
        rows = []
        for idx, u in zip(indices.tolist(), r):
            status = "correct" if u < solve["p_correct"] else ("wrong" if u < 0.975 else "timeout")
            rows.append({"idx": idx, "status": status, "rounds": 1, "calls": int(u * 100),
                         "forwards": 64, "n_resets": 0, "solution": "", "verified": False})
        return rows


def make_spec(stop=None, **kw) -> EvalSpec:
    lst = index_list()
    return EvalSpec(
        task="fake", checkpoint="/ckpt/fake.pt", index_list="explicit",
        solve={"p_correct": 0.97},
        stop=(stop or SequentialStop(max_failures=30, n_min=200)).to_dict(),
        block_size=64, explicit_indices=lst.indices.tolist(), explicit_split="test", **kw,
    )


def index_list():
    return index_lists.explicit("fake", "test", np.random.default_rng(0).permutation(5000))


def block_solver(spec: EvalSpec, jitter_s: float = 0.0):
    """block id -> rows, serialized on the global torch RNG like one process
    per worker would be, with optional random delay to shuffle completion."""
    task, lst = FakeTask(), index_list()

    def solve(block: int):
        if jitter_s:
            time.sleep(random.random() * jitter_s)
        with _TORCH_LOCK:
            return solve_block(task, None, block_args(spec, lst, block))

    return solve


class ReorderExecutor:
    """Completes submitted blocks one at a time in a random (seeded) order;
    `slow` blocks only complete when nothing else is pending."""

    def __init__(self, solve, seed: int = 0, slow: frozenset = frozenset()):
        self.solve = solve
        self.rng = random.Random(seed)
        self.slow = slow
        self.pending: list[int] = []
        self.submitted: list[int] = []

    def submit(self, block: int) -> None:
        self.pending.append(block)
        self.submitted.append(block)

    def wait_any(self):
        fast = [b for b in self.pending if b not in self.slow]
        block = self.rng.choice(fast or self.pending)
        self.pending.remove(block)
        return [(block, self.solve(block))]
