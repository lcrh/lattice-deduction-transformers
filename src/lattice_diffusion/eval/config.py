"""The `eval` block of an experiment config and its translation to an EvalSpec."""

from __future__ import annotations

from dataclasses import dataclass

from lattice_diffusion.eval.records import EvalSpec
from lattice_diffusion.eval.stop import PRESETS


@dataclass
class EvalConfig:
    # Named list from lattice_diffusion.eval.indices, or "range" for the
    # first `n` items of `split` in dataset order.
    index_list: str = "sudoku_test_perm"
    split: str | None = None
    n: int | None = None
    stop: str = "sequential"            # stop-rule preset (lattice_diffusion.eval.stop)
    dropout_p: float = 0.05             # eval-time dropout, used as search noise
    eval_seed: int = 0
    block_size: int = 512
    compile: bool = False
    tag: str = ""
    # Cost / wall-clock only; not part of the eval's identity.
    workers: int = 32
    max_in_flight: int | None = None    # None = workers
    keep_warm_s: int = 300

    def __post_init__(self):
        if self.stop not in PRESETS:
            raise ValueError(f"eval.stop must be one of {sorted(PRESETS)}, got {self.stop!r}")
        if self.index_list == "range" and (self.split is None or self.n is None):
            raise ValueError("eval.index_list=range needs eval.split and eval.n")


def build_spec(task: str, checkpoint: str, cfg: EvalConfig, solve: dict,
               explicit_indices: list[int] | None = None,
               explicit_split: str | None = None) -> EvalSpec:
    """EvalSpec for `checkpoint`. `explicit_indices` (with `explicit_split`)
    replaces the configured index list, e.g. for a portfolio failure set."""
    index_list = cfg.index_list
    if explicit_indices is None and index_list == "range":
        explicit_indices, explicit_split = list(range(cfg.n)), cfg.split
    if explicit_indices is not None:
        tag = cfg.tag or (f"range-{explicit_split}-{len(explicit_indices)}"
                          if index_list == "range" else "explicit")
        index_list = "explicit"
    else:
        tag = cfg.tag or index_list
    return EvalSpec(
        task=task,
        checkpoint=checkpoint,
        index_list=index_list,
        solve={**solve, "dropout_p": cfg.dropout_p},
        stop=PRESETS[cfg.stop].to_dict(),
        block_size=cfg.block_size,
        eval_seed=cfg.eval_seed,
        compile=cfg.compile,
        tag=tag,
        explicit_indices=explicit_indices,
        explicit_split=explicit_split,
        n_workers=cfg.workers,
        max_in_flight=cfg.max_in_flight,
        keep_warm_s=cfg.keep_warm_s,
    )
