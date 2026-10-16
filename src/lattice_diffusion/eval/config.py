"""The `eval` block of an experiment config and its translation to an EvalSpec."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from lattice_diffusion.eval.indices import filter_tag
from lattice_diffusion.eval.records import EvalSpec
from lattice_diffusion.eval.stop import PRESETS
from lattice_diffusion.provenance import Provenance


@dataclass
class EvalConfig:
    """How a trained checkpoint is evaluated."""

    # Named list (lattice_diffusion.eval.indices), or "range" for the first `n` items of `split`.
    index_list: str = "sudoku_test_perm"
    split: str | None = None
    n: int | None = None
    index_filter: dict | None = None    # see EvalSpec.index_filter
    stop: str = "sequential"            # stop-rule preset (lattice_diffusion.eval.stop)
    dropout_p: float = 0.05             # eval-time dropout, used as search noise
    eval_seed: int = 0
    block_size: int = 512
    compile: bool = True                # torch.compile; GPU results then match only statistically
    tag: str = ""
    # Cost and wall-clock only; not part of the eval's identity.
    workers: int = 1
    max_in_flight: int | None = None    # None = workers
    keep_warm_s: int = 300

    def __post_init__(self):
        if self.stop not in PRESETS:
            raise ValueError(f"eval.stop must be one of {sorted(PRESETS)}, got {self.stop!r}")
        if self.index_list == "range" and (self.split is None or self.n is None):
            raise ValueError("eval.index_list=range needs eval.split and eval.n")


def build_spec(task: str, checkpoint: str, cfg: EvalConfig, solve: dict,
               explicit_indices: list[int] | None = None,
               explicit_split: str | None = None,
               provenance: Provenance | None = None) -> EvalSpec:
    """EvalSpec for `checkpoint` under eval config `cfg`.

    Args:
        explicit_indices: Replace the configured list (e.g. a portfolio failure set);
            needs explicit_split.
        provenance: Recorded with the spec, not hashed.
    """
    if explicit_indices is None and cfg.index_list == "range":
        explicit_indices, explicit_split = list(range(cfg.n)), cfg.split
    if explicit_indices is not None:
        default_tag = (f"range-{explicit_split}-{len(explicit_indices)}"
                       if cfg.index_list == "range" else "explicit")
    elif cfg.index_filter is not None:
        default_tag = f"{cfg.index_list}[{filter_tag(cfg.index_filter)}]"
    else:
        default_tag = cfg.index_list
    return EvalSpec(
        task=task,
        checkpoint=checkpoint,
        index_list="explicit" if explicit_indices is not None else cfg.index_list,
        solve={**solve, "dropout_p": cfg.dropout_p},
        stop=PRESETS[cfg.stop].to_dict(),
        block_size=cfg.block_size,
        eval_seed=cfg.eval_seed,
        compile=cfg.compile,
        tag=cfg.tag or default_tag,
        explicit_indices=explicit_indices,
        explicit_split=explicit_split,
        index_filter=cfg.index_filter,
        n_workers=cfg.workers,
        max_in_flight=cfg.max_in_flight,
        keep_warm_s=cfg.keep_warm_s,
        **asdict(provenance or Provenance()),
    )
