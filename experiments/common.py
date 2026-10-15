"""Shared experiment plumbing: the config base class and the Modal run flow.

Each benchmark defines an `Experiment` subclass (experiments/<b>/config.py)
with `name`, `model`, `data`, `train`, `solve` and `eval` blocks, and a
Modal app (experiments/<b>/run.py) whose functions call `run_experiment`
remotely and `launch` locally.

Checkpoints live at a deterministic path derived from the training config,
so rerunning a config finds its finished checkpoint instead of retraining,
and a launcher can tell which runs exist.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Callable, ClassVar

from lattice_diffusion import config as cfglib
from lattice_diffusion.eval.config import build_spec
from lattice_diffusion.eval.records import EvalSpec
from lattice_diffusion.provenance import Provenance, header

# Training fields that do not change the trained weights: progress logging and
# in-training monitoring (which runs under a forked RNG), resume-snapshot
# cadence, and prefetch depth. The seed is kept out of the hash and put in
# the checkpoint name instead.
_UNHASHED_TRAIN = ("seed", "log_every", "monitor_every", "monitor_n_puzzles",
                   "monitor_max_rounds", "monitor_n_chains", "checkpoint_every")


@dataclasses.dataclass
class Experiment:
    BENCHMARK: ClassVar[str]
    # Keys computed from other fields in __post_init__; may not be set by hand.
    DERIVED: ClassVar[tuple[str, ...]] = ()
    # Keys that depend on where the job runs (data mounts), set by `localized`.
    ENVIRONMENT: ClassVar[tuple[str, ...]] = ()

    # Which tracked experiment launched this run and from which commit.
    # Stored with checkpoints and evals; not part of any hash.
    provenance: Provenance = dataclasses.field(default_factory=Provenance)

    def training_dict(self) -> dict:
        """Everything that determines the trained weights."""
        d = {k: cfglib.to_dict(getattr(self, k)) for k in ("model", "data", "train")}
        for key in _UNHASHED_TRAIN:
            d["train"].pop(key, None)
        for path in self.ENVIRONMENT:
            section, key = path.split(".")
            d[section].pop(key, None)
        for key in ("batch_size", "prefetch_batches"):
            d["data"].pop(key, None)    # batch_size is derived from train.batch_size
        return d

    def train_hash(self) -> str:
        return cfglib.stable_hash(self.training_dict())

    def checkpoint_path(self, root: str | Path) -> Path:
        return (Path(root) / self.BENCHMARK
                / f"{self.name}-{self.train_hash()}-seed{self.train.seed}.pt")

    def localized(self, data_dir: str):
        """Copy with data paths resolved against `data_dir`."""
        return self

    def solve_dict(self) -> dict:
        return cfglib.to_dict(self.solve)

    def eval_spec(self, checkpoint: str, **kw) -> EvalSpec:
        return build_spec(self.BENCHMARK, checkpoint, self.eval, self.solve_dict(),
                          provenance=self.provenance, **kw)

    def to_json(self) -> str:
        return json.dumps(cfglib.to_dict(self))

    @classmethod
    def from_json(cls, s: str):
        return cfglib.from_dict(cls, json.loads(s))

    def checkpoint_meta(self) -> dict:
        """Stored in every checkpoint so evals can recover the config."""
        return {"benchmark": self.BENCHMARK, "experiment": cfglib.to_dict(self),
                "train_hash": self.train_hash()}


def print_config(exp: Experiment) -> None:
    """Flattened config with values that differ from the class defaults starred."""
    defaults = cfglib.flatten(cfglib.to_dict(type(exp)()))
    values = cfglib.flatten(cfglib.to_dict(exp))
    width = max(len(k) for k in values)
    changed = 0
    for key, value in values.items():
        mark = " " if defaults.get(key) == value else "*"
        changed += mark == "*"
        tail = "" if mark == " " else f"   (default {defaults.get(key)!r})"
        print(f"  {mark} {key:<{width}} = {value!r}{tail}", flush=True)
    print(f"  {changed}/{len(values)} differ from {type(exp).__name__} defaults", flush=True)


def train_record_path(ckpt: Path) -> Path:
    """Sidecar next to a checkpoint with its training wall-clock time."""
    return ckpt.with_suffix(".train.json")


def run_experiment(exp: Experiment, train_fn: Callable, checkpoint_root: str, data_dir: str,
                   commit: Callable[[], None], evaluate: bool = True) -> dict:
    """Train unless the checkpoint exists, then spawn the shared evaluator.

    `train_fn(exp, ckpt_path, commit)` writes the checkpoint atomically;
    `commit` persists the volume (trainers that snapshot for resume call it).
    Runs inside the training container. The eval is spawned, not awaited, so
    the training GPU is released and a --detach run keeps evaluating after
    the local client disconnects. `evaluate=False` only trains (e.g. models
    that are only ever evaluated on an explicit puzzle list).
    """
    from lattice_diffusion.eval.modal_app import EVAL_ROOT, coordinate

    ckpt = exp.checkpoint_path(checkpoint_root)
    spec = exp.eval_spec(str(ckpt)) if evaluate else None
    print(header("train", exp_id=exp.provenance.exp_id, entry=exp.provenance.entry,
                 train_hash=exp.train_hash(), code=exp.provenance.code_sha, checkpoint=ckpt,
                 eval_run_dir=spec.run_dir(EVAL_ROOT) if spec else None), flush=True)
    if ckpt.exists():
        print(f"Checkpoint exists, skipping training: {ckpt}", flush=True)
    else:
        print(f"Training -> {ckpt}", flush=True)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        train_fn(exp.localized(data_dir), ckpt, commit)
        # Wall-clock of this call only: a resumed run reports its last leg.
        train_record_path(ckpt).write_text(json.dumps({
            "train_seconds": time.time() - t0, "gpu": "B200", "n_gpus": 1,
            "train_hash": exp.train_hash(), **dataclasses.asdict(exp.provenance)}, indent=2))
        commit()
    if spec is None:
        return {"checkpoint": str(ckpt), "eval_call_id": None, "eval_run_dir": None}
    call = coordinate.spawn(spec.to_json())
    run_dir = str(spec.run_dir(EVAL_ROOT))
    print(f"Spawned eval {run_dir} (call {call.object_id})", flush=True)
    return {"checkpoint": str(ckpt), "eval_call_id": call.object_id, "eval_run_dir": run_dir}


def launch(exp_cls, app, run_fn, config: str, overrides: str, wait: bool,
           evaluate: bool = True) -> None:
    """Local entrypoint body: resolve the config and run remotely.

    Records the launching commit (refusing a dirty tree) in the run's
    provenance and appends a ledger record with the app and call IDs.
    With `wait`, block until training is done and then until the eval is.
    Without it, spawn and return at once; run with `modal run --detach` so
    the job outlives the local client (this is how paper/launch.py submits
    many runs).
    """
    import modal

    from lattice_diffusion.eval.modal_app import EVAL_ROOT
    from lattice_diffusion.modal.image import CHECKPOINT_MOUNT, checkpoint_volume
    from lattice_diffusion.provenance import (
        LedgerRecord, announce, app_run_name, app_url, require_clean_sha, write_ledger,
    )

    exp = cfglib.load(exp_cls, config or None, overrides)
    exp = dataclasses.replace(exp, provenance=dataclasses.replace(
        exp.provenance, code_sha=require_clean_sha()))
    print(f"{exp_cls.BENCHMARK} config {config or '(defaults)'}  overrides: {overrides or '-'}",
          flush=True)
    print_config(exp)
    ckpt = exp.checkpoint_path(CHECKPOINT_MOUNT)
    spec = exp.eval_spec(str(ckpt)) if evaluate else None
    call = run_fn.spawn(exp.to_json(), evaluate)
    record = LedgerRecord(
        kind="train+eval" if evaluate else "train",
        exp_id=exp.provenance.exp_id, entry=exp.provenance.entry,
        # `modal run -n <name>` sets the run's description (modal.runner.run_app).
        app_name=app.description or app_run_name(exp.provenance.exp_id, exp.provenance.entry,
                                                 exp.train_hash()),
        app_id=app.app_id, app_url=app_url(app), call_id=call.object_id,
        code_sha=exp.provenance.code_sha, checkpoint=str(ckpt), train_hash=exp.train_hash(),
        eval_run_dir=str(spec.run_dir(EVAL_ROOT)) if spec else None,
        eval_hash=spec.spec_hash() if spec else None,
    )
    announce(record, write_ledger(checkpoint_volume, record))
    if not wait:
        return
    result = call.get()
    print(f"\nCheckpoint: {result['checkpoint']}\nEval: {result['eval_run_dir']}", flush=True)
    if result["eval_call_id"]:
        eval_result = modal.FunctionCall.from_id(result["eval_call_id"]).get()
        print(f"\n{eval_result['summary']}", flush=True)
