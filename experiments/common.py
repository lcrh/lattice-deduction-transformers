"""Experiment config base class and the train-then-evaluate Modal flow.

Each benchmark defines an `Experiment` subclass (experiments/<b>/config.py)
and a Modal app (experiments/<b>/run.py) that calls `run_experiment`
remotely and `launch` locally. Checkpoint paths derive from the training
config, so a finished config is found instead of retrained.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypedDict

from ldt import config as cfglib
from ldt.eval.config import EvalConfig, build_spec
from ldt.eval.records import EvalSpec
from ldt.provenance import LedgerRecord, Provenance, header
from ldt.training.utils.checkpoint import load_checkpoint

if TYPE_CHECKING:
    import modal

    from ldt.models.looped_transformer import LoopedTransformerConfig
    from experiments.sudoku.solve import SolveConfig
    from experiments.sudoku.train import TrainConfig

GPU = "B200"

# Training fields that do not change the trained weights (logging, monitoring
# under a forked RNG, snapshots). The seed goes in the checkpoint name.
_UNHASHED_TRAIN = ("seed", "log_every", "monitor_every", "monitor_n_puzzles",
                   "monitor_max_rounds", "monitor_n_chains", "checkpoint_every",
                   "keep_step_snapshots")
# batch_size follows train.batch_size; prefetch depth does not change the data.
_UNHASHED_DATA = ("batch_size", "prefetch_batches")


def with_step(block, **kw):
    """Copy of a train / solve block with its StepConfig fields replaced."""
    return dataclasses.replace(block, step=dataclasses.replace(block.step, **kw))


def check_options(model: LoopedTransformerConfig, train: TrainConfig, solve: SolveConfig,
                  carry_supported: bool = True) -> None:
    """Cross-block constraints between model, training and solver options."""
    if model.cls_token != (train.conflict_loss_weight > 0):
        raise ValueError(
            f"model.cls_token={model.cls_token} but train.conflict_loss_weight="
            f"{train.conflict_loss_weight}; the conflict head is trained iff it exists")
    if train.supervise not in ("all", "final"):
        raise ValueError(f"train.supervise must be all or final, got {train.supervise!r}")
    if model.carry != "off":
        if not carry_supported:
            raise ValueError("latent carry is only implemented for the Sudoku trainer")
        # The latent lives in a fixed frame and belongs to one trajectory.
        for where, step in (("train.step", train.step), ("solve.step", solve.step)):
            if step.augment or step.max_passes != 1:
                raise ValueError(f"model.carry={model.carry!r} needs {where}.augment=false "
                                 f"and {where}.deduce_passes=1")
        if solve.backtrack != "root":
            raise ValueError(f"model.carry={model.carry!r} needs solve.backtrack=root")


class RunResult(TypedDict):
    checkpoint: str
    eval_call_id: str | None
    eval_run_dir: str | None


@dataclasses.dataclass
class Experiment:
    """One run: model, data, training, solver and eval settings plus provenance."""

    BENCHMARK: ClassVar[str]
    # Keys computed from other fields in __post_init__; may not be set by hand.
    DERIVED: ClassVar[tuple[str, ...]] = ()
    # Keys that depend on where the job runs (data mounts), set by `localized`.
    ENVIRONMENT: ClassVar[tuple[str, ...]] = ()

    # Which tracked experiment launched this run and from which commit.
    # Stored with checkpoints and evals; not part of any hash.
    provenance: Provenance = dataclasses.field(default_factory=Provenance)

    if TYPE_CHECKING:
        model: LoopedTransformerConfig
        data: Any
        train: TrainConfig
        solve: SolveConfig
        eval: EvalConfig

    def training_dict(self) -> dict:
        """Everything that determines the trained weights."""
        d = {k: cfglib.to_dict(getattr(self, k)) for k in ("model", "data", "train")}
        for key in _UNHASHED_TRAIN:
            d["train"].pop(key, None)
        for path in self.ENVIRONMENT:
            section, key = path.split(".")
            d[section].pop(key, None)
        for key in _UNHASHED_DATA:
            d["data"].pop(key, None)
        return d

    def train_hash(self) -> str:
        return cfglib.stable_hash(self.training_dict())

    def checkpoint_path(self, root: str | Path) -> Path:
        """Depends only on the training settings, so configs that train alike share it."""
        return Path(root) / self.BENCHMARK / f"{self.train_hash()}-seed{self.train.seed}.pt"

    def localized(self, data_dir: str) -> Experiment:
        """Copy with every ENVIRONMENT key pointed at `data_dir`."""
        exp = self
        for path in self.ENVIRONMENT:
            section, key = path.split(".")
            exp = dataclasses.replace(
                exp, **{section: dataclasses.replace(getattr(exp, section), **{key: data_dir})})
        return exp

    def solve_dict(self) -> dict:
        return cfglib.to_dict(self.solve)

    def eval_spec(self, checkpoint: str, **kw) -> EvalSpec:
        return build_spec(self.BENCHMARK, checkpoint, self.eval, self.solve_dict(),
                          provenance=self.provenance, **kw)

    def to_json(self) -> str:
        return json.dumps(cfglib.to_dict(self))

    @classmethod
    def from_json(cls, s: str) -> Experiment:
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
                   commit: Callable[[], None], evaluate: bool = True) -> RunResult:
    """Train unless the checkpoint exists, then spawn the shared evaluator.

    Runs in the training container. The eval is spawned, not awaited, so the GPU is released and a --detach run
    keeps evaluating after the client disconnects.

    Args:
        train_fn: `train_fn(exp, ckpt_path, commit)` writes the checkpoint atomically.
        commit: Persists the volume (trainers that snapshot for resume call it).
        evaluate: False only trains.
    """
    from ldt.eval.modal_app import EVAL_ROOT, coordinate

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
        # train_seconds covers this call; the checkpoint's train_wallclock sums every
        # leg of a resumed run.
        wallclock = load_checkpoint(str(ckpt)).get("train_wallclock", {})
        train_record_path(ckpt).write_text(json.dumps({
            "train_seconds": time.time() - t0, **wallclock, "gpu": GPU, "n_gpus": 1,
            "train_hash": exp.train_hash(), **dataclasses.asdict(exp.provenance)}, indent=2))
        commit()
    if spec is None:
        return RunResult(checkpoint=str(ckpt), eval_call_id=None, eval_run_dir=None)
    call = coordinate.spawn(spec.to_json())
    run_dir = str(spec.run_dir(EVAL_ROOT))
    print(f"Spawned eval {run_dir} (call {call.object_id})", flush=True)
    return RunResult(checkpoint=str(ckpt), eval_call_id=call.object_id, eval_run_dir=run_dir)


def with_code_sha(exp: Experiment) -> Experiment:
    """`exp` with the launching commit recorded (refuses a dirty tree)."""
    from ldt.provenance import require_clean_sha
    return dataclasses.replace(exp, provenance=dataclasses.replace(
        exp.provenance, code_sha=require_clean_sha()))


def record_launch(app: modal.App, call: modal.FunctionCall, kind: str, exp: Experiment,
                  checkpoint: str, spec: EvalSpec | None) -> None:
    """Write and print the ledger record of a launch."""
    from ldt.eval.modal_app import EVAL_ROOT
    from ldt.modal.image import checkpoint_volume
    from ldt.provenance import announce, app_run_name, app_url, write_ledger

    prov = exp.provenance
    record = LedgerRecord(
        kind=kind, exp_id=prov.exp_id, entry=prov.entry,
        # `modal run -n <name>` sets the run's description.
        app_name=app.description or app_run_name(prov.exp_id, prov.entry, exp.train_hash(),
                                                 spec.spec_hash() if spec else None),
        app_id=app.app_id, app_url=app_url(app), call_id=call.object_id,
        code_sha=prov.code_sha, checkpoint=checkpoint, train_hash=exp.train_hash(),
        eval_run_dir=str(spec.run_dir(EVAL_ROOT)) if spec else None,
        eval_hash=spec.spec_hash() if spec else None,
    )
    announce(record, write_ledger(checkpoint_volume, record))


def launch(exp_cls: type[Experiment], app: modal.App, run_fn: modal.Function, config: str,
           overrides: str, wait: bool, evaluate: bool = True) -> None:
    """Local entrypoint body: resolve the config, run remotely, record the launch.

    Without `wait`, spawn and return at once (use `modal run --detach` so the job
    outlives the client); with it, block until training and then the eval finish.
    """
    import modal

    from ldt.modal.image import CHECKPOINT_MOUNT

    exp = with_code_sha(cfglib.load(exp_cls, config or None, overrides))
    print(f"{exp_cls.BENCHMARK} config {config or '(defaults)'}  overrides: {overrides or '-'}",
          flush=True)
    print_config(exp)
    ckpt = exp.checkpoint_path(CHECKPOINT_MOUNT)
    spec = exp.eval_spec(str(ckpt)) if evaluate else None
    call = run_fn.spawn(exp.to_json(), evaluate)
    record_launch(app, call, "train+eval" if evaluate else "train", exp, str(ckpt), spec)
    if not wait:
        return
    result = call.get()
    print(f"\nCheckpoint: {result['checkpoint']}\nEval: {result['eval_run_dir']}", flush=True)
    if result["eval_call_id"]:
        eval_result = modal.FunctionCall.from_id(result["eval_call_id"]).get()
        print(f"\n{eval_result['summary']}", flush=True)
