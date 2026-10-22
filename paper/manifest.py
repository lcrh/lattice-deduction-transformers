"""Paper manifests: which runs back each table and figure.

A manifest (paper/manifests/<name>.yaml) lists entries; each entry expands
to one run per seed (and per eval seed). A run is fully resolved locally:
its experiment config, checkpoint path, training hash, eval spec and the
summary file the evaluator will write. That is how the launcher finds what
already exists on the volume without asking the jobs.

Manifest keys (all optional except `benchmark` and `entries`):

    exp_id      tracker experiment ID, e.g. "A1" (entries may override)
    benchmark   sudoku | snowflake | maze | baselines
    config      base config file for every entry
    eval        eval preset (paper/manifests/_eval.yaml) for every entry
    seeds       training seeds, default [0]
    set         overrides applied to every entry ({dotted.key: value})
    table       hints for the table/figure scripts (TableHints)
    entries     list of entries

Entry keys:

    name        unique within the manifest
    exp_id, config, eval, seeds, set    as above, per entry
    sweep       {dotted.key: [v1, v2, ...]} -> one entry per value, named
                <name>_<value>, with `x` = value (for figures)
    grid        {k1: [...], k2: [...]} -> cartesian product, named
                <name>_<v1>_<v2>
    eval_of     name of a training entry: evaluate its checkpoints with this
                entry's eval/solve overrides instead of training
    eval_seeds  eval seeds (independent search attempts), default [0]
    indices_file, split
                evaluate an explicit puzzle list (one index per line)
    train_only  train, do not evaluate
    command     baselines: argv after `modal run`; `result` names the local
                JSON file the command writes
    label, group, x
                passed through to results for tables and figures
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import yaml

from lattice_diffusion import config as cfglib
from lattice_diffusion.eval.records import EvalSpec, summary_path
from lattice_diffusion.provenance import app_run_name, entry_key

from experiments.registry import experiment_class

if TYPE_CHECKING:
    from experiments.common import Experiment

REPO = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO / "paper" / "manifests"
CHECKPOINT_MOUNT = "/checkpoints"           # lattice_diffusion.modal.image.CHECKPOINT_MOUNT
EVAL_ROOT = f"{CHECKPOINT_MOUNT}/eval"      # lattice_diffusion.eval.modal_app.EVAL_ROOT

_ENTRY_KEYS = {"name", "exp_id", "config", "eval", "seeds", "set", "sweep", "grid", "eval_of",
               "eval_seeds", "indices_file", "split", "train_only", "command", "result",
               "label", "group", "x"}
_MANIFEST_KEYS = {"exp_id", "benchmark", "config", "eval", "seeds", "set", "table", "entries"}

RunKind = Literal["train+eval", "train", "eval", "command"]


class TableHints(TypedDict, total=False):
    """The `table:` block of a manifest; each table/figure script reads its own keys."""

    reference: str              # entry the ablation rows are compared to
    groups: dict[str, str]      # entry group -> section heading
    per_order: list[int]        # Snowflake: one accuracy column per order
    caption: str
    maze_step: int | Literal["best", "last"]    # which baseline checkpoint to report
    histogram_x: int            # Figure 2: training steps of the histogram panel


def volume_path(path: str | Path) -> str:
    """Path on the checkpoint volume for a path under the container mount."""
    p = str(path)
    if not p.startswith(CHECKPOINT_MOUNT + "/"):
        raise ValueError(f"{p} is not under {CHECKPOINT_MOUNT}")
    return p[len(CHECKPOINT_MOUNT) + 1:]


def load_presets(path: Path = MANIFEST_DIR / "_eval.yaml") -> dict[str, dict]:
    return yaml.safe_load(path.read_text())["presets"]


@dataclass
class Run:
    """One resolved run of a manifest entry (one seed and eval seed)."""

    manifest: str
    exp_id: str
    entry: str                      # entry name (after sweep/grid expansion)
    key: str                        # <exp_id>/<entry>/seed<s>[/eval<e>]
    kind: RunKind
    benchmark: str
    seed: int | None = None
    eval_seed: int = 0
    config: str | None = None       # repo-relative config path
    overrides: dict[str, Any] = field(default_factory=dict)   # dotted keys
    exp: Experiment | None = None
    checkpoint: str | None = None   # container path under /checkpoints
    train_hash: str | None = None
    spec: EvalSpec | None = None
    train_key: str | None = None    # run key of the training run (eval-only runs)
    indices_file: str | None = None
    split: str | None = None
    command: list[str] | None = None
    result: str | None = None       # local result file (command runs)
    label: str = ""
    group: str = ""
    x: Any = None

    @property
    def eval_hash(self) -> str | None:
        return self.spec.spec_hash() if self.spec else None

    @property
    def run_dir(self) -> str | None:
        return str(self.spec.run_dir(EVAL_ROOT)) if self.spec else None

    @property
    def summary_file(self) -> str | None:
        return str(summary_path(Path(self.run_dir), self.spec.stop)) if self.spec else None

    @property
    def train_record(self) -> str | None:
        return str(Path(self.checkpoint).with_suffix(".train.json")) if self.checkpoint else None

    def app_name(self, with_eval: bool | None = None) -> str:
        """Modal run name; includes the eval hash for eval runs (or when `with_eval`)."""
        with_eval = self.kind == "eval" if with_eval is None else with_eval
        entry = self.entry + (f"-s{self.seed}" if self.seed is not None else "")
        return app_run_name(self.exp_id, entry, self.train_hash or "nohash",
                            self.eval_hash if with_eval else None)


def _scalar(value) -> str:
    """A value as YAML text: bare when it reads back as itself, else JSON."""
    if isinstance(value, str) and value and " " not in value and yaml.safe_load(value) == value:
        return value
    return json.dumps(value)


def _tokens(overrides: dict[str, Any]) -> list[str]:
    """Dotted overrides -> `key=value` tokens (parsed back with YAML)."""
    return [f"{k}={_scalar(v)}" for k, v in overrides.items()]


def _name_part(value) -> str:
    """Value as an entry-name suffix (0.5 -> 0p5, -1 -> m1)."""
    return str(value).replace(".", "p").replace("-", "m")


def _expand(entry: dict) -> list[dict]:
    """Sweep/grid entries -> plain entries."""
    if "sweep" in entry and "grid" in entry:
        raise ValueError(f"entry {entry['name']}: use sweep or grid, not both")
    axes = entry.get("sweep") or entry.get("grid")
    if not axes:
        return [entry]
    if "sweep" in entry and len(axes) != 1:
        raise ValueError(f"entry {entry['name']}: sweep takes one key; use grid")
    keys = list(axes)
    out = []
    for values in itertools.product(*(axes[k] for k in keys)):
        e = {k: v for k, v in entry.items() if k not in ("sweep", "grid")}
        e["name"] = "_".join([entry["name"], *(_name_part(v) for v in values)])
        e["set"] = {**entry.get("set", {}), **dict(zip(keys, values))}
        if "sweep" in entry:
            e.setdefault("x", values[0])
        out.append(e)
    return out


def _read_indices(rel_path: str) -> list[int] | None:
    """An explicit puzzle list (one index per line), or None if not written yet."""
    path = REPO / rel_path
    return [int(t) for t in path.read_text().split()] if path.exists() else None


def _resolve(benchmark: str, config: str | None, overrides: dict[str, Any]):
    cls = experiment_class(benchmark)
    return cfglib.load(cls, str(REPO / config) if config else None, _tokens(overrides))


def table_hints(name: str, manifest_dir: Path = MANIFEST_DIR) -> TableHints:
    return yaml.safe_load((manifest_dir / f"{name}.yaml").read_text()).get("table") or {}


def load_manifest(name: str, manifest_dir: Path = MANIFEST_DIR,
                  presets: dict[str, dict] | None = None) -> list[Run]:
    """Parse `<manifest_dir>/<name>.yaml` and expand it into runs."""
    path = manifest_dir / f"{name}.yaml"
    m = yaml.safe_load(path.read_text())
    unknown = set(m) - _MANIFEST_KEYS
    if unknown:
        raise KeyError(f"{path}: unknown manifest keys {sorted(unknown)}")
    presets = presets if presets is not None else load_presets(manifest_dir / "_eval.yaml")
    benchmark = m["benchmark"]

    entries = [e for raw in m["entries"] for e in _expand(raw)]
    names = [e["name"] for e in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate entry names")
    by_name = {e["name"]: e for e in entries}
    runs: list[Run] = []
    train_runs: dict[tuple[str, int], Run] = {}

    for e in entries:
        unknown = set(e) - _ENTRY_KEYS
        if unknown:
            raise KeyError(f"{path}: entry {e['name']}: unknown keys {sorted(unknown)}")
        exp_id = e.get("exp_id", m.get("exp_id", ""))
        if not exp_id:
            raise ValueError(f"{path}: entry {e['name']} has no exp_id")
        common = dict(manifest=name, exp_id=exp_id, entry=e["name"], benchmark=benchmark,
                      label=e.get("label", e["name"]), group=e.get("group", ""),
                      x=e.get("x"))

        if "command" in e:
            runs.append(Run(key=entry_key(exp_id, e["name"], 0), kind="command",
                            command=list(e["command"]), result=e.get("result"), **common))
            continue

        train_entry = by_name[e["eval_of"]] if "eval_of" in e else e
        config = train_entry.get("config", m.get("config"))
        seeds = e.get("seeds", train_entry.get("seeds", m.get("seeds", [0])))
        preset = presets[e.get("eval", m.get("eval", "whole"))]
        train_set = {**m.get("set", {}), **train_entry.get("set", {})}
        eval_set = e.get("set", {}) if "eval_of" in e else {}
        bad = [k for k in eval_set if k.split(".")[0] not in ("eval", "solve")]
        if bad:
            raise ValueError(f"{path}: eval-only entry {e['name']} sets {bad}; "
                             f"only eval.* and solve.* may differ from {e['eval_of']}")

        kind: RunKind = ("eval" if "eval_of" in e or "indices_file" in e
                         else "train" if e.get("train_only") else "train+eval")
        # An explicit list that is not written yet leaves the run without a spec (blocked).
        indices = _read_indices(e["indices_file"]) if "indices_file" in e else None
        has_spec = kind != "train" and ("indices_file" not in e or indices is not None)
        for seed in seeds:
            for eval_seed in e.get("eval_seeds", [0]):
                key = entry_key(exp_id, e["name"], seed, eval_seed)
                overrides = {**train_set, **preset, **eval_set, "train.seed": seed,
                             "eval.eval_seed": eval_seed,
                             "provenance.exp_id": exp_id, "provenance.entry": key}
                exp = _resolve(benchmark, config, overrides)
                ckpt = str(exp.checkpoint_path(CHECKPOINT_MOUNT))
                spec = None
                if has_spec:
                    spec = exp.eval_spec(ckpt, explicit_indices=indices,
                                         explicit_split=e.get("split", "test") if indices else None)
                run = Run(key=key, kind=kind, seed=seed, eval_seed=eval_seed, config=config,
                          overrides=overrides, exp=exp, checkpoint=ckpt,
                          train_hash=exp.train_hash(), spec=spec,
                          indices_file=e.get("indices_file"), split=e.get("split"), **common)
                if kind == "eval":
                    trainer = train_runs.get((e.get("eval_of", e["name"]), seed))
                    if "eval_of" in e and trainer is None:
                        raise ValueError(f"{path}: {e['name']} evaluates {e['eval_of']} "
                                         f"seed {seed}, which is not declared before it")
                    if trainer is not None and trainer.train_hash != run.train_hash:
                        raise ValueError(f"{path}: {e['name']} changes training settings of "
                                         f"{e['eval_of']}")
                    run.train_key = trainer.key if trainer else None
                else:
                    train_runs[(e["name"], seed)] = run
                runs.append(run)
    return runs


def eval_overrides(run: Run) -> list[str]:
    """`--set` tokens for experiments/evaluate.py: the run's eval and solve blocks plus provenance.

    Passing the whole blocks makes the remote spec equal the locally resolved
    one, whatever config the checkpoint was trained under.
    """
    exp = run.exp
    derived = set(getattr(type(exp), "DERIVED", ()))
    flat = {f"eval.{k}": v for k, v in cfglib.flatten(cfglib.to_dict(exp.eval)).items()}
    flat.update({f"solve.{k}": v for k, v in cfglib.flatten(cfglib.to_dict(exp.solve)).items()})
    flat = {k: v for k, v in flat.items() if k not in derived}
    flat["provenance.exp_id"] = run.exp_id
    flat["provenance.entry"] = run.key
    return _tokens(flat)


def train_overrides(run: Run) -> list[str]:
    """`--set` tokens for a run's training entry point."""
    return _tokens(run.overrides)


def list_manifests(manifest_dir: Path = MANIFEST_DIR) -> list[str]:
    return sorted(str(p.relative_to(manifest_dir).with_suffix(""))
                  for p in manifest_dir.rglob("*.yaml") if not p.name.startswith("_"))
