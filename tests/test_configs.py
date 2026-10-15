import importlib
import tempfile
from pathlib import Path

import pytest
import torch

from lattice_diffusion import config as cfglib
from lattice_diffusion.eval.stop import FixedN, stop_from_dict
from experiments.registry import experiment_class
from tests.equivalence import fakes, train_runs

CONFIGS = Path(__file__).parent.parent / "configs"
BENCHMARKS = ["sudoku", "snowflake", "maze"]


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_paper_yaml_is_the_class_default(benchmark):
    cls = experiment_class(benchmark)
    loaded = cfglib.load(cls, CONFIGS / benchmark / "paper.yaml")
    assert cfglib.to_dict(loaded) == cfglib.to_dict(cls())


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*/*.yaml")), ids=lambda p: p.stem)
def test_every_config_loads(path):
    exp = cfglib.load(experiment_class(path.parent.name), path)
    exp.eval_spec("/ckpt/x.pt")


def test_overrides_are_typed_and_strict():
    cls = experiment_class("sudoku")
    exp = cfglib.load(cls, None, "train.steps=2000 model.use_rope=true data.n_puzzles=null "
                                 "train.betas=[0.9,0.99]")
    assert exp.train.steps == 2000 and exp.model.use_rope is True
    assert exp.data.n_puzzles is None and exp.train.betas == (0.9, 0.99)
    with pytest.raises(KeyError):
        cfglib.load(cls, None, "train.stpes=2000")


def test_derived_keys_cannot_be_set():
    with pytest.raises(KeyError):
        cfglib.load(experiment_class("maze"), None, "model.seq_len=100")
    exp = cfglib.load(experiment_class("maze"), CONFIGS / "maze" / "ksweep.yaml")
    assert (exp.model.grid_rows, exp.model.seq_len) == (15, 225)


def test_conflict_head_must_match_its_loss():
    with pytest.raises(ValueError):
        cfglib.load(experiment_class("sudoku"), None, "train.conflict_loss_weight=0")
    exp = cfglib.load(experiment_class("sudoku"), None,
                      "train.conflict_loss_weight=0 model.cls_token=false")
    assert not exp.model.cls_token


def test_checkpoint_path_tracks_training_config_only():
    cls = experiment_class("sudoku")
    base = cls()
    same = cfglib.load(cls, None, "train.monitor_every=7 train.log_every=3 eval.stop=full "
                                  "solve.max_rounds=5 data.cache_dir=/elsewhere")
    assert same.train_hash() == base.train_hash()
    assert cfglib.load(cls, None, "train.steps=2000").train_hash() != base.train_hash()
    seed1 = cfglib.load(cls, None, "train.seed=1")
    assert seed1.train_hash() == base.train_hash()
    assert seed1.checkpoint_path("/c") != base.checkpoint_path("/c")


def test_json_round_trip():
    for benchmark in BENCHMARKS:
        cls = experiment_class(benchmark)
        exp = cls()
        assert cfglib.to_dict(cls.from_json(exp.to_json())) == cfglib.to_dict(exp)


def test_ablation_config_uses_first_1000_test_puzzles():
    exp = cfglib.load(experiment_class("sudoku"), CONFIGS / "sudoku" / "ablation.yaml")
    spec = exp.eval_spec("/ckpt/a.pt")
    assert spec.index_list == "sudoku_test_perm"
    assert stop_from_dict(spec.stop) == FixedN(1000)


def test_maze_synthetic_eval_spec():
    exp = cfglib.load(experiment_class("maze"), CONFIGS / "maze" / "ksweep.yaml")
    spec = exp.eval_spec("/ckpt/m.pt")
    assert spec.index_list == "explicit" and spec.explicit_split == "synthetic"
    assert spec.explicit_indices == list(range(1000))
    assert spec.solve["maze_data"]["grid_size"] == 15
    assert spec.solve["maze_data"]["n_puzzles"] == 1000


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_monitoring_does_not_change_trained_weights(benchmark, monkeypatch):
    """Monitoring solves run under a forked RNG, so a run with monitoring
    every step trains the same weights as the reference run without it."""
    module = importlib.import_module(f"experiments.{benchmark}.train")
    calls = []
    real = module.monitor_solve
    monkeypatch.setattr(module, "monitor_solve", lambda *a, **k: calls.append(1) or real(*a, **k))
    with tempfile.TemporaryDirectory() as tmp:
        state = train_runs.RUNS[benchmark](Path(tmp), monitor_every=1, monitor_n_puzzles=4,
                                           monitor_max_rounds=2, monitor_n_chains=4)
    assert len(calls) == 3
    ref = torch.load(fakes.FIXTURES / "train.pt", weights_only=False)[benchmark]
    assert fakes.state_dict_equal(state, ref)
