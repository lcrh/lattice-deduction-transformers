"""Model / training ablation options: latent carry, deep supervision, and
the cross-option checks in the experiment configs."""

import tempfile
from pathlib import Path

import pytest
import torch

from lattice_diffusion import config as cfglib
from lattice_diffusion.models.looped_transformer import PowersetModel
from experiments.registry import experiment_class
from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.solve import SolveConfig, solve
from experiments.sudoku.train import _losses
from tests.equivalence import fakes


def test_default_state_dict_schema_is_unchanged():
    ref = torch.load(fakes.FIXTURES / "dpll.pt", weights_only=False)["sudoku/aug"]["model_state"]
    assert fakes.tiny_model().state_dict().keys() == ref.keys()


def test_carry_h_without_a_latent_is_the_plain_model():
    plain = fakes.tiny_model(seed=0).eval()
    carry = fakes.tiny_model(seed=0, carry="h").eval()
    assert {k: v for k, v in carry.state_dict().items() if not k.startswith("carry_proj")}.keys() \
        == plain.state_dict().keys()
    x = fakes.random_lattice(2, 81, 9, seed=1)
    a, b = plain(x, use_final=True), carry(x, use_final=True)
    assert torch.equal(a["bce"], b["bce"])
    assert b["carry_out"].shape == (2, 81, 16)


@pytest.mark.parametrize("mode", ["z", "zero_z"])
def test_scratchpad_modes(mode):
    model = fakes.tiny_model(seed=0, carry=mode).eval()
    x = fakes.random_lattice(2, 81, 9, seed=1)
    with pytest.raises(ValueError):
        model(x, use_final=True)                      # needs the puzzle
    out = model(x, return_all=True, orig_x=x)
    assert len(out["bce"]) == 1                       # n_loops=2 -> one cycle
    assert ("carry_out" in out) == (mode == "z")
    with pytest.raises(ValueError):
        fakes.tiny_model_cfg(carry=mode, n_loops=3)   # cycles need even n_loops


@pytest.mark.parametrize("mode", ["h", "z", "zero_z"])
def test_carry_trains_and_solves(mode):
    with tempfile.TemporaryDirectory() as tmp:
        state = _train_with_carry(Path(tmp), mode)
    model = PowersetModel(fakes.tiny_model_cfg(carry=mode))
    model.load_state_dict(state)
    x, y = fakes.sudoku_tensors(4, (2, 5), seed=3)
    res = solve(model, x.float(), y.float(), x.float().sum(-1) == 1,
                SolveConfig(step=StepConfig(augment=False), max_rounds=4, n_chains=4,
                            batch_size=8), verbose=False)
    assert torch.equal(res.correct | res.wrong | res.timeouts, torch.ones(4, dtype=torch.bool))


def _train_with_carry(tmp: Path, mode: str) -> dict:
    from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig
    from experiments.sudoku.config import SudokuExperiment
    from experiments.sudoku.train import TrainConfig, train

    fakes.write_sudoku_caches(tmp)
    exp = SudokuExperiment(
        model=fakes.tiny_model_cfg(carry=mode),
        data=SudokuExtremeConfig(cache_dir=str(tmp), n_puzzles=100),
        train=TrainConfig(steps=3, batch_size=16, step=StepConfig(augment=False),
                          compile=False, monitor_every=0),
        solve=SolveConfig(step=StepConfig(augment=False)),
    )
    with fakes.small_sudoku_val():
        train(exp, tmp / "c.pt")
    return torch.load(tmp / "c.pt", weights_only=False)["model_state_dict"]


def test_final_only_supervision_uses_the_last_readout():
    torch.manual_seed(0)
    B, S, C, L = 3, 81, 9, 4
    out = {"bce": [torch.randn(B, S, C) for _ in range(L)],
           "softmax": [torch.randn(B, S, C) for _ in range(L)],
           "conflict": [torch.randn(B, 1) for _ in range(L)]}
    state = fakes.random_lattice(B, S, C, seed=2)
    y = torch.nn.functional.one_hot(torch.randint(0, C, (B, S)), C).float()
    args = (state, y, torch.zeros(B, S, dtype=torch.bool), torch.ones(B, dtype=torch.bool),
            torch.zeros(B, dtype=torch.bool), 4.0, 0.5, 0.2, 0.1)
    last = {k: v[-1:] for k, v in out.items()}
    assert torch.isclose(_losses(out, *args, supervise="final"), _losses(last, *args))
    assert not torch.isclose(_losses(out, *args), _losses(last, *args))


def test_cross_option_checks():
    sudoku = experiment_class("sudoku")
    with pytest.raises(ValueError):           # carry needs a fixed frame
        cfglib.load(sudoku, None, "model.carry=z")
    exp = cfglib.load(sudoku, None, "model.carry=z train.step.augment=false "
                                    "solve.step.augment=false")
    assert exp.model.carry == "z"
    with pytest.raises(ValueError):
        cfglib.load(sudoku, None, "model.carry=h train.step.augment=false "
                                  "solve.step.augment=false solve.backtrack=last")
    with pytest.raises(ValueError):
        cfglib.load(sudoku, None, "train.supervise=middle")
    with pytest.raises(ValueError):           # only the Sudoku trainer carries a latent
        cfglib.load(experiment_class("maze"), None, "model.carry=h train.step.augment=false "
                                                    "solve.step.augment=false")


def test_static_transformer_is_one_loop_of_a_deeper_backbone():
    exp = cfglib.load(experiment_class("sudoku"), None, "model.n_loops=1 model.num_layers=16")
    model = PowersetModel(exp.model)
    assert len(model.backbone.layers) == 16
    assert len(model(torch.ones(1, 81, 9), return_all=True)["bce"]) == 1
