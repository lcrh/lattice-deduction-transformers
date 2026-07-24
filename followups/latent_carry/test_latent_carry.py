from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.solve import SolveConfig, solve
from followups.latent_carry import configs
from lattice_diffusion.models.looped_transformer import (
    LoopedTransformerConfig,
    PowersetModel,
)


def _cfg(carry_mode: str = "off", *, n_loops: int = 4) -> LoopedTransformerConfig:
    return LoopedTransformerConfig(
        dim=8,
        seq_len=4,
        n_channels=3,
        grid_rows=2,
        grid_cols=2,
        num_layers=1,
        n_heads=2,
        n_loops=n_loops,
        dropout=0.0,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        cls_token=True,
        carry_mode=carry_mode,
    )


def test_off_mode_keeps_legacy_state_dict_and_outputs():
    torch.manual_seed(0)
    default = PowersetModel(_cfg()).eval()
    explicit = PowersetModel(_cfg("off")).eval()
    explicit.load_state_dict(default.state_dict())
    assert not any("carry_proj" in key for key in default.state_dict())

    x = torch.randn(2, 4, 3)
    with torch.no_grad():
        a = default(x, return_all=True)
        b = explicit(x, return_all=True)
    for key in ("bce", "softmax", "cls", "conflict"):
        for av, bv in zip(a[key], b[key], strict=True):
            torch.testing.assert_close(av, bv, rtol=0, atol=0)
    assert "carry_out" not in a


def test_h_mode_carries_final_hidden_and_zero_is_noop():
    torch.manual_seed(7)
    baseline = PowersetModel(_cfg("off")).eval()
    torch.manual_seed(7)
    model = PowersetModel(_cfg("h")).eval()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(2, 4, 3)
    zeros = torch.zeros(2, 4, 8)
    with torch.no_grad():
        baseline_out = baseline(x, return_all=True)
        absent = model(x, return_all=True)
        zero = model(x, carry=zeros, return_all=True)
    assert len(absent["bce"]) == 4
    assert absent["carry_out"].shape == (2, 4, 8)
    torch.testing.assert_close(
        baseline_out["bce"][-1], absent["bce"][-1], rtol=0, atol=0,
    )
    torch.testing.assert_close(absent["bce"][-1], zero["bce"][-1], rtol=0, atol=0)

    with torch.no_grad():
        model.carry_proj.weight.copy_(torch.eye(8))
        changed = model(x, carry=torch.ones_like(zeros), use_final=True)
    assert not torch.equal(absent["bce"][-1], changed["bce"])


def test_z_mode_compute_match_and_answer_readouts():
    model = PowersetModel(_cfg("z")).eval()
    calls = 0

    def count_call(_module, _args, _output):
        nonlocal calls
        calls += 1

    handle = model.backbone.register_forward_hook(count_call)
    state = torch.randn(2, 4, 3)
    orig_x = torch.randn(2, 4, 3)
    try:
        with torch.no_grad():
            out = model(state, orig_x=orig_x, return_all=True)
    finally:
        handle.remove()

    assert calls == 4
    assert len(out["bce"]) == 2
    assert len(out["conflict"]) == 2
    assert out["carry_out"].shape == (2, 4, 8)

    try:
        model(state)
    except ValueError as exc:
        assert "orig_x" in str(exc)
    else:
        raise AssertionError("z mode accepted a forward without orig_x")


def test_dpll_threads_detached_carry():
    model = PowersetModel(_cfg("h")).eval()
    state = torch.ones(2, 4, 3)
    given = torch.zeros(2, 4, dtype=torch.bool)
    cfg = StepConfig(
        threshold=0.0,
        temp_decide=0.0,
        augment=False,
        carry_latent="h",
    )
    _, _, _, _, info = dpll_step(
        model,
        state,
        given,
        cfg,
        orig_x=state,
        carry=torch.zeros(2, 4, 8),
        want_stats=False,
    )
    assert info["carry_out"].shape == (2, 4, 8)
    assert not info["carry_out"].requires_grad


class _ConflictThenRecord(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(dim=5)
        self.seen_carries: list[torch.Tensor] = []

    def forward(self, state, use_final=True, *, orig_x=None, carry=None):
        assert orig_x is not None
        self.seen_carries.append(carry.detach().clone())
        batch, cells, channels = state.shape
        return {
            "bce": torch.full_like(state, 10.0),
            "softmax": torch.zeros_like(state),
            "conflict": torch.full((batch, 1), 10.0, device=state.device),
            "carry_out": torch.ones(batch, cells, 5, device=state.device),
        }


def test_solve_zeros_carry_after_conflict_reset():
    model = _ConflictThenRecord()
    puzzle = torch.ones(1, 4, 3)
    truth = torch.zeros_like(puzzle)
    truth[..., 0] = 1
    given = torch.zeros(1, 4, dtype=torch.bool)
    result = solve(
        model,
        puzzle,
        truth,
        given,
        SolveConfig(
            step=StepConfig(
                augment=False,
                carry_latent="h",
                cls_threshold=0.5,
            ),
            max_rounds=2,
            n_chains=1,
            batch_size=1,
        ),
        verbose=False,
    )
    assert result.timeouts.tolist() == [True]
    assert len(model.seen_carries) == 2
    assert torch.count_nonzero(model.seen_carries[0]) == 0
    assert torch.count_nonzero(model.seen_carries[1]) == 0


def test_e6_matrix_is_fixed_frame_and_complete():
    assert set(configs.CONFIGS) == {
        "baseline", "carry_h", "carry_z", "zero_carry_z",
    }
    assert len(list(configs.iter_runs())) == 12
    expected = {
        "baseline": "off",
        "carry_h": "h",
        "carry_z": "z",
        "zero_carry_z": "zero_z",
    }
    for name, policy in expected.items():
        flags = configs.effective_flags(name)
        assert flags["carry_latent"] == policy
        assert flags["augment"] is False
        assert flags["data_augment_digit_perm"] is True
        assert flags["data_augment_dihedral"] is True
        command = configs.launch_command(name, 0)
        assert "--ckpt-subdir followups/e6" in command
        assert f"--carry-latent {policy}" in command
        assert "--no-augment" in command
        assert "--data-augment-digit-perm" in command
        assert "--data-augment-dihedral" in command
        assert "--augment False" not in command
