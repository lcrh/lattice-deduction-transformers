"""Fast regression checks for Tier-2 isolation (hooks + launchers)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import torch

from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.hooks import public_asdict
from experiments.sudoku.solve import SolveConfig
from experiments.sudoku.train import TrainConfig
from followups.deduction_operator import configs as e3_configs
from followups.deduction_operator.step_extension import (
    attach_step_extension, make_extended_step,
)
from followups.search_process import configs as e2_configs
from followups.search_process.pool import make_pool_strategy
from followups.search_process.search import make_search_strategy


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


class _TinyModel(torch.nn.Module):
    """Deterministic stub: BCE favors killing non-argmax digits slowly."""

    def forward(self, state, use_final=True):
        # state [B,S,C] in {0,1}; emit logits that keep alive bits high.
        bce = state * 4.0 - 2.0  # alive -> +2 (~0.88), dead -> -2
        sm = state * 2.0
        return {"bce": bce, "softmax": sm}


def test_public_asdict_drops_hooks():
    cfg = StepConfig(_extension=object())
    d = public_asdict(cfg)
    assert "_extension" not in d
    assert "threshold" in d


def test_make_extended_step_none_at_defaults():
    assert make_extended_step() is None
    assert make_extended_step(deduce_passes=2) is not None
    assert make_extended_step(cell_policy="mrv") is not None


def test_make_search_strategy_none_at_defaults():
    assert make_search_strategy() is None
    assert make_search_strategy(track_per_pass=True) is not None
    assert make_search_strategy(backtrack="last") is not None
    assert make_search_strategy(digit_policy="rank_k") is not None


def test_make_pool_strategy_none_at_defaults():
    assert make_pool_strategy() is None
    assert make_pool_strategy(backtrack="geometric") is not None


def test_legacy_dpll_single_pass_accounting():
    model = _TinyModel()
    B, S, C = 2, 4, 3
    state = torch.ones(B, S, C)
    given = torch.zeros(B, S, dtype=torch.bool)
    cfg = StepConfig(threshold=0.01, temp_decide=0.0, augment=False)
    new_state, conflict, solved, out, info = dpll_step(
        model, state, given, cfg, want_stats=True,
    )
    assert info["n_passes"] == 1
    assert info["per_pass_deduce_masks"] == []
    assert new_state.shape == state.shape


def test_extended_multipass_reports_n_passes():
    class _KillWeak(torch.nn.Module):
        def forward(self, state, use_final=True):
            # Prefer killing channel 0; leave others high so multi-pass can run.
            bce = torch.full_like(state, 4.0)
            bce[..., 0] = -4.0
            return {"bce": bce, "softmax": state * 2.0}

    model = _KillWeak()
    B, S, C = 2, 4, 3
    state = torch.ones(B, S, C)
    given = torch.zeros(B, S, dtype=torch.bool)
    cfg = StepConfig(threshold=0.5, temp_decide=0.0, augment=False)
    attach_step_extension(cfg, make_extended_step(deduce_passes=3))
    _, _, _, _, info = dpll_step(model, state, given, cfg, want_stats=False)
    # Pass 1 kills channel 0; pass 2 sees nothing left below threshold -> fixpoint.
    assert info["n_passes"] >= 2
    assert len(info["per_pass_deduce_masks"]) == info["n_passes"]
    assert info["n_passes"] <= 3


def test_search_snapshot_lifecycle_root_vs_last():
    root = make_search_strategy(backtrack="root", track_per_pass=True)
    assert root is not None
    assert root.needs_snapshots() is False
    assert root.attach(4, 9, 9, torch.device("cpu")) is None

    last = make_search_strategy(backtrack="last")
    assert last.needs_snapshots() is True
    snap = last.attach(4, 9, 9, torch.device("cpu"))
    assert snap is not None
    assert snap["snap_state"].shape[0] == 4
    last.on_fill(slice(0, 2), snap=snap)
    assert int(snap["chain_depth"][0].item()) == 0


def test_compose_step_and_search_hooks_on_configs():
    step = StepConfig()
    attach_step_extension(step, make_extended_step(cell_policy="mrv"))
    search = make_search_strategy(backtrack="last")
    solve_cfg = SolveConfig(step=step, _search=search)
    assert solve_cfg.step._extension is not None
    assert solve_cfg._search is not None
    pool = make_pool_strategy(backtrack="last")
    train_cfg = TrainConfig(_pool_strategy=pool)
    assert train_cfg._pool_strategy is not None


def test_e3_launcher_points_at_modal_eval():
    cmd = e3_configs.launch_command(next(iter(e3_configs.CONFIGS)), 0)
    assert "followups/deduction_operator/modal_eval.py" in cmd
    assert "experiments/sudoku/eval_only.py" not in cmd
    assert "--skip-if-done" in cmd
    assert "--ckpt-subdir" in cmd


def test_e2_launcher_points_at_followup_entrypoints():
    train_name = next(n for n, c in e2_configs.CONFIGS.items() if c["kind"] == "train")
    eval_name = next(n for n, c in e2_configs.CONFIGS.items() if c["kind"] == "eval")
    tcmd = e2_configs.launch_command(train_name, 0)
    ecmd = e2_configs.launch_command(eval_name, 0)
    assert "followups/search_process/modal_train.py" in tcmd
    assert "followups/search_process/modal_eval.py" in ecmd
    assert "experiments/sudoku/run.py" not in tcmd
    assert "experiments/sudoku/eval_only.py" not in ecmd


_BANNED = re.compile(r"\b(E2|E3)\b|\bphase\s*[12]\b|\bstudy\b", re.I)


def test_experiments_core_has_no_tier2_study_terms():
    bad: list[str] = []
    for path in EXPERIMENTS.rglob("*.py"):
        text = path.read_text()
        # Allow generic English "phase" in comments like "drain phase".
        for i, line in enumerate(text.splitlines(), 1):
            if "drain phase" in line or "drain-phase" in line:
                continue
            if _BANNED.search(line):
                bad.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not bad, "banned terminology in experiments/:\n" + "\n".join(bad[:40])


def test_core_files_parse():
    for rel in (
        "experiments/sudoku/dpll.py",
        "experiments/sudoku/solve.py",
        "experiments/sudoku/train.py",
        "experiments/sudoku/eval_only.py",
        "experiments/sudoku/run.py",
        "experiments/sudoku/eval_runner.py",
        "followups/deduction_operator/modal_eval.py",
        "followups/search_process/modal_eval.py",
        "followups/search_process/modal_train.py",
        "followups/deduction_operator/step_extension.py",
        "followups/search_process/search.py",
        "followups/search_process/pool.py",
    ):
        ast.parse((ROOT / rel).read_text(), filename=rel)


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(failed)
