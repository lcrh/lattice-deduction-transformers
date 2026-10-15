"""Sudoku-Extreme adapter for the shared evaluator (lattice_diffusion.eval).

Every Sudoku-Extreme puzzle is satisfiable and evaluation is zero-hint with
no data augmentation, so an eval sample is exactly the cached (x, y) pair;
we index the cache directly instead of going through the training sampler.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from lattice_diffusion.data.sudoku_extreme import _download_and_cache
from lattice_diffusion.eval.records import EvalSpec
from lattice_diffusion.eval.stop import PRESETS
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, solve


def default_solve() -> dict:
    """Paper eval settings: eval-time CLS threshold 0.6, 64 chains, 1000
    rounds per puzzle, dropout 0.05 at eval."""
    return {
        "step": asdict(StepConfig(cls_threshold=0.6, augment=True)),
        "max_rounds": 1000,
        "n_chains": 64,
        "batch_size": 512,
        "dropout_p": 0.05,
    }


def build_solve_config(solve_dict: dict) -> SolveConfig:
    d = dict(solve_dict)
    d.pop("dropout_p", None)
    return SolveConfig(step=StepConfig(**d.pop("step")), **d)


def make_eval_spec(
    checkpoint: str,
    index_list: str = "sudoku_test_perm",
    stop: str = "sequential",
    workers: int = 32,
    max_in_flight: int = 0,
    keep_warm_s: int = 300,
    block_size: int = 512,
    eval_seed: int = 0,
    tag: str = "",
    compile: bool = False,
    threshold: float | None = None,
    temp_decide: float | None = None,
    cls_threshold: float | None = None,
    augment: bool | None = None,
    n_chains: int | None = None,
    batch_size: int | None = None,
    max_rounds: int | None = None,
    dropout_p: float | None = None,
    explicit_indices: list[int] | None = None,
    split: str | None = None,
) -> EvalSpec:
    """Build an EvalSpec from paper eval defaults plus any overrides."""
    solve = default_solve()
    for key, val in (("threshold", threshold), ("temp_decide", temp_decide),
                     ("cls_threshold", cls_threshold), ("augment", augment)):
        if val is not None:
            solve["step"][key] = val
    for key, val in (("n_chains", n_chains), ("batch_size", batch_size),
                     ("max_rounds", max_rounds), ("dropout_p", dropout_p)):
        if val is not None:
            solve[key] = val
    if explicit_indices is not None:
        index_list = "explicit"
    return EvalSpec(
        task="sudoku",
        checkpoint=checkpoint,
        index_list=index_list,
        solve=solve,
        stop=PRESETS[stop].to_dict(),
        block_size=block_size,
        eval_seed=eval_seed,
        compile=compile,
        tag=tag or index_list,
        explicit_indices=explicit_indices,
        explicit_split=split if explicit_indices is not None else None,
        n_workers=workers,
        max_in_flight=max_in_flight or None,
        keep_warm_s=keep_warm_s,
    )



def load_eval_model(checkpoint: str, dropout_p: float, device: torch.device,
                    compile: bool = False) -> nn.Module:
    """Load a PowersetModel checkpoint for evaluation.

    EMA weights are swapped in when present. With dropout_p > 0 every
    nn.Dropout and MHA attention dropout is set to dropout_p and the model
    is left in train() mode: dropout acts as test-time noise for the search.
    """
    ckpt = load_checkpoint(checkpoint)
    model = PowersetModel(LoopedTransformerConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
    if dropout_p > 0.0:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout_p
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = dropout_p
        model.train()
    if compile and device.type == "cuda":
        model = torch.compile(model, dynamic=False)
    return model


def grid_string(solution: torch.Tensor) -> str:
    """[81, 9] lattice -> 81-char digit string, '0' for non-singleton cells."""
    single = solution.sum(dim=-1) == 1
    digits = (solution.argmax(dim=-1) + 1) * single.long()
    return "".join(str(d) for d in digits.tolist())


def is_valid_solution(grid: str, puzzle: torch.Tensor) -> bool:
    """Rule check: complete grid, givens kept, rows/cols/boxes all-different."""
    if len(grid) != 81 or "0" in grid:
        return False
    cells = [int(ch) for ch in grid]
    given = puzzle.sum(dim=-1) == 1
    given_digits = (puzzle.argmax(dim=-1) + 1).tolist()
    for i in range(81):
        if given[i] and cells[i] != given_digits[i]:
            return False
    full = set(range(1, 10))
    for r in range(9):
        if set(cells[r * 9:(r + 1) * 9]) != full:
            return False
    for c in range(9):
        if set(cells[c::9]) != full:
            return False
    for br in range(3):
        for bc in range(3):
            box = [cells[(br * 3 + i) * 9 + bc * 3 + j] for i in range(3) for j in range(3)]
            if set(box) != full:
                return False
    return True


class SudokuTask:
    name = "sudoku"

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self._pools: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _pool(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        if split not in self._pools:
            data = torch.load(_download_and_cache(self.data_dir, split),
                              map_location="cpu", weights_only=True)
            self._pools[split] = (data["x"], data["y"])
        return self._pools[split]

    def split_size(self, split: str) -> int:
        return self._pool(split)[0].shape[0]

    def default_solve(self) -> dict:
        return default_solve()

    def load_model(self, checkpoint: str, dropout_p: float, compile: bool):
        return load_eval_model(checkpoint, dropout_p, self.device, compile)

    def warmup(self, model) -> None:
        self.solve_block(model, "test", np.arange(8), self.default_solve())

    def solve_block(self, model, split: str, indices: np.ndarray, solve_dict: dict) -> list[dict]:
        x_all, y_all = self._pool(split)
        idx = torch.as_tensor(indices, dtype=torch.long)
        x = x_all.index_select(0, idx).to(self.device).float()
        y = y_all.index_select(0, idx).to(self.device).float()
        given_mask = x.sum(dim=-1) == 1
        cfg = build_solve_config(solve_dict)
        with torch.no_grad():
            res = solve(model, x, y, given_mask, cfg, verbose=False)

        correct, wrong = res.correct.tolist(), res.wrong.tolist()
        rounds, calls = res.round_solved.tolist(), res.puzzle_calls.tolist()
        resets = res.n_resets.tolist()
        solution, x = res.solution.cpu(), x.cpu()
        rows = []
        for j, dataset_idx in enumerate(indices.tolist()):
            status = "correct" if correct[j] else ("wrong" if wrong[j] else "timeout")
            grid = grid_string(solution[j]) if status != "timeout" else ""
            rows.append({
                "idx": dataset_idx,
                "status": status,
                "rounds": rounds[j],
                "calls": calls[j],
                "forwards": ((rounds[j] + 1) if correct[j] else cfg.max_rounds) * cfg.n_chains,
                "n_resets": resets[j],
                "solution": grid,
                "verified": bool(grid) and is_valid_solution(grid, x[j]),
            })
        return rows
