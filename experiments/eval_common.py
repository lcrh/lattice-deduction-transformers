"""Helpers shared by the benchmark adapters (experiments/<benchmark>/eval_task.py)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from lattice_diffusion import config as cfglib
from lattice_diffusion.eval.tasks import TaskBlock
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, SolveResult


def eval_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwrap(model: nn.Module) -> nn.Module:
    """The module inside a torch.compile wrapper."""
    return getattr(model, "_orig_mod", model)


def build_solve_config(solve_dict: dict) -> SolveConfig:
    """SolveConfig from an EvalSpec's solve dict; `dropout_p` belongs to model loading."""
    return cfglib.from_dict(SolveConfig, {k: v for k, v in solve_dict.items() if k != "dropout_p"})


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


def result_rows(res: SolveResult, indices: np.ndarray, cfg: SolveConfig, solutions: list[str],
                verified: list[bool], given_mask: torch.Tensor,
                extra: list[dict] | None = None) -> TaskBlock:
    """One eval row per puzzle (see lattice_diffusion.eval.scheduler.Row)."""
    correct, wrong = res.correct.tolist(), res.wrong.tolist()
    rounds, calls = res.round_solved.tolist(), res.puzzle_calls.tolist()
    # Forwards count deduction passes: rounds times the puzzle's mean passes per round.
    multi_pass = cfg.step.max_passes > 1
    held = res.puzzle_rounds.tolist() if multi_pass else None
    resets = res.n_resets.tolist()
    n_givens = given_mask.sum(dim=-1).tolist()
    diag = res.puzzle_diag.tolist()
    pass_diag = res.puzzle_pass_diag.tolist() if res.puzzle_pass_diag is not None else None
    seq = None
    if cfg.estimate_sequential:
        seq = list(zip(res.forwards_seq.tolist(), res.seq_winning_idx.tolist(),
                       res.seq_attempts_done.tolist()))
    rows = []
    for j, dataset_idx in enumerate(indices.tolist()):
        status = "correct" if correct[j] else ("wrong" if wrong[j] else "timeout")
        passes = calls[j] / held[j] if multi_pass and held[j] > 0 else 1
        row = {
            "idx": dataset_idx,
            "status": status,
            "rounds": rounds[j],
            "calls": calls[j],
            "forwards": round(((rounds[j] + 1) if correct[j] else cfg.max_rounds)
                              * cfg.n_chains * passes),
            "n_resets": resets[j],
            "n_givens": int(n_givens[j]),
            "solution": solutions[j] if status != "timeout" else "",
            "verified": verified[j] if status != "timeout" else False,
            "deduced": diag[j][0],
            "unsound": diag[j][1],
            "cls": diag[j][2:],
        }
        if seq is not None:
            fwd, win, done = seq[j]
            fwd = round(fwd * passes)
            row.update(forwards_seq=fwd, seq_winning_idx=win, seq_attempts_done=done,
                       seq_avg_attempt_len=fwd / (win + 1) if win >= 0 else None)
        if pass_diag is not None:
            row["pass_deduced_unsound"] = pass_diag[j]
        rows.append(row | (extra[j] if extra else {}))
    return TaskBlock(rows, res.model_calls)
