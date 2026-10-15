"""Helpers shared by the benchmark adapters for lattice_diffusion.eval
(experiments/<b>/eval_task.py)."""

from __future__ import annotations

import torch
from torch import nn

from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig


def build_solve_config(solve_dict: dict) -> SolveConfig:
    """SolveConfig from an EvalSpec's solve dict (task-specific keys removed
    by the caller; `dropout_p` belongs to model loading)."""
    d = dict(solve_dict)
    d.pop("dropout_p", None)
    return SolveConfig(step=StepConfig(**d.pop("step")), **d)


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


def result_rows(res, indices, cfg: SolveConfig, solutions: list[str],
                verified: list[bool | None], extra: list[dict] | None = None) -> list[dict]:
    """One eval row per puzzle (schema in lattice_diffusion/eval/records.py)."""
    correct, wrong = res.correct.tolist(), res.wrong.tolist()
    rounds, calls = res.round_solved.tolist(), res.puzzle_calls.tolist()
    resets = res.n_resets.tolist()
    forwards_seq = res.forwards_seq.tolist() if cfg.estimate_sequential else None
    rows = []
    for j, dataset_idx in enumerate(indices.tolist()):
        status = "correct" if correct[j] else ("wrong" if wrong[j] else "timeout")
        rows.append({
            "idx": dataset_idx,
            "status": status,
            "rounds": rounds[j],
            "calls": calls[j],
            "forwards": ((rounds[j] + 1) if correct[j] else cfg.max_rounds) * cfg.n_chains,
            "n_resets": resets[j],
            **({"forwards_seq": forwards_seq[j]} if forwards_seq is not None else {}),
            "solution": solutions[j] if status != "timeout" else "",
            "verified": verified[j] if status != "timeout" else False,
            **(extra[j] if extra else {}),
        })
    return rows
