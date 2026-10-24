"""Compute accounting: analytic FLOPs per forward pass and measured B200 hours from results.

FLOPs count multiply-adds as 2 FLOPs and include the attention projections,
attention scores/values and the feed-forward block of every layer
application; embeddings, norms and output heads are omitted (under 2% at
these sizes). An LDT training step costs 4x the forward FLOPs per sample:
forward and backward through every loop (3x) and the no-grad forward that
advances the on-policy pool (1x); periodic monitoring solves change no
weights and are not counted. HRM/TRM/EqR train one ACT segment per sample
and step, backpropagating through the last recursion cycle only (see
`RecursiveArch.train_factor`). These are estimates for a compute table, not
measurements. Measured training hours are the post-compile training loop
(`post_compile_secs`), excluding the compile step and monitoring solves.

    uv run python -m paper.compute table1_sudoku table3_maze appendix_a_recurrence
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from paper.tables.common import OUT, RESULTS, by_entry, mean, read_results, write_output

Mixer = Literal["attention", "mlp"]


def layer_flops(tokens: int, dim: int, ffn_hidden: float) -> float:
    """FLOPs of one transformer layer over `tokens` positions."""
    proj = 4 * dim * dim * tokens                           # Q, K, V, O
    scores = 2 * tokens * tokens * dim                      # QK^T and AV
    ffn = 2 * dim * ffn_hidden * tokens                     # up + down
    return 2 * (proj + scores + ffn)


def ldt_forward_flops(dim: int, layers: int, loops: int, seq_len: int, ffn_mult: float = 4.0,
                      cls_token: bool = True) -> float:
    """One chain, one model call: `loops` passes through `layers` layers."""
    tokens = seq_len + int(cls_token)
    return loops * layers * layer_flops(tokens, dim, ffn_mult * dim)


def swiglu_width(size: int, expansion: float) -> int:
    """Intermediate width of the HRM/TRM SwiGLU: 2/3 * expansion * size, rounded up to 256."""
    return -(round(expansion * size * 2 / 3) // -256) * 256


@dataclass(frozen=True)
class RecursiveArch:
    """HRM/TRM-style model: per ACT step, `applications` layer applications."""

    name: str
    dim: int
    tokens: int                 # cells plus puzzle-embedding positions
    expansion: float
    applications: int
    act_steps: int              # ACT steps at evaluation (run to the maximum)
    mixer: Mixer = "attention"
    breadth: int = 1            # independent restarts at evaluation (EqR)
    # Forward-equivalents per sample and training step: one forward through all
    # cycles plus a backward (2x) through the last one; HRM also runs one more
    # no-grad forward for its halting target.
    train_factor: float = 1 + 2 / 3
    train_steps: int = 0        # steps to the evaluated checkpoint
    train_batch: int = 0

    def forward_flops(self) -> float:
        t, d = self.tokens, self.dim
        ffn = 3 * d * swiglu_width(d, self.expansion) * t          # SwiGLU: gate, up, down
        if self.mixer == "mlp":                                    # SwiGLU across positions
            mix = 3 * t * swiglu_width(t, self.expansion) * d
        else:
            mix = 4 * d * d * t + 2 * t * t * d
        return self.applications * 2 * (mix + ffn)

    def max_inference_flops(self) -> float:
        return self.act_steps * self.breadth * self.forward_flops()

    def training_flops(self) -> float:
        return self.train_factor * self.forward_flops() * self.train_batch * self.train_steps


# Upstream defaults (HRM arch/hrm_v1.yaml; TRM arch/trm.yaml plus the variant
# overrides in baselines/hrm_trm/trm_eval/cfg_*.yaml): hidden 512, expansion 4,
# 16 ACT steps. HRM puts one puzzle-embedding position in front of the cells,
# TRM 16 (puzzle_emb_len). EqR (config/arch/eqr.yaml, eqr_sudoku.yaml) is TRM's
# MLP variant, evaluated at depth 64 and breadth 128.
# Training: HRM's release trains 40,000 epochs at global batch 768 (52,083
# steps); the alphaXiv TRM Sudoku checkpoints stop at step 32,550 of a
# batch-1,536 run; EqR trains 50,000 steps at 768; the Maze rows are our
# retrains at 768, counted to the reported checkpoint.
# HRM: per ACT step, H_cycles x (L_cycles x L_layers + H_layers) = 2 x (2x4 + 4).
# TRM: per ACT step, H_cycles x (L_cycles + 1) x L_layers.
_HRM_APPLICATIONS = 2 * (2 * 4 + 4)
# One forward with gradient through the last L and H module calls (8 of 24
# layer applications), plus the no-grad forward for the halting target.
_HRM_TRAIN_FACTOR = 1 + 2 * 8 / 24 + 1


def _trm_applications(l_cycles: int) -> int:
    return 3 * (l_cycles + 1) * 2


BASELINES = {
    "hrm_sudoku": RecursiveArch("HRM", dim=512, tokens=81 + 1, expansion=4,
                                applications=_HRM_APPLICATIONS, act_steps=16,
                                train_factor=_HRM_TRAIN_FACTOR, train_steps=52_083,
                                train_batch=768),
    "trm_mlp_sudoku": RecursiveArch("TRM-MLP", dim=512, tokens=81 + 16, expansion=4,
                                    applications=_trm_applications(6), act_steps=16, mixer="mlp",
                                    train_steps=32_550, train_batch=1536),
    "trm_att_sudoku": RecursiveArch("TRM-attention", dim=512, tokens=81 + 16, expansion=4,
                                    applications=_trm_applications(6), act_steps=16,
                                    train_steps=32_550, train_batch=1536),
    "eqr_sudoku": RecursiveArch("EqR", dim=512, tokens=81 + 16, expansion=4,
                                applications=_trm_applications(6), act_steps=64, breadth=128,
                                mixer="mlp", train_steps=50_000, train_batch=768),
    "trm_maze": RecursiveArch("TRM (maze)", dim=512, tokens=900 + 16, expansion=4,
                              applications=_trm_applications(4), act_steps=16,
                              train_steps=19_530, train_batch=768),
    "hrm_maze": RecursiveArch("HRM (maze)", dim=512, tokens=900 + 1, expansion=4,
                              applications=_HRM_APPLICATIONS, act_steps=16,
                              train_factor=_HRM_TRAIN_FACTOR, train_steps=39_060,
                              train_batch=768),
}


def sotaku_flops(iterations: int = 1024, steps: int = 50_000, batch: int = 2048,
                 late_fraction: float = 0.2, late_horizons=(32, 64, 128, 256, 512)) -> dict:
    """Sotaku v2: 4 layers at d=128, feed-forward 512, over 81 cells.

    Training runs 16 iterations with gradients (3x each); on `late_fraction` of
    batches it first runs a no-grad burn-in to a random horizon and supervises
    only the window after it.
    """
    per_iteration = 4 * layer_flops(81, 128, 512)
    per_sample = 3 * 16 + late_fraction * sum(late_horizons) / len(late_horizons)
    return {"inference": iterations * per_iteration,
            "training": per_sample * per_iteration * batch * steps}


def run_compute(rec: dict) -> dict:
    """Per-run FLOPs and B200 hours from one collected results row."""
    fwd = ldt_forward_flops(rec["model_dim"], rec["model_layers"], rec["model_loops"],
                            rec["model_seq_len"], rec.get("model_ffn_mult") or 4.0,
                            bool(rec.get("model_cls")))
    train_s = rec.get("train_post_compile_seconds")
    eval_s = rec.get("gpu_seconds")
    out = {
        "forward_gflops": fwd / 1e9,
        "train_pflops": 4 * fwd * rec["train_batch"] * rec["train_steps"] / 1e15,
        # Training loop after the compile step, excluding monitoring solves; None
        # until the run has trained (or been evaluated), so means skip it.
        "train_b200_h": train_s / 3600 if train_s is not None else None,
        "eval_b200_h": ((eval_s + (rec.get("startup_seconds") or 0)) / 3600
                        if eval_s is not None else None),
    }
    if rec.get("calls_mean") is not None:
        # Each model call runs every chain of the puzzle once.
        out["inference_gflops_per_puzzle"] = (rec["calls_mean"] * rec["solve_chains"] * fwd / 1e9)
    return out


def _cell(x: float | None, digits: int = 2) -> str:
    return "--" if x is None or math.isnan(x) else f"{x:.{digits}f}"


def compute_table(manifests: list[str], results_dir: Path = RESULTS) -> str:
    """Markdown table of per-entry LDT compute plus the baseline architectures."""
    lines = ["| Manifest | Entry | Fwd GFLOPs | Train PFLOPs | Train B200 h | Eval B200 h | "
             "Inference GFLOPs / puzzle |", "|---|---|---:|---:|---:|---:|---:|"]
    for manifest in manifests:
        for entry, recs in by_entry(read_results(manifest, results_dir)).items():
            recs = [r for r in recs if r.get("model_dim") is not None]
            if not recs:
                continue
            per = [run_compute(r) for r in recs]
            avg = {k: mean([p.get(k) for p in per]) for k in per[0]}
            lines.append(
                f"| {manifest} | {entry} | {avg['forward_gflops']:.3f} | "
                f"{avg['train_pflops']:.2f} | {_cell(avg['train_b200_h'])} | "
                f"{_cell(avg['eval_b200_h'])} | "
                f"{_cell(avg.get('inference_gflops_per_puzzle'), 1)} |")
    lines += ["", "| Baseline | Fwd GFLOPs (one ACT step) | Inference GFLOPs / puzzle | "
              "Train PFLOPs |", "|---|---:|---:|---:|"]
    for arch in BASELINES.values():
        lines.append(f"| {arch.name} | {arch.forward_flops() / 1e9:.2f} | "
                     f"{arch.max_inference_flops() / 1e9:.1f} | "
                     f"{arch.training_flops() / 1e15:.0f} |")
    sotaku = sotaku_flops()
    lines.append(f"| Sotaku v2 | -- | {sotaku['inference'] / 1e9:.1f} | "
                 f"{sotaku['training'] / 1e15:.0f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(prog="paper.compute")
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    text = compute_table(args.manifests, args.results)
    print(text)
    write_output("compute.md", text, args.out)
    write_output("baseline_archs.json",
                 json.dumps({k: vars(v) for k, v in BASELINES.items()}, indent=2), args.out)


if __name__ == "__main__":
    main()
