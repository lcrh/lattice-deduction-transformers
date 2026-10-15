"""Compute accounting (appendix J): analytic FLOPs per forward pass, and
measured B200 hours from the collected results.

FLOPs count multiply-adds as 2 FLOPs and include the attention projections,
attention scores/values and the feed-forward block of every layer
application; embeddings, norms and output heads are omitted (under 2% at
these sizes). Training is costed as 3x the forward FLOPs (forward plus
backward) per sample per step. These are estimates for a compute table, not
measurements.

    uv run python -m paper.compute table1_sudoku table3_maze appendix_a_recurrence
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from paper.tables.common import OUT, RESULTS, by_entry, mean_std, read_results, write_output


def layer_flops(tokens: int, dim: int, ffn_hidden: float, mixer: str = "attention") -> float:
    """FLOPs of one transformer layer over `tokens` positions."""
    proj = 4 * dim * dim * tokens                           # Q, K, V, O
    scores = 2 * tokens * tokens * dim                      # QK^T and AV
    ffn = 2 * dim * ffn_hidden * tokens                     # up + down
    if mixer == "mlp":                                      # token-mixing MLP over positions
        return 2 * (2 * tokens * tokens * dim + ffn)
    return 2 * (proj + scores + ffn)


def ldt_forward_flops(dim: int, layers: int, loops: int, seq_len: int, ffn_mult: float = 4.0,
                      cls_token: bool = True) -> float:
    """One chain, one model call: `loops` passes through `layers` layers."""
    tokens = seq_len + int(cls_token)
    return loops * layers * layer_flops(tokens, dim, ffn_mult * dim)


@dataclass(frozen=True)
class RecursiveArch:
    """HRM/TRM-style model: per ACT step, `applications` layer applications."""
    name: str
    dim: int
    tokens: int
    expansion: float            # SwiGLU: 3 matrices of dim x (2/3 * expansion * dim)
    applications: int
    act_steps: int              # maximum ACT steps (halting can stop earlier)
    mixer: str = "attention"

    def forward_flops(self) -> float:
        ffn_hidden = 1.5 * (2 / 3) * self.expansion * self.dim     # SwiGLU's 3 matrices
        return self.applications * layer_flops(self.tokens, self.dim, ffn_hidden, self.mixer)

    def max_inference_flops(self) -> float:
        return self.act_steps * self.forward_flops()


# Upstream defaults (HRM arch/hrm_v1.yaml; TRM arch/trm.yaml plus the variant
# overrides in baselines/hrm_trm/trm_eval/cfg_*.yaml): hidden 512, expansion 4,
# 16 ACT steps, one puzzle-embedding token in front of the 81 cells.
# HRM: per ACT step, H_cycles x (L_cycles x L_layers + H_layers) = 2 x (2x4 + 4).
# TRM: per ACT step, H_cycles x (L_cycles + 1) x L_layers.
BASELINES = {
    "hrm_sudoku": RecursiveArch("HRM", 512, 82, 4, 2 * (2 * 4 + 4), 16),
    "trm_mlp_sudoku": RecursiveArch("TRM-MLP", 512, 82, 4, 3 * (6 + 1) * 2, 16, mixer="mlp"),
    "trm_att_sudoku": RecursiveArch("TRM-attention", 512, 82, 4, 3 * (6 + 1) * 2, 16),
    "trm_maze": RecursiveArch("TRM (maze)", 512, 901, 4, 3 * (4 + 1) * 2, 16),
    "hrm_maze": RecursiveArch("HRM (maze)", 512, 901, 4, 2 * (2 * 4 + 4), 16),
}


def run_compute(rec: dict) -> dict:
    """Per-run FLOPs and B200 hours from one collected results row."""
    fwd = ldt_forward_flops(rec["model_dim"], rec["model_layers"], rec["model_loops"],
                            rec["model_seq_len"], rec.get("model_ffn_mult") or 4.0,
                            bool(rec.get("model_cls")))
    out = {
        "forward_gflops": fwd / 1e9,
        "train_pflops": 3 * fwd * rec["train_batch"] * rec["train_steps"] / 1e15,
        "train_b200_h": (rec.get("train_seconds") or 0) / 3600,
        "eval_b200_h": ((rec.get("gpu_seconds") or 0) + (rec.get("startup_seconds") or 0)) / 3600,
    }
    if rec.get("calls_mean") is not None:
        # Each model call runs every chain of the puzzle once.
        out["inference_gflops_per_puzzle"] = (rec["calls_mean"] * rec["solve_chains"] * fwd / 1e9)
    return out


def table(manifests: list[str]) -> str:
    lines = ["| Manifest | Entry | Fwd GFLOPs | Train PFLOPs | Train B200 h | Eval B200 h | "
             "Inference GFLOPs / puzzle |", "|---|---|---:|---:|---:|---:|---:|"]
    for manifest in manifests:
        for entry, recs in by_entry(read_results(manifest, RESULTS)).items():
            recs = [r for r in recs if r.get("model_dim") is not None]
            if not recs:
                continue
            per = [run_compute(r) for r in recs]
            avg = {k: mean_std([p.get(k) for p in per])[0] for k in per[0]}
            lines.append(
                f"| {manifest} | {entry} | {avg['forward_gflops']:.3f} | "
                f"{avg['train_pflops']:.2f} | {avg['train_b200_h']:.2f} | "
                f"{avg['eval_b200_h']:.2f} | {avg.get('inference_gflops_per_puzzle', 0):.1f} |")
    lines += ["", "| Baseline | Fwd GFLOPs (one ACT step) | Max inference GFLOPs / puzzle |",
              "|---|---:|---:|"]
    for arch in BASELINES.values():
        lines.append(f"| {arch.name} | {arch.forward_flops() / 1e9:.2f} | "
                     f"{arch.max_inference_flops() / 1e9:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(prog="paper.compute")
    ap.add_argument("manifests", nargs="+")
    args = ap.parse_args()
    text = table(args.manifests)
    print(text)
    write_output("compute.md", text, OUT)
    write_output("compute.json", json.dumps({k: vars(v) for k, v in BASELINES.items()}, indent=2),
                 OUT)


if __name__ == "__main__":
    main()
