"""Analytic training-step FLOPs for LDT / TRM / HRM / Sotaku.

Hardware-independent compute column for the E8 cost table. Everything here is a
paper-and-pencil transformer FLOP count from architecture constants (dim,
seq_len, layers, heads, ffn_mult) times the number of times the backbone is
run per training step (loops / cycles / segments) times fwd+bwd factor.

WHAT WE COUNT (and what we don't)
---------------------------------
We count matmul FLOPs only: the linear projections in attention (Q, K, V, out)
and the two FFN matmuls, plus the attention score+context matmuls
(QK^T and softmax@V). This is the standard "6ND" style accounting used for MFU:
elementwise ops (LayerNorm, GELU, softmax normalization, residual adds, dropout)
and the embedding lookups are <1% of matmul FLOPs at these shapes and are
omitted, exactly as in Kaplan et al. 2020 / the PaLM MFU convention.

Conventions:
  * A matmul of [m,k] x [k,n] costs 2*m*k*n FLOPs (the factor 2 = one multiply +
    one add per MAC).
  * Backward pass costs ~2x the forward matmul FLOPs (one matmul for the input
    grad, one for the weight grad). So fwd+bwd = 3x forward. This is the
    universal convention (Kaplan 2020 App. B; "6 FLOPs per param per token").
  * Deep supervision: if the loss is applied at every loop/cycle output and all
    of them backprop (LDT default supervise="all"), then ALL loop iterations are
    on the backward graph -> the 3x applies to every loop. If only the final
    iteration is supervised and earlier ones run under no_grad (TRM/HRM's
    1-step-gradient trick), only the graph-tracked segment gets the 3x and the
    rest is forward-only (1x). We model both via `grad_loops` vs `nograd_loops`.

PER-TOKEN FORWARD MATMUL FLOPs for one transformer layer (batch factored out):
    attention projections (Q,K,V,out): 4 * (2 * dim * dim)          = 8 * dim^2
    attention scores QK^T + context@V: 2 * (2 * seq_len * dim)      = 4 * seq_len * dim
        (per token: QK^T is [seq_len,dim]x[dim,seq_len]->contributes 2*seq_len*dim
         per query token; context @V likewise 2*seq_len*dim per token)
    FFN (two matmuls, hidden = ffn_mult*dim): 2 * (2 * dim * hidden) = 4 * ffn_mult * dim^2
  => per_token_layer_fwd = 8*dim^2 + 4*ffn_mult*dim^2 + 4*seq_len*dim

Multiply by seq_len (tokens) * num_layers (layers) * backbone_runs (loops) *
fwd_bwd_factor * batch * steps to get total. See `model_step_flops`.

CROSS-CHECK against the "6ND" rule of thumb (6 FLOPs per param per token,
fwd+bwd): for LDT the backbone (795,648 params) is re-run 16x/step, so
tokens-applied = batch*seq_len*loops = 512*82*16, giving 6*N*D = 3.21e12
FLOPs/step, vs this formula's 3.51e12 — within 9%. The gap is exactly the
attention score/context matmuls (which 6ND ignores) minus the embedding params
(which 6ND includes but which aren't matmuls here). Independent confirmation
the formula is right.

MFU SANITY CHECK
----------------
`mfu_check(...)` takes a measured wall-clock s/step for LDT and returns achieved
MFU = (per-step FLOPs) / (s_per_step * B200_PEAK_FLOPS). A correct dense
transformer formula on a well-utilized GPU lands ~20-60% MFU. LDT is TINY
(dim=128, ~800K params, seq_len 82) so its per-step FLOPs are minuscule and it
will be heavily KERNEL-LAUNCH / MEMORY-BANDWIDTH bound, NOT matmul bound -> we
expect its measured MFU to be FAR below 20% (typically well under 1%), which is the
correct, expected physics for a tiny model, not a formula bug. We document that
loudly: the 20-60% band is the diagnostic for a LARGE compute-bound model; for
LDT the check confirms the formula is not ABOVE peak (which would be the real
bug) and reports the (expectedly low) achieved MFU. No committed LDT wall-clock
exists in the repo yet, so `mfu_check` runs only when a wall-clock arg is
supplied; run it once real numbers land (see main()).

Independent estimates; not from the papers' own FLOP tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Hardware constant.
# --------------------------------------------------------------------------
# NVIDIA B200 dense BF16 matmul peak, spec-sheet number (no sparsity):
# ~2.25 PFLOP/s dense. (NVIDIA lists ~4.5 PFLOP/s with 2:4 structured sparsity;
# dense = half of that.) This is the theoretical peak, not an achievable rate.
B200_PEAK_FLOPS = 2.25e15  # dense BF16 FLOP/s, spec sheet peak


# --------------------------------------------------------------------------
# Architecture descriptor. All FLOPs derive from these.
# --------------------------------------------------------------------------
@dataclass
class Arch:
    name: str
    task: str
    dim: int
    seq_len: int
    num_layers: int          # transformer layers inside ONE backbone pass
    ffn_mult: float
    # How many times the backbone (num_layers) is run per training step.
    # For deep-supervised looped models where every loop is on the backward
    # graph, all of these are grad_loops. For 1-step-gradient models (TRM/HRM),
    # the recurrent unroll runs under no_grad and only the final segment is
    # graph-tracked: split into grad_loops (get 3x fwd+bwd) and nograd_loops
    # (1x forward only).
    grad_loops: int          # backbone passes that are on the backward graph
    nograd_loops: int        # backbone passes run forward-only (no_grad)
    batch: int
    steps: int
    params: int              # total trainable params (for reference / 6ND cross-check)
    source: str              # where the constants come from

    def per_token_layer_fwd_flops(self) -> float:
        """Forward matmul FLOPs for ONE layer, per token (batch=1, one token)."""
        d = self.dim
        hidden = self.ffn_mult * d
        attn_proj = 8.0 * d * d                 # Q,K,V,out : 4 * 2*d*d
        attn_score = 4.0 * self.seq_len * d     # QK^T + context@V : 2 * 2*L*d
        ffn = 4.0 * hidden * d                   # two matmuls : 2 * 2*d*hidden
        return attn_proj + attn_score + ffn

    def backbone_fwd_flops(self) -> float:
        """Forward matmul FLOPs for ONE full backbone pass (all layers, all
        tokens, batch=1)."""
        return (self.per_token_layer_fwd_flops()
                * self.seq_len * self.num_layers)

    def step_flops(self) -> float:
        """FLOPs for ONE training step (whole batch), fwd+bwd, all backbone
        passes."""
        fwd = self.backbone_fwd_flops()
        grad_flops = self.grad_loops * fwd * 3.0     # fwd+bwd = 3x
        nograd_flops = self.nograd_loops * fwd * 1.0  # forward only
        return (grad_flops + nograd_flops) * self.batch

    def total_pflops(self) -> float:
        """Total PFLOPs (1e15 FLOPs) over the whole training run."""
        return self.step_flops() * self.steps / 1e15

    def step_pflops(self) -> float:
        return self.step_flops() / 1e15


# --------------------------------------------------------------------------
# The four architectures, with every constant cited.
# --------------------------------------------------------------------------
def ldt_arch() -> Arch:
    """LDT (this repo's PowersetModel).

    Constants from src/lattice_diffusion/models/looped_transformer.py
    (LoopedTransformerConfig defaults) and experiments/sudoku/run.py defaults:
      dim=128, num_layers=4, n_heads=4, ffn_mult=4.0, n_loops=16, seq_len=81.
    Default run has conflict_loss_weight=0.1 > 0 -> cls_token=True -> the
    backbone sequence is seq_len+1 = 82 (run.py: cls_token=conflict_loss_weight>0).
    Deep supervision: train.py TrainConfig.supervise defaults to "all", and the
    forward is called with return_all=True, so EVERY one of the 16 loops feeds
    the loss and is on the backward graph -> grad_loops=16, nograd_loops=0.
    Batch 512, steps 4000 (run.py defaults).
    Total params measured by building the model: 799,635.
    """
    return Arch(
        name="LDT",
        task="sudoku-extreme",
        dim=128,
        seq_len=82,              # 81 grid cells + 1 CLS (cls_token=True by default)
        num_layers=4,
        ffn_mult=4.0,
        grad_loops=16,           # supervise="all": all 16 loops backprop
        nograd_loops=0,
        batch=512,
        steps=4000,
        params=799_635,
        source="repo: looped_transformer.py + sudoku/run.py defaults (measured params)",
    )


def trm_maze_arch() -> Arch:
    """TRM (Tiny Recursive Model) on maze-30x30-hard.

    Constants from repro/trm_eval/modal_train.py TRAIN_ARGS:
      arch.L_layers=2, arch.H_cycles=3, arch.L_cycles=4, global_batch_size=768,
      epochs=50000 (-> ~65k steps; repro README: steps = epochs * 1.302).
    TRM's released maze config (alphaXiv TinyRecursiveModels, arch=trm) uses
    hidden_size=512, num_heads=8, expansion=4 (ffn_mult=4), seq_len=900
    (30x30 grid) + a small number of puzzle/answer embedding tokens; we use 900.
    Recurrence: H_cycles * L_cycles = 3*4 = 12 inner recurrent steps, each
    running the L_layers=2 block. TRM uses the 1-step-gradient trick: only the
    LAST recurrent step is graph-tracked (grad), the earlier 11 run under
    no_grad. So one "backbone pass" here = the 2-layer block; grad passes = 1
    recurrent step's worth = 1 block; nograd passes = 11 blocks. (This is the
    dominant term; the tiny puzzle-embedding matmul is ignored.)
    params ~7M (repro README).
    steps: 50000 epochs * 1.302 ~= 65100 (repro README formula).

    Convention (matches hrm_maze_arch): we fold the layer count into
    grad/nograd loops and set num_layers=1, so `backbone_fwd_flops` counts one
    layer and the loop fields carry the total LAYER-passes. grad = 1 block ×
    2 layers = 2 grad layer-passes (×3 fwd+bwd); nograd = 11 blocks × 2 layers
    = 22 layer-passes (×1). Total weighted layer-passes = 2·3 + 22 = 28.
    (Setting num_layers=2 AND layer-count loops would double-count the layers.)
    """
    L_layers = 2
    H_cycles, L_cycles = 3, 4
    total_recur = H_cycles * L_cycles  # 12
    return Arch(
        name="TRM",
        task="maze-30x30-hard",
        dim=512,
        seq_len=900,             # 30x30 grid
        num_layers=1,            # layer count folded into grad/nograd loops
        ffn_mult=4.0,
        # 1-step gradient: last recurrent step tracked, rest no_grad. Loops are
        # LAYER-passes (num_layers=1 above): 1 grad block = 2 layers; 11 nograd
        # blocks = 22 layers.
        grad_loops=L_layers,                       # final block = 2 grad layer-passes
        nograd_loops=L_layers * (total_recur - 1),  # 11 blocks = 22 layer-passes
        batch=768,
        steps=65_100,            # 50000 epochs * 1.302 (repro README)
        params=7_000_000,
        source="repro/trm_eval/modal_train.py TRAIN_ARGS + TRM arch=trm config",
    )


def hrm_maze_arch() -> Arch:
    """HRM (Hierarchical Reasoning Model) on maze-30x30-hard.

    Constants from repro/hrm_eval/modal_train.py TRAIN_ARGS + HRM maze config
    (sapientinc/HRM, cfg_pretrain maze): hidden_size=512, num_heads=8,
    H_layers=4, L_layers=4, H_cycles=2, L_cycles=2, expansion=4, seq_len=900.
    global_batch_size=768 (repro). 27M params (repro README).
    HRM runs H_cycles * L_cycles = 2*2 = 4 hierarchical steps; each step runs
    the L-module (L_layers=4) and, at H-boundaries, the H-module (H_layers=4).
    Per full recurrence: 4 L-passes (4 layers each) + 2 H-passes (4 layers each).
    HRM also uses the 1-step-gradient trick (only the final segment tracked).
    We approximate one "backbone pass" as the 4-layer module and count:
      total module-passes per step = L_cycles*H_cycles (L) + H_cycles (H)
                                    = 4 + 2 = 6 module-passes of 4 layers each.
      grad = final 1 module-pass (4 layers); nograd = remaining 5 (20 layers).
    steps: HRM ran ~150k steps in the repro 24h budget, but the RECIPE point is
    20000 epochs. repro README: total steps = epochs * 1.302. We report the
    recipe-point step count (20000 * 1.302 ~= 26040) as the comparable training
    cost; the overtrain-to-150k run is a separate trajectory, not the recipe.
    """
    L_layers = 4
    H_layers = 4
    H_cycles, L_cycles = 2, 2
    l_passes = H_cycles * L_cycles        # 4 L-module passes
    h_passes = H_cycles                   # 2 H-module passes
    # layers run in the no_grad segment vs the final grad segment. Treat the
    # final module pass (4 layers) as grad-tracked, the rest forward-only.
    total_layer_passes = l_passes * L_layers + h_passes * H_layers  # 4*4 + 2*4 = 24
    grad_layers = L_layers                                          # final 4-layer pass
    nograd_layers = total_layer_passes - grad_layers               # 20
    return Arch(
        name="HRM",
        task="maze-30x30-hard",
        dim=512,
        seq_len=900,
        num_layers=1,            # we fold the layer count into grad/nograd loops below
        ffn_mult=4.0,
        grad_loops=grad_layers,       # 4 layers on grad graph (x3)
        nograd_loops=nograd_layers,   # 20 layers forward-only
        batch=768,
        steps=26_040,            # recipe point: 20000 epochs * 1.302 (repro README)
        params=27_000_000,
        source="repro/hrm_eval/modal_train.py TRAIN_ARGS + HRM maze cfg_pretrain",
    )


def sotaku_arch() -> Arch:
    """Sotaku on Sudoku.

    NOTE: Sotaku's architecture constants are NOT in this repo. The LDT paper
    cites Sotaku as a comparison point but we have no config file to read here.
    We model it as a same-family looped transformer at the numbers reported in
    the LDT paper's comparison (placeholder: dim/layers/loops TBD from the
    Sotaku paper). Marked TBD so nobody mistakes it for a grounded estimate.
    Returns an Arch with params/loops flagged; total_pflops is therefore a
    ROUGH order-of-magnitude figure, source-labeled "reported/TBD" downstream.
    """
    return Arch(
        name="Sotaku",
        task="sudoku",
        dim=128,                 # TBD — placeholder, same family as LDT
        seq_len=81,
        num_layers=4,            # TBD
        ffn_mult=4.0,
        grad_loops=16,           # TBD
        nograd_loops=0,
        batch=512,
        steps=4000,              # TBD
        params=0,                # unknown
        source="TBD — Sotaku config not in repo; placeholder (do not cite as measured)",
    )


ALL_ARCHES = [ldt_arch, trm_maze_arch, hrm_maze_arch, sotaku_arch]


def all_estimates() -> list[dict]:
    """Per model x task: step_pflops, total_pflops, params, source."""
    rows = []
    for factory in ALL_ARCHES:
        a = factory()
        rows.append({
            "model": a.name,
            "task": a.task,
            "params": a.params,
            "grad_loops": a.grad_loops,
            "nograd_loops": a.nograd_loops,
            "batch": a.batch,
            "steps": a.steps,
            "step_pflops": a.step_pflops(),
            "total_pflops": a.total_pflops(),
            "source": a.source,
            "is_tbd": a.name == "Sotaku",
        })
    return rows


def mfu_check(s_per_step: float, arch: Arch | None = None,
              peak_flops: float = B200_PEAK_FLOPS) -> dict:
    """Achieved MFU for a measured wall-clock s/step.

    MFU = per-step matmul FLOPs / (s_per_step * peak_flops).
    Returns the raw MFU plus a verdict string. See module docstring: for a TINY
    model like LDT the physically-correct MFU is well under 1% (kernel-launch /
    bandwidth bound), so the 20-60% "healthy compute-bound" band is NOT the pass
    condition for LDT — the REAL failure mode to catch is MFU > 100% (formula
    over-counts / faster than the GPU can physically go).
    """
    if arch is None:
        arch = ldt_arch()
    flops = arch.step_flops()
    achieved = flops / (s_per_step * peak_flops)
    if achieved > 1.0:
        verdict = ("IMPOSSIBLE (>100% of peak) — the FLOPs formula is WRONG "
                   "(over-counting) or the wall-clock is mis-measured. FIX.")
    elif achieved >= 0.20:
        verdict = "healthy compute-bound MFU (20-60% band)"
    elif achieved >= 0.01:
        verdict = ("low MFU (<20%) — EXPECTED for a tiny model (dim=128, "
                   "~0.8M params): kernel-launch / bandwidth bound, not a "
                   "formula bug. Formula is plausible (not above peak).")
    else:
        verdict = ("very low MFU (<1%) — plausible for a tiny model but "
                   "double-check the wall-clock is steady-state (exclude compile).")
    return {
        "s_per_step": s_per_step,
        "step_flops": flops,
        "step_pflops": flops / 1e15,
        "peak_flops": peak_flops,
        "achieved_mfu": achieved,
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analytic training FLOPs per model x task.")
    ap.add_argument("--ldt-s-per-step", type=float, default=None,
                    help="Measured LDT steady-state s/step (post-compile) to run "
                         "the MFU sanity check. Omit if no wall-clock yet.")
    ap.add_argument("--peak-flops", type=float, default=B200_PEAK_FLOPS,
                    help="Override B200 dense-BF16 peak FLOP/s.")
    args = ap.parse_args()

    print("=" * 92)
    print("Analytic training-step FLOPs  (matmul-only; fwd+bwd; hardware-independent)")
    print(f"B200 dense BF16 peak (spec sheet): {args.peak_flops:.3e} FLOP/s")
    print("=" * 92)
    hdr = (f"{'model':<8} {'task':<18} {'params':>11} {'grad/ng':>8} "
           f"{'batch':>6} {'steps':>7} {'PFLOP/step':>12} {'total PFLOP':>13}  source")
    print(hdr)
    print("-" * len(hdr))
    for r in all_estimates():
        tbd = "  [TBD]" if r["is_tbd"] else ""
        print(f"{r['model']:<8} {r['task']:<18} {r['params']:>11,} "
              f"{r['grad_loops']:>3}/{r['nograd_loops']:<4} "
              f"{r['batch']:>6} {r['steps']:>7} "
              f"{r['step_pflops']:>12.4e} {r['total_pflops']:>13.2f}  {r['source']}{tbd}")
    print("-" * len(hdr))
    print("Notes: grad = backbone passes on the backward graph (x3 fwd+bwd); "
          "ng = no_grad forward-only passes (x1).")
    print("       LDT supervises all 16 loops (deep supervision); TRM/HRM use "
          "the 1-step-gradient trick (most passes no_grad).")
    print("       Sotaku row is a PLACEHOLDER (config not in repo) — do not cite "
          "as a grounded estimate.")

    print("\n" + "=" * 92)
    print("MFU sanity check (LDT)")
    print("=" * 92)
    if args.ldt_s_per_step is not None:
        res = mfu_check(args.ldt_s_per_step, ldt_arch(), args.peak_flops)
        print(f"  measured s/step         : {res['s_per_step']:.4f}")
        print(f"  per-step FLOPs          : {res['step_flops']:.4e}")
        print(f"  per-step PFLOPs         : {res['step_pflops']:.4e}")
        print(f"  B200 peak FLOP/s        : {res['peak_flops']:.3e}")
        print(f"  achieved MFU            : {res['achieved_mfu']:.4%}")
        print(f"  verdict                 : {res['verdict']}")
    else:
        print("  No --ldt-s-per-step supplied and no committed LDT wall-clock in")
        print("  the repo. The check is a function (mfu_check) that MUST be run")
        print("  once a measured LDT steady-state s/step lands, e.g.:")
        print("      python followups/cost_accounting/flops.py --ldt-s-per-step 0.35")
        print("  Expected: sub-1% MFU (tiny model, kernel/bandwidth bound);")
        print("  the FAILURE to catch is MFU > 100% (formula over-counts).")


if __name__ == "__main__":
    main()
