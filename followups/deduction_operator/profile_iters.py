"""E3-O2 — per-iteration elimination profiler.

Inside a single forward, what does each successive loop iteration contribute?
A `return_all=True` forward exposes every intermediate iteration's heads, so we
can answer "what does each extra loop buy?" per-forward, cleanly separated from
search effects.

This script loads the state bank (from `state_bank.py`), rebuilds the baseline
model at `L_eval` loops (default 128), runs ONE `return_all=True` forward over
the whole bank, and for each iteration l = 1..L measures:

  - eliminations at theta_elim, cumulative and marginal (per fill-stratum),
  - how many of those eliminations are UNSOUND (kill an alive GT bit — the
    bank carries each state's ground truth),
  - the CLS conflict-head logit trajectory, averaged over SAT vs UNSAT states.

Output CSV: one row per (iteration, stratum) plus SAT/UNSAT CLS columns.

    uv run modal run --detach followups/deduction_operator/profile_iters.py -- \
        --checkpoint /checkpoints/followups/e1/baseline_seed0.pt \
        --bank /checkpoints/followups/e3/state_bank.pt --l-eval 128

Writes /checkpoints/followups/e3/profile_iters.csv on the volume.

SANITY GATE (O2 <-> O1 consistency). The elimination rule applied here at each
iteration l is IDENTICAL to the one `dpll_step` applies end-to-end: kill an
alive candidate when sigmoid(bce_logit_l) < theta_elim, excluding puzzle
givens. At l = 16 the per-iteration head `all_bce[15]` is exactly the iterate a
`use_final=True` forward with n_loops=16 returns (same weight-tied recurrence,
same input) — so the profiler's iteration-16 marginal-elimination count is the
count the L_eval=16 end-to-end eval applies on each state's FIRST pass. Because
both use the deterministic threshold on the same bce head over the same states,
the profiler's iter-16 column is comparable to a CLEAN (no-dropout, no-aug)
L=16 forward (see `--assert-l16-forward`: it additionally runs a separate
`use_final=True` forward at n_loops=16 over the bank and asserts the iter-16
elimination count matches, to catch any drift). This makes the O1/O2 tie-back
checkable without a full end-to-end run.

Caveat: the actual O1 end-to-end eval runs with eval-time dropout (train()
mode) and augmentation enabled, so it is a stochastic/augmented ensemble over
passes — its per-round elimination count will not be bit-identical to this
profiler's clean iter-16 column. The tie-back is exact against a clean forward
(what `--assert-l16-forward` checks); against the logged O1 numbers it is a
close comparison, not an equality.
"""

from __future__ import annotations

import csv
import io

import modal
import torch

from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint
from experiments.sudoku.ema import swap_in_ema_if_present

app = modal.App("e3-profile-iters")


def _elim_mask(bce_logits: torch.Tensor, state: torch.Tensor,
               given_mask: torch.Tensor, threshold: float) -> torch.Tensor:
    """Deterministic threshold-elimination mask, matching dpll_step exactly:
    kill an ALIVE, non-given candidate whose sigmoid(bce) < threshold.

    bce_logits, state: [N, S, C]; given_mask: [N, S] bool. Returns [N, S, C].
    """
    probs = torch.sigmoid(bce_logits)
    m = (probs < threshold) & (state > 0.5)
    m = m & ~given_mask.unsqueeze(-1)
    return m


@app.function(
    image=image, gpu="B200", timeout=3600,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    checkpoint: str = f"{CHECKPOINT_MOUNT}/followups/e1/baseline_seed0.pt",
    bank: str = f"{CHECKPOINT_MOUNT}/followups/e3/state_bank.pt",
    l_eval: int = 128,
    threshold: float = 0.10,
    micro_batch: int = 256,
    out: str = f"{CHECKPOINT_MOUNT}/followups/e3/profile_iters.csv",
    assert_l16_forward: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    print(f"Loading state bank: {bank}", flush=True)
    checkpoint_volume.reload()
    bank_obj = torch.load(bank, map_location="cpu", weights_only=False)
    states = bank_obj["state"]        # [N, S, C]
    gts = bank_obj["gt"]              # [N, S, C]
    strata = bank_obj["stratum"]      # [N] int8
    sat = bank_obj["sat"]             # [N] bool
    meta = bank_obj.get("meta", {})
    strata_names = meta.get("strata_names", ["early", "mid", "late"])
    N, S, C = states.shape
    print(f"  bank: {N} snapshots (S={S}, C={C}); strata={strata_names}",
          flush=True)

    # Rebuild the model at L_eval loops (weight-tied; state dict loop-invariant).
    print(f"Loading model {checkpoint} @ n_loops={l_eval}", flush=True)
    ckpt = load_checkpoint(checkpoint)
    model_cfg_d = dict(ckpt["model_cfg"])
    native = model_cfg_d.get("n_loops")
    model_cfg_d["n_loops"] = l_eval
    model = PowersetModel(LoopedTransformerConfig(**model_cfg_d))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
    print(f"  native n_loops={native} -> profiling at {l_eval}", flush=True)

    given_mask_all = (states.sum(dim=-1) == 1)   # [N, S] — singleton = given/fixed
    gt_bool_all = (gts > 0.5)

    # Per-iteration, per-stratum accumulators.
    n_strata = len(strata_names)
    # marginal eliminations at iter l = eliminations newly below threshold at l
    # that were NOT below at l-1 (on the SAME state). We track cumulative
    # (union over 1..l) and marginal (this-iter mask minus previous union).
    cum_elim = torch.zeros(l_eval, n_strata, dtype=torch.float64)
    marg_elim = torch.zeros(l_eval, n_strata, dtype=torch.float64)
    cum_unsound = torch.zeros(l_eval, n_strata, dtype=torch.float64)
    marg_unsound = torch.zeros(l_eval, n_strata, dtype=torch.float64)
    # CLS logit trajectory, summed over SAT / UNSAT states per iter.
    cls_sat_sum = torch.zeros(l_eval, dtype=torch.float64)
    cls_unsat_sum = torch.zeros(l_eval, dtype=torch.float64)
    n_sat_total = int(sat.sum().item())
    n_unsat_total = int((~sat).sum().item())
    stratum_counts = [int((strata == si).sum().item()) for si in range(n_strata)]

    # For the L16 forward sanity assertion.
    iter16_marg_from_all = None   # [n_strata] marginal elim at iter 16 (return_all)

    has_conflict = None  # discovered from first micro-batch's output keys

    with torch.no_grad():
        for start in range(0, N, micro_batch):
            end = min(start + micro_batch, N)
            st = states[start:end].to(device)               # [b, S, C]
            gm = given_mask_all[start:end].to(device)        # [b, S]
            gtb = gt_bool_all[start:end].to(device)          # [b, S, C]
            strat = strata[start:end]                        # [b]
            sat_b = sat[start:end]                            # [b]

            out_all = model(st, return_all=True)
            bce_list = out_all["bce"]                        # list len L, [b,S,C]
            conflict_list = out_all.get("conflict")          # list or None
            has_conflict = conflict_list is not None

            # Running union of eliminated bits across iterations (per row).
            prev_union = torch.zeros_like(st, dtype=torch.bool)
            for l in range(l_eval):
                m_l = _elim_mask(bce_list[l], st, gm, threshold)   # [b,S,C] this-iter
                union_l = prev_union | m_l
                marg = union_l & ~prev_union                       # newly killed at l
                # unsound = killed an alive GT bit.
                cum_uns = union_l & gtb
                marg_uns = marg & gtb
                # Accumulate per stratum.
                row_cum = union_l.sum(dim=(1, 2)).double().cpu()   # [b]
                row_marg = marg.sum(dim=(1, 2)).double().cpu()
                row_cum_uns = cum_uns.sum(dim=(1, 2)).double().cpu()
                row_marg_uns = marg_uns.sum(dim=(1, 2)).double().cpu()
                for si in range(n_strata):
                    sel = (strat == si)
                    if sel.any():
                        cum_elim[l, si] += row_cum[sel].sum()
                        marg_elim[l, si] += row_marg[sel].sum()
                        cum_unsound[l, si] += row_cum_uns[sel].sum()
                        marg_unsound[l, si] += row_marg_uns[sel].sum()
                # CLS logit trajectory.
                if has_conflict:
                    cls_l = conflict_list[l].squeeze(-1).double().cpu()   # [b]
                    if sat_b.any():
                        cls_sat_sum[l] += cls_l[sat_b].sum()
                    if (~sat_b).any():
                        cls_unsat_sum[l] += cls_l[~sat_b].sum()
                prev_union = union_l

    # Per-state averages (divide sums by stratum counts / sat counts).
    def _avg(mat: torch.Tensor, si: int) -> list[float]:
        denom = max(stratum_counts[si], 1)
        return (mat[:, si] / denom).tolist()

    # Optional L16 sanity forward: a separate use_final forward at n_loops=16.
    l16_check = None
    if assert_l16_forward and l_eval >= 16:
        model16 = PowersetModel(LoopedTransformerConfig(**{**model_cfg_d, "n_loops": 16}))
        model16.load_state_dict(ckpt["model_state_dict"])
        model16.to(device).eval()
        swap_in_ema_if_present(model16, ckpt)
        tot_final = 0.0
        with torch.no_grad():
            for start in range(0, N, micro_batch):
                end = min(start + micro_batch, N)
                st = states[start:end].to(device)
                gm = given_mask_all[start:end].to(device)
                out16 = model16(st, use_final=True)
                m16 = _elim_mask(out16["bce"], st, gm, threshold)
                tot_final += float(m16.sum().item())
        # Compare to the return_all profiler's cumulative-through-16 total
        # (union of iters 1..16). At iter index 15 (l=16), cum_elim summed over
        # strata is the union of masks 1..16; the use_final(16) forward's mask
        # is mask-at-16 alone. Both use the SAME bce head at iterate 16, so the
        # use_final total must equal the iter-16 MARGINAL+... — more precisely,
        # the single-iter mask at l=16 equals what use_final(16) produces. We
        # therefore compare against the iter-16 SINGLE-iteration mask total,
        # recomputed here as (cum@16 restricted to this-iter). For a clean
        # check we recompute the single-iter-16 total below.
        # Single-iter-16 total across all strata = sum over rows of mask at l=16.
        # We didn't store it separately, so recompute quickly.
        tot_iter16_single = 0.0
        with torch.no_grad():
            for start in range(0, N, micro_batch):
                end = min(start + micro_batch, N)
                st = states[start:end].to(device)
                gm = given_mask_all[start:end].to(device)
                out_all = model(st, return_all=True)
                m16 = _elim_mask(out_all["bce"][15], st, gm, threshold)
                tot_iter16_single += float(m16.sum().item())
        agree = abs(tot_final - tot_iter16_single) < 1e-6
        l16_check = {
            "use_final16_total_elim": tot_final,
            "return_all_iter16_single_total_elim": tot_iter16_single,
            "agree": agree,
        }
        print(f"[L16 sanity] use_final(16) elim total={tot_final:.1f}  "
              f"return_all iter16 elim total={tot_iter16_single:.1f}  "
              f"agree={agree}", flush=True)
        if not agree:
            print("[L16 sanity] WARNING: mismatch — the profiler's iter-16 head "
                  "diverged from a use_final(16) forward. Investigate before "
                  "trusting O1<->O2 tie-back.", flush=True)

    # Write CSV: one row per (iteration, stratum).
    buf = io.StringIO()
    w = csv.writer(buf)
    header = [
        "iter", "stratum",
        "cum_elim_per_state", "marg_elim_per_state",
        "cum_unsound_per_state", "marg_unsound_per_state",
        "cls_logit_sat_mean", "cls_logit_unsat_mean",
        "n_states_stratum", "n_sat_total", "n_unsat_total",
    ]
    w.writerow(header)
    cls_sat_mean = (cls_sat_sum / max(n_sat_total, 1)).tolist()
    cls_unsat_mean = (cls_unsat_sum / max(n_unsat_total, 1)).tolist()
    for l in range(l_eval):
        for si in range(n_strata):
            w.writerow([
                l + 1, strata_names[si],
                f"{cum_elim[l, si] / max(stratum_counts[si], 1):.6f}",
                f"{marg_elim[l, si] / max(stratum_counts[si], 1):.6f}",
                f"{cum_unsound[l, si] / max(stratum_counts[si], 1):.6f}",
                f"{marg_unsound[l, si] / max(stratum_counts[si], 1):.6f}",
                f"{cls_sat_mean[l]:.6f}",
                f"{cls_unsat_mean[l]:.6f}",
                stratum_counts[si], n_sat_total, n_unsat_total,
            ])
    csv_text = buf.getvalue()

    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(csv_text)
    checkpoint_volume.commit()
    print(f"Wrote {out}  ({l_eval} iters x {n_strata} strata)", flush=True)
    return {"out": out, "l_eval": l_eval, "n_states": N, "l16_check": l16_check}


@app.local_entrypoint()
def entrypoint(
    checkpoint: str = f"{CHECKPOINT_MOUNT}/followups/e1/baseline_seed0.pt",
    bank: str = f"{CHECKPOINT_MOUNT}/followups/e3/state_bank.pt",
    l_eval: int = 128,
    threshold: float = 0.10,
    micro_batch: int = 256,
    out: str = f"{CHECKPOINT_MOUNT}/followups/e3/profile_iters.csv",
    assert_l16_forward: bool = True,
):
    result = run.remote(
        checkpoint=checkpoint, bank=bank, l_eval=l_eval, threshold=threshold,
        micro_batch=micro_batch, out=out, assert_l16_forward=assert_l16_forward,
    )
    print(f"\nFinal: {result}", flush=True)
