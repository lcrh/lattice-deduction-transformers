"""E3-O2 — harvest a stratified state bank from baseline solve trajectories.

The per-iteration elimination profiler (`profile_iters.py`) needs a fixed,
diverse set of intermediate solver states to run a single `return_all=True`
forward over. This script generates that bank: it runs the E1 baseline
checkpoint's solver on SAT eval puzzles, snapshots the batched state at a range
of solve rounds, stratifies the snapshots by fill level (early / mid / late),
and saves them (with each state's ground truth, fill fraction, and a SAT/UNSAT
reachability label) to the Modal volume at

    /checkpoints/followups/e3/state_bank.pt

The SAT/UNSAT label is a *reachability* flag: a snapshot is UNSAT once any of
its ground-truth bits has been eliminated (the true solution is no longer in
the lattice) — this is exactly the "did a wrong guess / unsound deduction
knock out the answer" event, so the bank carries both healthy (SAT) and
collapsed (UNSAT) states for the CLS-logit trajectory panel in O2.

This is a GENERATION script — committed, but NOT run here (needs a GPU + the
E1 baseline checkpoint on the volume). Launch:

    uv run modal run --detach followups/deduction_operator/state_bank.py -- \
        --checkpoint /checkpoints/followups/e1/baseline_seed0.pt

Optional flags: --n-puzzles, --n-chains, --max-rounds, --per-stratum
(target snapshots kept per fill stratum), --out (volume path).

The saved object is a dict of stacked tensors + metadata:
    {
      "state":  [N, S, C] float   — snapshot states (canonical frame),
      "gt":     [N, S, C] float   — one-hot ground truth per snapshot,
      "fill":   [N] float         — fraction of cells that are singletons,
      "sat":    [N] bool          — GT still reachable (no GT bit killed),
      "stratum":[N] int8          — 0=early, 1=mid, 2=late (by fill),
      "meta":   {checkpoint, n_loops, strata_edges, counts, ...},
    }
`profile_iters.py` loads this and iterates the model over `state`.
"""

from __future__ import annotations

import modal
import torch

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig, SudokuExtremeDataset
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import swap_in_ema_if_present

app = modal.App("e3-state-bank")

# Fill-level strata by singleton fraction (fraction of cells decided). These
# edges bucket a snapshot into early / mid / late solve. [0, .33) / [.33, .66)
# / [.66, 1.0]. Documented so profile_iters.py can label its per-stratum lines.
STRATA_EDGES = (0.0, 0.34, 0.67, 1.01)
STRATA_NAMES = ("early", "mid", "late")


@app.function(
    image=image, gpu="B200", timeout=3600,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    checkpoint: str = f"{CHECKPOINT_MOUNT}/followups/e1/baseline_seed0.pt",
    n_puzzles: int = 200,
    n_chains: int = 32,
    batch_size: int = 512,
    max_rounds: int = 60,
    snapshot_every: int = 2,
    per_stratum: int = 400,
    threshold: float = 0.10,
    cls_threshold: float = 0.6,
    augment: bool = True,
    out: str = f"{CHECKPOINT_MOUNT}/followups/e3/state_bank.pt",
    seed: int = 200,
):
    """Harvest and save the stratified state bank.

    Runs a plain (no eviction bookkeeping) batched solve: fill B = M*K rows
    with M puzzles x K chains, step `dpll_step`, and snapshot the whole batch
    every `snapshot_every` rounds. Each snapshot row -> one bank entry, bucketed
    by its singleton fill fraction; we keep up to `per_stratum` rows per bucket
    (reservoir-trim at the end) so the bank is balanced across fill levels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    print(f"Loading baseline checkpoint: {checkpoint}", flush=True)
    ckpt = load_checkpoint(checkpoint)
    model_cfg = LoopedTransformerConfig(**ckpt["model_cfg"])
    model = PowersetModel(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
    n_loops = model_cfg.n_loops
    print(f"  n_loops(native)={n_loops}  params="
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    # SAT eval puzzles (same protocol as eval_only.py).
    ds = SudokuExtremeDataset(SudokuExtremeConfig(
        cache_dir=DATA_MOUNT, split="test", n_puzzles=n_puzzles,
        batch_size=n_puzzles, seed=seed,
        zero_hint_weight=1.0, correct_hint_weight=0.0, error_hint_weight=0.0,
        augment_digit_perm=False, augment_dihedral=False,
    ))
    x_all, y_all, sat = ds.next_batch(); ds.close()
    sat_mask = sat.bool()
    x = x_all[sat_mask].to(device).float()
    y = y_all[sat_mask].to(device).float()
    P, S, C = x.shape
    print(f"Loaded {P} SAT puzzles (S={S}, C={C})", flush=True)

    K = n_chains
    M = max(1, batch_size // K)
    step_cfg = StepConfig(threshold=threshold, cls_threshold=cls_threshold,
                          augment=augment)

    # Per-stratum accumulators (lists of CPU tensors, trimmed at the end).
    bank_state: list[list[torch.Tensor]] = [[] for _ in STRATA_NAMES]
    bank_gt: list[list[torch.Tensor]] = [[] for _ in STRATA_NAMES]
    bank_fill: list[list[float]] = [[] for _ in STRATA_NAMES]
    bank_sat: list[list[bool]] = [[] for _ in STRATA_NAMES]

    def _stratum_of(fill: float) -> int:
        for i in range(len(STRATA_NAMES)):
            if STRATA_EDGES[i] <= fill < STRATA_EDGES[i + 1]:
                return i
        return len(STRATA_NAMES) - 1

    def _snapshot(state_b: torch.Tensor, gt_b: torch.Tensor) -> None:
        """Bucket each row of the current batch into the bank by fill level.

        state_b, gt_b: [B, S, C]. fill = fraction of cells with exactly one
        alive bit. sat = no GT bit has been eliminated (GT still reachable).
        """
        alive = (state_b > 0.5)
        n_alive = alive.sum(dim=-1)                       # [B, S]
        singleton = (n_alive == 1)                        # [B, S]
        fill = singleton.float().mean(dim=-1)             # [B]
        gt_bool = (gt_b > 0.5)
        gt_reachable = (alive & gt_bool).any(dim=-1)      # [B, S] — GT bit alive
        sat = gt_reachable.all(dim=-1)                    # [B]
        fill_cpu = fill.cpu()
        sat_cpu = sat.cpu()
        state_cpu = state_b.cpu()
        gt_cpu = gt_b.cpu()
        for b in range(state_b.shape[0]):
            f = float(fill_cpu[b])
            si = _stratum_of(f)
            if len(bank_fill[si]) >= per_stratum:
                continue  # this stratum is full
            bank_state[si].append(state_cpu[b].clone())
            bank_gt[si].append(gt_cpu[b].clone())
            bank_fill[si].append(f)
            bank_sat[si].append(bool(sat_cpu[b]))

    def _full() -> bool:
        return all(len(b) >= per_stratum for b in bank_fill)

    # Stream puzzles through in blocks of M; run each block for up to
    # max_rounds, snapshotting the batch every snapshot_every rounds.
    next_p = 0
    while next_p < P and not _full():
        block = list(range(next_p, min(next_p + M, P)))
        next_p += len(block)
        m = len(block)
        B = m * K
        state = torch.zeros(B, S, C, device=device)
        gt_b = torch.zeros(B, S, C, device=device)
        gm_b = torch.zeros(B, S, dtype=torch.bool, device=device)
        original = torch.zeros(B, S, C, device=device)
        for i, p in enumerate(block):
            rows = slice(i * K, (i + 1) * K)
            state[rows] = x[p].unsqueeze(0).expand(K, -1, -1)
            original[rows] = x[p].unsqueeze(0).expand(K, -1, -1)
            gt_b[rows] = y[p].unsqueeze(0).expand(K, -1, -1)
            gm_b[rows] = (x[p].sum(dim=-1) == 1).unsqueeze(0).expand(K, -1)

        with torch.no_grad():
            for r in range(max_rounds):
                if r % snapshot_every == 0:
                    _snapshot(state, gt_b)
                    if _full():
                        break
                new_state, conflict, solved, _, _ = dpll_step(
                    model, state, gm_b, step_cfg, want_stats=False,
                )
                # Reset conflicted chains to their puzzle original (so a bad
                # guess doesn't freeze the trajectory) — mirrors solve()'s
                # reset, minus the eviction bookkeeping.
                if conflict.any():
                    new_state[conflict] = original[conflict]
                # Freeze solved chains (leave them; harmless to keep stepping).
                state = new_state
        counts = [len(b) for b in bank_fill]
        print(f"  block puzzles {block[0]}..{block[-1]}  "
              f"bank counts (early/mid/late) = {counts}", flush=True)

    # Stack + trim each stratum to per_stratum, then concatenate.
    states, gts, fills, sats, strata = [], [], [], [], []
    for si, name in enumerate(STRATA_NAMES):
        n = min(len(bank_fill[si]), per_stratum)
        if n == 0:
            continue
        states.append(torch.stack(bank_state[si][:n]))
        gts.append(torch.stack(bank_gt[si][:n]))
        fills.append(torch.tensor(bank_fill[si][:n], dtype=torch.float32))
        sats.append(torch.tensor(bank_sat[si][:n], dtype=torch.bool))
        strata.append(torch.full((n,), si, dtype=torch.int8))

    if not states:
        raise RuntimeError("state bank empty — no snapshots harvested "
                           "(check checkpoint / puzzle count).")

    bank = {
        "state": torch.cat(states),
        "gt": torch.cat(gts),
        "fill": torch.cat(fills),
        "sat": torch.cat(sats),
        "stratum": torch.cat(strata),
        "meta": {
            "checkpoint": checkpoint,
            "n_loops_native": n_loops,
            "strata_edges": list(STRATA_EDGES),
            "strata_names": list(STRATA_NAMES),
            "per_stratum_counts": [int(x.shape[0]) for x in states],
            "threshold": threshold,
            "cls_threshold": cls_threshold,
            "augment": augment,
            "S": S, "C": C,
        },
    }
    total = int(bank["state"].shape[0])
    n_sat = int(bank["sat"].sum().item())
    print(f"\nState bank: {total} snapshots "
          f"(strata {bank['meta']['per_stratum_counts']}), "
          f"SAT={n_sat} UNSAT={total - n_sat}", flush=True)

    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, out)
    checkpoint_volume.commit()
    print(f"Wrote {out}", flush=True)
    return {"out": out, "total": total, "sat": n_sat}


@app.local_entrypoint()
def entrypoint(
    checkpoint: str = f"{CHECKPOINT_MOUNT}/followups/e1/baseline_seed0.pt",
    n_puzzles: int = 200,
    n_chains: int = 32,
    batch_size: int = 512,
    max_rounds: int = 60,
    snapshot_every: int = 2,
    per_stratum: int = 400,
    threshold: float = 0.10,
    cls_threshold: float = 0.6,
    augment: bool = True,
    out: str = f"{CHECKPOINT_MOUNT}/followups/e3/state_bank.pt",
    seed: int = 200,
):
    result = run.remote(
        checkpoint=checkpoint, n_puzzles=n_puzzles, n_chains=n_chains,
        batch_size=batch_size, max_rounds=max_rounds,
        snapshot_every=snapshot_every, per_stratum=per_stratum,
        threshold=threshold, cls_threshold=cls_threshold, augment=augment,
        out=out, seed=seed,
    )
    print(f"\nFinal: {result}", flush=True)
