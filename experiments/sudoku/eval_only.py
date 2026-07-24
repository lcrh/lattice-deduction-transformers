"""Eval an existing checkpoint with the hybrid-batched solver (baseline).

Eval data: seed=200, zero_hint_weight=1.0 (all SAT), n=200 (post-filter).

Usage:
    uv run modal run --detach experiments/sudoku/eval_only.py \
      --checkpoint /checkpoints/sudoku/seed0_4000s_bs512_aug1_<ts>.pt

Extended operating points (multi-pass deduction, decision/backtrack
policies, routed artifact names) live under
`followups/deduction_operator/modal_eval.py` and
`followups/search_process/modal_eval.py`.
"""

import modal

from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)

from experiments.sudoku.eval_runner import EvalRunSpec, run_eval


app = modal.App("sudoku-eval-only")


@app.function(
    image=image, gpu="B200", timeout=7200,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    checkpoint: str,
    n_eval: int = 200,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.6,
    n_chains: int = 64,
    batch_size: int = 512,
    max_rounds: int = 1000,
    augment: bool = True,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    dropout_p: float = 0.05,
    log_per_round_fill: bool = False,
    out_suffix: str = ".eval.fixed.json",
    split: str = "test",
    compile: bool = False,
    eval_max_timeouts: int = 50,
):
    return run_eval(EvalRunSpec(
        checkpoint=checkpoint,
        n_eval=n_eval,
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        n_chains=n_chains,
        batch_size=batch_size,
        max_rounds=max_rounds,
        augment=augment,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        dropout_p=dropout_p,
        log_per_round_fill=log_per_round_fill,
        out_suffix=out_suffix,
        split=split,
        compile=compile,
        eval_max_timeouts=eval_max_timeouts,
    ))


@app.local_entrypoint()
def entrypoint(
    checkpoint: str,
    n_eval: int = 200,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.6,
    n_chains: int = 64,
    batch_size: int = 512,
    max_rounds: int = 1000,
    augment: bool = True,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    dropout_p: float = 0.05,
    log_per_round_fill: bool = False,
    out_suffix: str = ".eval.fixed.json",
    split: str = "test",
    compile: bool = False,
    eval_max_timeouts: int = 50,
):
    result = run.remote(
        checkpoint=checkpoint, n_eval=n_eval,
        threshold=threshold, temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        n_chains=n_chains, batch_size=batch_size, max_rounds=max_rounds,
        augment=augment,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        dropout_p=dropout_p,
        log_per_round_fill=log_per_round_fill,
        eval_max_timeouts=eval_max_timeouts,
        out_suffix=out_suffix,
        split=split,
        compile=compile,
    )
    print(f"\nFinal: {result}", flush=True)
