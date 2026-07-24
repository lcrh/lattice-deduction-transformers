"""E3 Modal eval: multi-pass deduction + eval-time loop override.

Composes followup strategies onto the shared eval runner. Artifact naming /
skip-if-done / collect schemas match the previous shared-CLI contract.
"""

from __future__ import annotations

import modal

from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.eval_runner import EvalRunSpec, run_eval
from experiments.sudoku.solve import SolveConfig
from followups.deduction_operator.step_extension import (
    attach_step_extension, make_extended_step,
)
from followups.search_process.search import make_search_strategy


app = modal.App("followup-e3-eval")


def _enrich_e3(out, res, ctx):
    deduce_passes = out["solver_config"].get("deduce_passes", 1)
    if res.per_pass_deduced_total:
        pp = [
            (i, d, u, (u / d if d else 0.0))
            for i, (d, u) in enumerate(
                zip(res.per_pass_deduced_total, res.per_pass_unsound_total))
        ]
        print(f"  Per-pass unsound (deduce_passes={deduce_passes}):", flush=True)
        for i, d, u, r in pp:
            print(f"    pass {i}: {u} unsound / {d} deduced  (rate={r:.4%})",
                  flush=True)


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
    eval_n_loops: int = 0,
    deduce_passes: int = 1,
    deduce_pass_cap: int = 16,
    ckpt_name: str = "",
    ckpt_subdir: str = "",
    skip_if_done: bool = False,
):
    step_cfg = StepConfig(
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        augment=augment,
    )
    ext = make_extended_step(
        deduce_passes=deduce_passes,
        deduce_pass_cap=deduce_pass_cap,
    )
    attach_step_extension(step_cfg, ext)
    search = make_search_strategy(track_per_pass=(deduce_passes != 1))
    _max_to = eval_max_timeouts if eval_max_timeouts > 0 else None
    solve_cfg = SolveConfig(
        step=step_cfg, max_rounds=max_rounds,
        n_chains=n_chains, batch_size=batch_size,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        log_per_round_fill=log_per_round_fill,
        eval_max_timeouts=_max_to,
        _search=search,
    )
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
        eval_n_loops=eval_n_loops,
        ckpt_name=ckpt_name,
        ckpt_subdir=ckpt_subdir,
        skip_if_done=skip_if_done,
        step_cfg=step_cfg,
        solve_cfg=solve_cfg,
        extra_solver_config={
            "deduce_passes": deduce_passes,
            "deduce_pass_cap": deduce_pass_cap,
        },
        enrich_result=_enrich_e3,
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
    eval_n_loops: int = 0,
    deduce_passes: int = 1,
    deduce_pass_cap: int = 16,
    ckpt_name: str = "",
    ckpt_subdir: str = "",
    skip_if_done: bool = False,
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
        out_suffix=out_suffix, split=split, compile=compile,
        eval_max_timeouts=eval_max_timeouts,
        eval_n_loops=eval_n_loops,
        deduce_passes=deduce_passes,
        deduce_pass_cap=deduce_pass_cap,
        ckpt_name=ckpt_name, ckpt_subdir=ckpt_subdir,
        skip_if_done=skip_if_done,
    )
    print(f"\nFinal: {result}", flush=True)
