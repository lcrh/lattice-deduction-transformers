"""Modal entry point for sudoku: train, then evaluate with the shared evaluator.

Training saves a checkpoint and spawns `lattice_diffusion.eval` on it
(sequential stopping on the full test split by default; see
experiments/sudoku/evaluate.py to re-evaluate a checkpoint).

Usage:
    uv run modal run experiments/sudoku/run.py
    uv run modal run experiments/sudoku/run.py --steps 2000 --batch-size 512
"""

import time
from typing import Literal

import modal


# Keep in sync with run() signature defaults. Used for the run-config table
# at the top of every run, with non-defaults highlighted.
_RUN_PARAMS: tuple[tuple[str, object], ...] = (
    ("steps",                    4000),
    ("batch_size",               512),
    ("n_train_puzzles",          None),
    ("seed",                     0),
    ("bce_pos_mult",             4.0),
    ("bce_neg_mult",             0.5),
    ("softmax_loss_weight",      0.2),
    ("conflict_loss_weight",     0.1),
    ("weight_decay",             0.1),
    ("lr",                       3e-3),
    ("max_age",                  100),
    ("warmup_fraction",          0.1),
    ("threshold",                0.10),
    ("temp_decide",              1.5),
    ("cls_threshold",            0.5),
    ("eval_cls_threshold",       0.6),
    ("eval_max_rounds",          1000),
    ("eval_n_chains",            64),
    ("eval_batch_size",          512),
    ("augment",                  True),
    ("data_augment_digit_perm",  True),
    ("data_augment_dihedral",    True),
    ("use_ema",                  False),
    ("ema_decay",                0.999),
    ("eval_dropout_p",           0.05),
    ("eval_stop",                "sequential"),
    ("eval_workers",             32),
    ("n_loops",                  16),
    ("pre_norm",                 True),
)


def _print_run_config(values: dict) -> None:
    """Tabular dump of run() args at startup. Non-default values are starred."""
    BOLD, RESET = "\033[1m", "\033[0m"
    print("=" * 64, flush=True)
    print(f"RUN CONFIG  (* = non-default; values bolded if your terminal supports ANSI)",
          flush=True)
    print("=" * 64, flush=True)
    n_changed = 0
    for name, default in _RUN_PARAMS:
        val = values.get(name, "<unset>")
        is_changed = (val != default)
        if is_changed:
            n_changed += 1
            marker = "*"
            shown = f"{BOLD}{val!r}{RESET}"
            tail = f"   (default {default!r})"
        else:
            marker = " "
            shown = f"{val!r}"
            tail = ""
        print(f"  {marker} {name:<28} = {shown}{tail}", flush=True)
    print(f"  {n_changed}/{len(_RUN_PARAMS)} non-default", flush=True)
    print("=" * 64, flush=True)

from lattice_diffusion.data.sudoku_extreme import SudokuExtremeConfig
from lattice_diffusion.eval.modal_app import EVAL_ROOT, coordinate, eval_app
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.eval_task import make_eval_spec
from experiments.sudoku.train import TrainConfig, train


app = modal.App("sudoku")
app.include(eval_app)


@app.function(
    image=image,
    gpu="B200",
    timeout=3600 * 4,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    steps: int = 4000,
    batch_size: int = 512,
    n_train_puzzles: int | None = None,
    seed: int = 0,
    bce_pos_mult: float = 4.0,
    bce_neg_mult: float = 0.5,
    softmax_loss_weight: float = 0.2,
    conflict_loss_weight: float = 0.1,
    weight_decay: float = 0.1,
    lr: float = 3e-3,
    max_age: int = 100,
    warmup_fraction: float = 0.1,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.5,
    eval_cls_threshold: float = 0.6,
    eval_max_rounds: int = 1000,
    eval_n_chains: int = 64,
    eval_batch_size: int = 512,
    augment: bool = True,
    data_augment_digit_perm: bool = True,
    data_augment_dihedral: bool = True,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    eval_dropout_p: float = 0.05,
    eval_stop: Literal["sequential", "full"] = "sequential",
    eval_workers: int = 32,
    n_loops: int = 16,
    pre_norm: bool = True,
):
    # Snapshot the call-site arg values BEFORE any local mutation, then dump
    # the config table. (Snapshot locals() outside the comprehension —
    # Python 3 comprehensions have their own scope, so locals() inside one
    # only sees the iter-vars, NOT the function's parameters.)
    _loc_snapshot = dict(locals())
    _arg_values = {name: _loc_snapshot[name] for name, _ in _RUN_PARAMS}
    _print_run_config(_arg_values)

    ts = time.strftime("%Y%m%d_%H%M%S")

    step_cfg = StepConfig(
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        augment=augment,
    )
    model_cfg = LoopedTransformerConfig(
        cls_token=conflict_loss_weight > 0,
        n_loops=n_loops,
        pre_norm=pre_norm,
    )
    ckpt_path = train(TrainConfig(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        lr=lr,
        weight_decay=weight_decay,
        bce_pos_mult=bce_pos_mult,
        bce_neg_mult=bce_neg_mult,
        softmax_loss_weight=softmax_loss_weight,
        conflict_loss_weight=conflict_loss_weight,
        warmup_fraction=warmup_fraction,
        step=step_cfg,
        max_age=max_age,
        use_ema=use_ema,
        ema_decay=ema_decay,
        model=model_cfg,
        data=SudokuExtremeConfig(
            cache_dir=DATA_MOUNT, batch_size=batch_size, seed=42,
            n_puzzles=n_train_puzzles,
            augment_digit_perm=data_augment_digit_perm,
            augment_dihedral=data_augment_dihedral,
        ),
        out_dir=f"{CHECKPOINT_MOUNT}/sudoku",
        name=f"seed{seed}_{steps}s_bs{batch_size}_aug{int(augment)}_{ts}",
    ))
    checkpoint_volume.commit()

    # Evaluate through the shared evaluator. Spawned rather than awaited so
    # the training GPU is released, and so a --detach run keeps evaluating
    # after the local client disconnects.
    spec = make_eval_spec(
        str(ckpt_path), stop=eval_stop, workers=eval_workers,
        threshold=threshold, temp_decide=temp_decide,
        cls_threshold=eval_cls_threshold, augment=augment,
        n_chains=eval_n_chains, batch_size=eval_batch_size,
        max_rounds=eval_max_rounds, dropout_p=eval_dropout_p,
    )
    eval_call = coordinate.spawn(spec.to_json())
    print(f"Spawned eval {spec.run_dir(EVAL_ROOT)} "
          f"(call {eval_call.object_id})", flush=True)
    return {"checkpoint": str(ckpt_path), "eval_call_id": eval_call.object_id}


@app.local_entrypoint()
def entrypoint(
    steps: int = 4000,
    batch_size: int = 512,
    n_train_puzzles: int | None = None,
    seed: int = 0,
    bce_pos_mult: float = 4.0,
    bce_neg_mult: float = 0.5,
    softmax_loss_weight: float = 0.2,
    conflict_loss_weight: float = 0.1,
    weight_decay: float = 0.1,
    lr: float = 3e-3,
    max_age: int = 100,
    warmup_fraction: float = 0.1,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.5,
    eval_cls_threshold: float = 0.6,
    eval_max_rounds: int = 1000,
    eval_n_chains: int = 64,
    eval_batch_size: int = 512,
    augment: bool = True,
    data_augment_digit_perm: bool = True,
    data_augment_dihedral: bool = True,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    eval_dropout_p: float = 0.05,
    eval_stop: Literal["sequential", "full"] = "sequential",
    eval_workers: int = 32,
    n_loops: int = 16,
    pre_norm: bool = True,
):
    result = run.remote(
        steps=steps, batch_size=batch_size,
        n_train_puzzles=n_train_puzzles, seed=seed,
        bce_pos_mult=bce_pos_mult, bce_neg_mult=bce_neg_mult,
        softmax_loss_weight=softmax_loss_weight,
        conflict_loss_weight=conflict_loss_weight,
        weight_decay=weight_decay,
        lr=lr,
        max_age=max_age,
        warmup_fraction=warmup_fraction,
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        eval_cls_threshold=eval_cls_threshold,
        eval_max_rounds=eval_max_rounds,
        eval_n_chains=eval_n_chains,
        eval_batch_size=eval_batch_size,
        augment=augment,
        data_augment_digit_perm=data_augment_digit_perm,
        data_augment_dihedral=data_augment_dihedral,
        use_ema=use_ema,
        ema_decay=ema_decay,
        eval_dropout_p=eval_dropout_p,
        eval_stop=eval_stop,
        eval_workers=eval_workers,
        n_loops=n_loops,
        pre_norm=pre_norm,
    )
    print(f"\nTrained: {result['checkpoint']}", flush=True)
    # Wait so a non-detached run does not tear down the spawned eval.
    eval_result = modal.FunctionCall.from_id(result["eval_call_id"]).get()
    print(f"\n{eval_result['run_dir']}\n{eval_result['summary']}", flush=True)
