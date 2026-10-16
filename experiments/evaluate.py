"""Evaluate an existing checkpoint with the shared parallel evaluator.

The checkpoint carries the experiment config it was trained with; the
`eval` and `solve` blocks of that config are used, with `--set` overrides
on top. Results land on the checkpoint volume under
/checkpoints/eval/<checkpoint stem>/; rerunning the same command resumes
from the blocks already written.

Usage:
    # The checkpoint's own eval settings
    uv run modal run --detach experiments/evaluate.py --checkpoint /checkpoints/sudoku/<name>.pt

    # Every test puzzle, more workers
    uv run modal run --detach experiments/evaluate.py --checkpoint <ckpt> \\
        --set "eval.stop=full eval.workers=100"

    # Explicit puzzle list (e.g. a portfolio failure set), one index per line
    uv run modal run --detach experiments/evaluate.py --checkpoint <ckpt> \\
        --indices-file hard.txt --split test --set "eval.stop=full solve.max_rounds=50000"

    # Submit and return (with --detach the eval keeps running)
    uv run modal run --detach experiments/evaluate.py --checkpoint <ckpt> --no-wait
"""

from pathlib import Path

import modal

from lattice_diffusion import config as cfglib
from lattice_diffusion.eval.modal_app import coordinate, eval_app
from lattice_diffusion.modal.image import CHECKPOINT_MOUNT, checkpoint_volume, image

from experiments.common import record_launch, with_code_sha
from experiments.registry import experiment_class


app = modal.App("evaluate")
app.include(eval_app)


@app.function(image=image, cpu=1, memory=4096, timeout=600,
              volumes={CHECKPOINT_MOUNT: checkpoint_volume})
def checkpoint_experiment(checkpoint: str) -> dict:
    """The benchmark and experiment config stored in a checkpoint."""
    import torch

    # mmap: only the config is read, not the weights.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    return {"benchmark": ckpt["benchmark"], "experiment": ckpt["experiment"]}


@app.local_entrypoint()
def entrypoint(checkpoint: str, set: str = "", indices_file: str = "", split: str = "test",
               wait: bool = True):
    stored = checkpoint_experiment.remote(checkpoint)
    cls = experiment_class(stored["benchmark"])
    data = stored["experiment"]
    if set:
        overrides = cfglib.parse_overrides(set)
        if overrides.keys() - {"eval", "solve", "provenance"}:
            raise ValueError("evaluate.py only overrides the eval, solve and provenance blocks")
        data = cfglib.merge(data, overrides)
    exp = with_code_sha(cfglib.from_dict(cls, data))

    explicit = None
    if indices_file:
        explicit = [int(tok) for tok in Path(indices_file).read_text().split()]
    spec = exp.eval_spec(checkpoint, explicit_indices=explicit,
                         explicit_split=split if explicit else None)
    print(f"Eval spec ({stored['benchmark']}): index_list={spec.index_list} "
          f"stop={spec.stop['kind']} tag={spec.tag} "
          f"explicit={len(explicit or [])} solve={spec.solve}", flush=True)
    call = coordinate.spawn(spec.to_json())
    record_launch(app, call, "eval", exp, checkpoint, spec)
    if wait:
        result = call.get()
        print(f"\n{result['run_dir']}\n{result['summary']}", flush=True)
