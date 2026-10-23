from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import torch


def _strip_compile_prefix(state: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def _as_dict(cfg):
    return dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_cfg,
    train_cfg=None,
    trajectory: list | None = None,
    extra: dict | None = None,
) -> Path:
    """Save a training checkpoint with model state, configs, and trajectory.

    Args:
        path: File path for the checkpoint (.pt).
        model: The model (unwrapped from DataParallel if needed).
        model_cfg: Dataclass or dict with model architecture config.
        train_cfg: Optional dataclass or dict with training config.
        trajectory: Optional list of eval snapshots during training.
        extra: Optional extra metadata to include.

    Returns:
        The resolved Path where the checkpoint was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_state_dict": _strip_compile_prefix(model.state_dict()),
        "model_cfg": _as_dict(model_cfg),
    }
    if train_cfg is not None:
        data["train_cfg"] = _as_dict(train_cfg)
    if trajectory is not None:
        data["trajectory"] = trajectory
    if extra is not None:
        data.update(extra)

    # Write-then-rename: an interrupted save never leaves a truncated checkpoint.
    tmp = path.with_name(path.name + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    """Load a checkpoint dict, stripping torch.compile prefixes if present."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        ckpt["model_state_dict"] = _strip_compile_prefix(ckpt["model_state_dict"])
    return ckpt
