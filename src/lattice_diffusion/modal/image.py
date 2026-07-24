"""Shared Modal image and volume definitions for training jobs."""

import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .uv_sync()
    .add_local_python_source("lattice_diffusion")
    .add_local_python_source("experiments")
    # E2/E3 Modal entrypoints import strategy hooks from this package.
    .add_local_python_source("followups")
)

# For caching the sudoku-extreme HF download across runs
data_volume = modal.Volume.from_name("lattice-diffusion-data", create_if_missing=True)
DATA_MOUNT = "/data"

# For persisting checkpoints across runs
checkpoint_volume = modal.Volume.from_name("lattice-diffusion-checkpoints", create_if_missing=True)
CHECKPOINT_MOUNT = "/checkpoints"

# HuggingFace token for gated datasets
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
