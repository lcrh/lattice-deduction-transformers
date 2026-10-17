"""Shared Modal image, volumes and secret for training and eval."""

import modal

# Dataset caches (HF downloads, generated data)
data_volume = modal.Volume.from_name("lattice-diffusion-data", create_if_missing=True)
DATA_MOUNT = "/data"
HF_HOME = f"{DATA_MOUNT}/huggingface"   # HF downloads persist across containers

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .uv_sync()
    .env({"HF_HOME": HF_HOME})
    .add_local_python_source("lattice_diffusion")
    .add_local_python_source("experiments")
)

# Checkpoints, eval runs and the launch ledger
checkpoint_volume = modal.Volume.from_name("lattice-diffusion-checkpoints", create_if_missing=True)
CHECKPOINT_MOUNT = "/checkpoints"

# HuggingFace token for gated datasets
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
