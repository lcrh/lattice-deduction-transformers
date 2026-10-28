"""Fill the data volume's dataset caches once, before launching many runs in parallel.

Usage:
    uv run modal run experiments/prefetch_data.py
"""

import modal

from ldt.modal.image import DATA_MOUNT, data_volume, hf_secret, image

app = modal.App("ldt-prefetch-data")


@app.function(image=image, cpu=4, memory=32768, timeout=3600, secrets=[hf_secret],
              volumes={DATA_MOUNT: data_volume})
def prefetch() -> None:
    from ldt.data import maze_hard, sudoku_extreme

    for split in ("train", "test"):
        print(sudoku_extreme._download_and_cache(DATA_MOUNT, split), flush=True)
        print(maze_hard._download_and_cache(DATA_MOUNT, split), flush=True)
    data_volume.commit()


@app.local_entrypoint()
def main() -> None:
    prefetch.remote()
