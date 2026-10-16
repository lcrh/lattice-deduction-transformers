"""Read-only views of the checkpoint volume for the launcher.

`ModalVolumeView` lists the whole volume with one recursive call per pass
(Modal rate-limits listings) and saves that listing to a local snapshot,
which `SnapshotVolumeView` replays offline (`status --offline`,
`launch --dry-run`). Ledger files never change once written, so both views
keep a local copy of every one they read.
"""

from __future__ import annotations

import json
import posixpath
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable, TypeVar

VOLUME_NAME = "lattice-diffusion-checkpoints"   # lattice_diffusion.modal.image
CACHE = Path(__file__).resolve().parent / ".cache"
SNAPSHOT = CACHE / "volume_listing.json"
FILE_CACHE = CACHE / "files"
IMMUTABLE = ("ledger/",)        # volume paths whose contents never change

T = TypeVar("T")


class VolumeView(ABC):
    """Read-only access to volume-relative paths."""

    @abstractmethod
    def listdir(self, path: str) -> list[str]:
        """Entry names under a directory ([] if it is missing)."""

    @abstractmethod
    def read(self, path: str) -> bytes: ...

    def exists(self, path: str) -> bool:
        parent, name = posixpath.split(path.rstrip("/"))
        return name in self.listdir(parent)


def tree(paths: list[str]) -> dict[str, list[str]]:
    """{directory: sorted entry names} for every directory on the given paths."""
    out: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        parts = path.strip("/").split("/")
        for i, name in enumerate(parts):
            out["/".join(parts[:i])].add(name)
    return {d: sorted(names) for d, names in out.items()}


def with_retry(fn: Callable[[], T], tries: int = 6, delay: float = 1.0) -> T:
    """Call `fn`, backing off exponentially while Modal reports a rate limit."""
    import modal

    for attempt in range(tries):
        try:
            return fn()
        except modal.exception.ResourceExhaustedError:
            if attempt == tries - 1:
                raise
            time.sleep(delay * 2 ** attempt)
    raise AssertionError("unreachable")


def _cached(path: str, file_cache: Path) -> Path | None:
    return file_cache / path if path.startswith(IMMUTABLE) else None


class ModalVolumeView(VolumeView):
    def __init__(self, name: str = VOLUME_NAME, snapshot: Path | None = SNAPSHOT,
                 file_cache: Path = FILE_CACHE):
        import modal

        self._volume = modal.Volume.from_name(name)
        self._tree: dict[str, list[str]] | None = None
        self._snapshot = snapshot
        self._file_cache = file_cache

    def listdir(self, path: str) -> list[str]:
        if self._tree is None:
            entries = with_retry(lambda: self._volume.listdir("/", recursive=True))
            self._tree = tree([e.path for e in entries])
        return self._tree.get(path.strip("/"), [])

    def read(self, path: str) -> bytes:
        local = _cached(path, self._file_cache)
        if local is not None and local.exists():
            return local.read_bytes()
        data = with_retry(lambda: b"".join(self._volume.read_file(path)))
        if local is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        return data

    def save_snapshot(self) -> None:
        if self._snapshot is None or self._tree is None:
            return
        self._snapshot.parent.mkdir(parents=True, exist_ok=True)
        self._snapshot.write_text(json.dumps(self._tree, sort_keys=True))


class SnapshotVolumeView(VolumeView):
    """The last listing `ModalVolumeView` saw, plus locally cached ledger files."""

    def __init__(self, snapshot: Path = SNAPSHOT, file_cache: Path = FILE_CACHE):
        self._listing = json.loads(snapshot.read_text()) if snapshot.exists() else {}
        self._file_cache = file_cache

    def listdir(self, path: str) -> list[str]:
        return self._listing.get(path.strip("/"), [])

    def read(self, path: str) -> bytes:
        local = _cached(path, self._file_cache)
        if local is None or not local.exists():
            raise RuntimeError(f"offline: cannot read {path}")
        return local.read_bytes()
