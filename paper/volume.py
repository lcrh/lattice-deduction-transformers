"""Read-only views of the checkpoint volume for the launcher.

`ModalVolumeView` lists and reads files with the Modal SDK and saves every
listing to a local snapshot, which `SnapshotVolumeView` replays offline
(`status --offline`, `launch --dry-run`).
"""

from __future__ import annotations

import json
import posixpath
from abc import ABC, abstractmethod
from pathlib import Path

VOLUME_NAME = "lattice-diffusion-checkpoints"   # lattice_diffusion.modal.image
SNAPSHOT = Path(__file__).resolve().parent / ".cache" / "volume_listing.json"


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


class ModalVolumeView(VolumeView):
    def __init__(self, name: str = VOLUME_NAME, snapshot: Path | None = SNAPSHOT):
        import modal

        self._volume = modal.Volume.from_name(name)
        self._cache: dict[str, list[str]] = {}
        self._snapshot = snapshot

    def listdir(self, path: str) -> list[str]:
        import modal

        if path not in self._cache:
            try:
                entries = self._volume.listdir(path or "/")
            except modal.exception.NotFoundError:
                entries = []
            self._cache[path] = sorted(posixpath.basename(e.path.rstrip("/")) for e in entries)
        return self._cache[path]

    def read(self, path: str) -> bytes:
        return b"".join(self._volume.read_file(path))

    def save_snapshot(self) -> None:
        if self._snapshot is None:
            return
        self._snapshot.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(self._snapshot.read_text()) if self._snapshot.exists() else {}
        self._snapshot.write_text(json.dumps({**old, **self._cache}, indent=1, sort_keys=True))


class SnapshotVolumeView(VolumeView):
    """The last listings `ModalVolumeView` saw; file contents are not available offline."""

    def __init__(self, snapshot: Path = SNAPSHOT):
        self._listing = json.loads(snapshot.read_text()) if snapshot.exists() else {}

    def listdir(self, path: str) -> list[str]:
        return self._listing.get(path, [])

    def read(self, path: str) -> bytes:
        raise RuntimeError(f"offline: cannot read {path}")
