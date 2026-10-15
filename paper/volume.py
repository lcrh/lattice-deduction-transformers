"""Read-only views of the checkpoint volume for the launcher.

`ModalVolumeView` lists and reads files with the Modal SDK. Every directory
listing it fetches is also written to a local snapshot, which
`SnapshotVolumeView` replays for `status --offline` and `launch --dry-run`
without network access. `FakeVolumeView` backs the tests.
"""

from __future__ import annotations

import json
import posixpath
from pathlib import Path

VOLUME_NAME = "lattice-diffusion-checkpoints"   # lattice_diffusion.modal.image
SNAPSHOT = Path(__file__).resolve().parent / ".cache" / "volume_listing.json"


class VolumeView:
    """Base: `listdir` returns entry names under a volume-relative directory
    ([] if missing); `read` returns a file's bytes."""

    def listdir(self, path: str) -> list[str]:
        raise NotImplementedError

    def read(self, path: str) -> bytes:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        parent, name = posixpath.split(path.rstrip("/"))
        return name in self.listdir(parent)


class FakeVolumeView(VolumeView):
    def __init__(self, files: dict[str, bytes | str] | None = None):
        self.files = {k: v.encode() if isinstance(v, str) else v for k, v in (files or {}).items()}

    def listdir(self, path: str) -> list[str]:
        prefix = path.rstrip("/") + "/" if path else ""
        names = set()
        for f in self.files:
            if f.startswith(prefix):
                names.add(f[len(prefix):].split("/", 1)[0])
        return sorted(names)

    def read(self, path: str) -> bytes:
        return self.files[path]


class ModalVolumeView(VolumeView):
    def __init__(self, name: str = VOLUME_NAME, snapshot: Path | None = SNAPSHOT):
        import modal

        self._volume = modal.Volume.from_name(name)
        self._cache: dict[str, list[str]] = {}
        self._snapshot = snapshot

    def listdir(self, path: str) -> list[str]:
        if path not in self._cache:
            try:
                entries = self._volume.listdir(path or "/")
            except Exception as exc:     # missing directories raise NotFoundError
                if "not found" not in str(exc).lower() and "NotFound" not in type(exc).__name__:
                    raise
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
    """The last listings `ModalVolumeView` saw; unknown directories read as
    empty. File contents are not available offline."""

    def __init__(self, snapshot: Path = SNAPSHOT):
        self._listing = json.loads(snapshot.read_text()) if snapshot.exists() else {}

    def listdir(self, path: str) -> list[str]:
        return self._listing.get(path, [])

    def read(self, path: str) -> bytes:
        raise RuntimeError(f"offline: cannot read {path}")
