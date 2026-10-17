"""Test doubles for the paper tooling."""

from __future__ import annotations

from paper.volume import VolumeView


class FakeVolumeView(VolumeView):
    """An in-memory volume: {path: contents}."""

    def __init__(self, files: dict[str, bytes | str] | None = None):
        self.files = {k: v.encode() if isinstance(v, str) else v for k, v in (files or {}).items()}

    def listdir(self, path: str) -> list[str]:
        prefix = path.rstrip("/") + "/" if path else ""
        return sorted({f[len(prefix):].split("/", 1)[0] for f in self.files if f.startswith(prefix)})

    def read(self, path: str) -> bytes:
        return self.files[path]
