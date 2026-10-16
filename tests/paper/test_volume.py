"""The launcher's volume views: one listing per pass, retries, cached ledger files."""

from types import SimpleNamespace

import modal
import pytest

from paper import volume


class FakeModalVolume:
    """Counts calls; the first listing hits the rate limit."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.listings = 0
        self.reads = 0

    def listdir(self, path, recursive=False):
        assert path == "/" and recursive
        self.listings += 1
        if self.listings == 1:
            raise modal.exception.ResourceExhaustedError("VolumeListFiles rate limit exceeded")
        return [SimpleNamespace(path=p) for p in self.files]

    def read_file(self, path):
        self.reads += 1
        yield self.files[path]


@pytest.fixture
def fake(monkeypatch):
    vol = FakeModalVolume({
        "sudoku/a-seed0.pt": b"",
        "eval/a-seed0/tag-1/summary.json": b"{}",
        "eval/a-seed0/tag-1/blocks/000000.jsonl": b"",
        "ledger/1.json": b'{"x": 1}',
    })
    monkeypatch.setattr(modal.Volume, "from_name", lambda name: vol)
    monkeypatch.setattr(volume.time, "sleep", lambda s: None)
    return vol


def test_one_listing_answers_every_directory(fake, tmp_path):
    view = volume.ModalVolumeView(snapshot=tmp_path / "snap.json", file_cache=tmp_path / "files")
    assert view.listdir("") == ["eval", "ledger", "sudoku"]
    assert view.exists("sudoku/a-seed0.pt") and not view.exists("sudoku/b-seed0.pt")
    assert view.listdir("eval/a-seed0/tag-1") == ["blocks", "summary.json"]
    assert view.listdir("eval/a-seed0/tag-1/blocks") == ["000000.jsonl"]
    assert view.listdir("eval/missing") == []
    assert fake.listings == 2           # one rate-limited attempt, then one listing for all


def test_ledger_files_are_read_once_and_replayed_offline(fake, tmp_path):
    snap, files = tmp_path / "snap.json", tmp_path / "files"
    view = volume.ModalVolumeView(snapshot=snap, file_cache=files)
    assert view.listdir("ledger") == ["1.json"]
    assert view.read("ledger/1.json") == b'{"x": 1}'
    assert view.read("ledger/1.json") == b'{"x": 1}'
    view.read("eval/a-seed0/tag-1/summary.json")
    view.read("eval/a-seed0/tag-1/summary.json")
    assert fake.reads == 3              # summaries are re-read, ledger files are not
    view.save_snapshot()

    offline = volume.SnapshotVolumeView(snapshot=snap, file_cache=files)
    assert offline.listdir("ledger") == ["1.json"]
    assert offline.read("ledger/1.json") == b'{"x": 1}'
    with pytest.raises(RuntimeError):
        offline.read("eval/a-seed0/tag-1/summary.json")


def test_retry_gives_up_after_the_last_try(monkeypatch):
    monkeypatch.setattr(volume.time, "sleep", lambda s: None)

    def always_limited():
        raise modal.exception.ResourceExhaustedError("limit")

    with pytest.raises(modal.exception.ResourceExhaustedError):
        volume.with_retry(always_limited, tries=3)
