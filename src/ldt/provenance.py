"""Run provenance: which experiment a Modal run belongs to and which code ran.

A `Provenance` is stored in checkpoints and eval specs but never hashed, so
the same config launched from two commits is still one run.

Launches are recorded in a ledger on the checkpoint volume, one JSON file per
launch under `ledger/`: volume commits are last-writer-wins, so concurrent
launches cannot share one appended file.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal

LEDGER_DIR = "ledger"                   # on the checkpoint volume
ALLOW_DIRTY_ENV = "LDT_ALLOW_DIRTY"     # set to 1 to launch from a dirty tree (dev only)
_APP_NAME_MAX = 64


@dataclass
class Provenance:
    """Which experiment a run belongs to and which code it ran."""

    exp_id: str = ""                    # experiment tracker ID
    entry: str = ""                     # entry key, see entry_key()
    code_sha: str = ""                  # git SHA the run was launched from


def code_version(repo: str | Path | None = None) -> tuple[str, bool]:
    """(HEAD SHA, whether tracked files differ from it)."""
    cwd = str(repo) if repo else None
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
                           check=True, capture_output=True, text=True).stdout.strip() != ""
    return sha, dirty


def require_clean_sha(repo: str | Path | None = None) -> str:
    """HEAD SHA, refusing a dirty tree.

    With LDT_ALLOW_DIRTY=1 a dirty tree is allowed and the SHA is marked `-dirty`.
    """
    sha, dirty = code_version(repo)
    if not dirty:
        return sha
    if os.environ.get(ALLOW_DIRTY_ENV) == "1":
        return f"{sha}-dirty"
    raise RuntimeError(f"working tree has uncommitted changes; commit them before launching "
                       f"(or set {ALLOW_DIRTY_ENV}=1 for a throwaway dev run)")


def entry_key(exp_id: str, entry: str, seed: int, eval_seed: int | None = None) -> str:
    """`<exp_id>/<entry>/seed<s>`, plus `/eval<e>` for a nonzero eval seed."""
    key = f"{exp_id}/{entry}/seed{seed}"
    return key if not eval_seed else f"{key}/eval{eval_seed}"


def app_run_name(exp_id: str, entry: str, train_hash: str, eval_hash: str | None = None) -> str:
    """Modal app name `ldt-<exp_id>-<entry>-<trainhash8>[-<evalhash8>]`.

    Lowercased and limited to [a-z0-9-]; the entry part is shortened first so
    both hashes survive the length limit.
    """
    tail = f"-{train_hash[:8]}" + (f"-{eval_hash[:8]}" if eval_hash else "")
    head = f"ldt-{exp_id}-{entry}"
    head = re.sub(r"[^a-z0-9-]+", "-", head.lower()).strip("-")
    head = re.sub(r"-{2,}", "-", head)
    return head[:_APP_NAME_MAX - len(tail)].rstrip("-") + tail


def header(kind: str, **fields) -> str:
    """One greppable log line, e.g. `[ldt-train] exp_id=A1 entry=... ...`."""
    return f"[ldt-{kind}] " + " ".join(f"{k}={v}" for k, v in fields.items())


@dataclass
class LedgerRecord:
    kind: Literal["train", "train+eval", "eval", "command"]
    exp_id: str
    entry: str
    app_name: str
    app_id: str | None
    app_url: str | None
    call_id: str | None
    code_sha: str
    checkpoint: str
    train_hash: str
    eval_run_dir: str | None = None
    eval_hash: str | None = None
    ts: float = field(default_factory=time.time)

    def new_filename(self) -> str:
        """A fresh, time-sortable ledger path for this record."""
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(self.ts))
        return f"{LEDGER_DIR}/{stamp}-{uuid.uuid4().hex[:8]}.json"


def app_url(app) -> str | None:
    """Dashboard URL of a running app.

    Modal keeps the URL on a private attribute; the id-only form is the fallback.
    """
    running = getattr(app, "_running_app", None)
    url = getattr(running, "app_page_url", None)
    if url:
        return url
    return f"https://modal.com/apps/{app.app_id}" if app.app_id else None


def write_ledger(volume, record: LedgerRecord) -> str:
    """Upload one ledger file to a modal.Volume from the local client."""
    path = record.new_filename()
    with volume.batch_upload() as batch:
        batch.put_file(io.BytesIO(json.dumps(asdict(record)).encode()), path)
    return path


def read_ledger(list_files: Callable[[str], Iterable[str]],
                read_file: Callable[[str], bytes | None]) -> list[LedgerRecord]:
    """All ledger records, oldest first.

    Args:
        list_files: File names under a volume-relative directory.
        read_file: Bytes of a volume-relative path, or None to skip a file that
            cannot be read.
    """
    records = []
    for name in list_files(LEDGER_DIR):
        if name.endswith(".json") and (data := read_file(f"{LEDGER_DIR}/{name}")) is not None:
            records.append(LedgerRecord(**json.loads(data)))
    return sorted(records, key=lambda r: r.ts)


def announce(record: LedgerRecord, ledger_path: str) -> None:
    """Print the launch header and where to find the run."""
    print(header("launch", exp_id=record.exp_id, entry=record.entry, app=record.app_name,
                 app_id=record.app_id, call=record.call_id), flush=True)
    print(f"  dashboard: {record.app_url}\n  ledger: {ledger_path}", flush=True)
