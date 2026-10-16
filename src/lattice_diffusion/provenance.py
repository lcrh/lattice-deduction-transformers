"""Run provenance: which experiment a Modal run belongs to and which code ran.

Every launched run carries a `Provenance` (experiment ID from the tracker,
e.g. "A1"; an entry key `<exp_id>/<entry>/seed<s>`; the git SHA of the code
it was launched from). It is stored in checkpoints and eval specs but never
hashed, so the same config launched from two commits is still one run.

Launches are recorded in a ledger on the checkpoint volume, one small JSON
file per launch under `ledger/` (separate files rather than one appended
file: concurrent launches would race on a shared file, and volume commits
are last-writer-wins). `read_ledger` merges them in time order.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

LEDGER_DIR = "ledger"                   # on the checkpoint volume
ALLOW_DIRTY_ENV = "LDT_ALLOW_DIRTY"     # set to 1 to launch from a dirty tree (dev only)
_APP_NAME_MAX = 64


@dataclass
class Provenance:
    exp_id: str = ""                    # experiment tracker ID, e.g. "A1"
    entry: str = ""                     # <exp_id>/<entry>/seed<s>[/eval<e>]
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
    """HEAD SHA, refusing a dirty tree (the SHA would not describe the code
    that runs) unless LDT_ALLOW_DIRTY=1, in which case it is marked."""
    sha, dirty = code_version(repo)
    if not dirty:
        return sha
    if os.environ.get(ALLOW_DIRTY_ENV) == "1":
        return f"{sha}-dirty"
    raise RuntimeError(f"working tree has uncommitted changes; commit them before launching "
                       f"(or set {ALLOW_DIRTY_ENV}=1 for a throwaway dev run)")


def entry_key(exp_id: str, entry: str, seed: int, eval_seed: int | None = None) -> str:
    key = f"{exp_id}/{entry}/seed{seed}"
    return key if not eval_seed else f"{key}/eval{eval_seed}"


def app_run_name(exp_id: str, entry: str, train_hash: str, eval_hash: str | None = None) -> str:
    """`ldt-<exp_id>-<entry>-<trainhash8>[-<evalhash8>]`, lowercased, with
    anything outside [a-z0-9-] replaced, and the entry part shortened so the
    hashes always survive the length limit."""
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
    kind: str                           # "train" | "train+eval" | "eval"
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

    def filename(self) -> str:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(self.ts))
        return f"{LEDGER_DIR}/{stamp}-{uuid.uuid4().hex[:8]}.json"


def app_url(app) -> str | None:
    """Dashboard URL of a running app. Modal keeps the backend-provided URL on
    a private attribute; fall back to the id-only form, which redirects."""
    running = getattr(app, "_running_app", None)
    url = getattr(running, "app_page_url", None)
    if url:
        return url
    return f"https://modal.com/apps/{app.app_id}" if app.app_id else None


def write_ledger(volume, record: LedgerRecord) -> str:
    """Upload one ledger file to a modal.Volume from the local client."""
    import io

    path = record.filename()
    with volume.batch_upload() as batch:
        batch.put_file(io.BytesIO(json.dumps(asdict(record)).encode()), path)
    return path


def read_ledger(list_files: Callable[[str], Iterable[str]],
                read_file: Callable[[str], bytes]) -> list[LedgerRecord]:
    """All ledger records, oldest first. `list_files(dir)` yields file names
    under `dir`; `read_file(path)` returns its bytes (volume-relative paths)."""
    records = []
    for name in sorted(list_files(LEDGER_DIR)):
        if name.endswith(".json"):
            records.append(LedgerRecord(**json.loads(read_file(f"{LEDGER_DIR}/{name}"))))
    return sorted(records, key=lambda r: r.ts)


def announce(record: LedgerRecord, ledger_path: str) -> None:
    print(header("launch", exp_id=record.exp_id, entry=record.entry, app=record.app_name,
                 app_id=record.app_id, call=record.call_id), flush=True)
    print(f"  dashboard: {record.app_url}\n  ledger: {ledger_path}", flush=True)
