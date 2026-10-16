"""Manifest resolution, status, launch planning, collect, naming and ledger."""

import csv
import dataclasses
import json
import subprocess
import textwrap

import pytest

from lattice_diffusion import config as cfglib
from lattice_diffusion.provenance import (
    LedgerRecord, app_run_name, entry_key, read_ledger, require_clean_sha,
)
import paper.manifest as manifest
from paper import launch
from paper.manifest import eval_overrides, load_manifest, volume_path
from tests.paper.fakes import FakeVolumeView

from experiments.registry import experiment_class


def _runs(name):
    return load_manifest(name)


def _by_key(runs):
    return {r.key: r for r in runs}


# --- manifest resolution -----------------------------------------------------

def test_shared_configs_share_checkpoints():
    t1 = _by_key(_runs("table1_sudoku"))
    port = _by_key(_runs("portfolio"))
    fig2 = _by_key(_runs("fig2_sudoku_compute"))
    for s in (0, 1, 2):
        ckpt = t1[f"M1/steps4k/seed{s}"].checkpoint
        assert port[f"X1/anchor/seed{s}"].checkpoint == ckpt
        assert fig2[f"M2/steps_4000/seed{s}"].checkpoint == ckpt
        # Same eval spec too, so the full-split blocks are shared.
        assert port[f"X1/anchor/seed{s}"].run_dir == t1[f"M1/steps4k/seed{s}"].run_dir
    assert t1["M1/steps1k/seed0"].checkpoint != t1["M1/steps1k/seed1"].checkpoint


def test_eval_only_entries_reuse_training_checkpoints():
    runs = _by_key(_runs("appendix_d_soundness"))
    for s in (0, 1, 2):
        arm, train = runs[f"A7/sym_matched/seed{s}"], runs[f"A7/sym/seed{s}"]
        assert arm.kind == "eval" and arm.train_key == train.key
        assert arm.checkpoint == train.checkpoint
        assert arm.eval_hash != train.eval_hash
        assert arm.spec.solve["step"]["threshold"] == 0.5


def test_sweep_and_grid_expansion():
    names = {r.entry for r in _runs("appendix_g_search")}
    assert {"s1_2k_uniform_sample", "s1_2k_max_entropy_rank_k", "s3_2k_negate_last"} <= names
    xs = {r.entry: r.x for r in _runs("fig3_maze_ksweep")}
    assert xs["k_512"] == 512


def test_appendix_run_counts():
    counts = {name: sum(r.kind != "eval" for r in _runs(name)) for name in (
        "appendix_a_recurrence", "appendix_b_supervision", "appendix_c_lambda_ce",
        "appendix_d_soundness", "appendix_e_latent_carry", "appendix_g_search")}
    assert counts == {"appendix_a_recurrence": 48, "appendix_b_supervision": 6,
                      "appendix_c_lambda_ce": 9, "appendix_d_soundness": 15,
                      "appendix_e_latent_carry": 12, "appendix_g_search": 9}


def _write_manifest(tmp_path, body):
    (tmp_path / "_eval.yaml").write_text("presets: {ablation: {eval.stop: ablation}}\n")
    (tmp_path / "m.yaml").write_text(textwrap.dedent(body))
    return tmp_path


def test_eval_only_entry_may_not_change_training(tmp_path):
    d = _write_manifest(tmp_path, """
        exp_id: X1
        benchmark: sudoku
        config: configs/sudoku/ablation.yaml
        eval: ablation
        entries:
          - name: base
          - name: bad
            eval_of: base
            set: {train.steps: 10}
        """)
    with pytest.raises(ValueError, match="only eval"):
        load_manifest("m", d)


def test_entries_need_an_exp_id(tmp_path):
    d = _write_manifest(tmp_path, """
        benchmark: sudoku
        entries: [{name: base}]
        """)
    with pytest.raises(ValueError, match="exp_id"):
        load_manifest("m", d)


def test_eval_overrides_reproduce_the_local_spec():
    """The remote spec built from the tokens equals the locally resolved one."""
    runs = _by_key(_runs("appendix_d_soundness"))
    arm = runs["A7/sym_matched/seed1"]
    # Pretend the checkpoint was trained under a different eval block.
    stored = cfglib.to_dict(runs["A7/sym/seed1"].exp)
    stored["eval"]["stop"] = "sequential"
    stored["eval"]["workers"] = 3
    stored["provenance"] = {"exp_id": "OTHER", "entry": "x", "code_sha": "abc"}
    merged = cfglib.merge(stored, cfglib.parse_overrides(eval_overrides(arm)))
    exp = cfglib.from_dict(experiment_class("sudoku"), merged)
    spec = exp.eval_spec(arm.checkpoint)
    assert spec.hashed_fields() == arm.spec.hashed_fields()
    assert spec.run_dir("/checkpoints/eval") == arm.spec.run_dir("/checkpoints/eval")
    assert (spec.exp_id, spec.entry) == ("A7", "A7/sym_matched/seed1")


# --- naming, provenance, ledger -----------------------------------------------

def test_app_names_are_sanitized_and_keep_hashes():
    name = app_run_name("A1", "S1 Policy/2K.uniform_sample" * 4, "0123456789ab", "fedcba987654")
    assert name.startswith("ldt-a1-s1-policy-2k-uniform-sample")
    assert name.endswith("-01234567-fedcba98")
    assert len(name) <= 64
    assert set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")
    assert entry_key("A1", "loops_4", 2) == "A1/loops_4/seed2"
    assert entry_key("X1", "rerun", 3, 2) == "X1/rerun/seed3/eval2"


def test_every_run_name_is_unique_within_a_manifest():
    for name in ("appendix_g_search", "portfolio", "fig3_maze_ksweep"):
        runs = _runs(name)
        assert len({r.key for r in runs}) == len(runs)
        named = [r.app_name() for r in runs if r.kind != "eval" or r.spec is not None]
        assert len(set(named)) == len(named)


def _ledger_record(entry, ts, call="fc-1", kind="train+eval"):
    return LedgerRecord(kind=kind, exp_id=entry.split("/")[0], entry=entry, app_name="n",
                        app_id="ap-1", app_url="u", call_id=call, code_sha="sha",
                        checkpoint="/checkpoints/x.pt", train_hash="h", ts=ts)


def test_ledger_reads_back_in_time_order():
    recs = [_ledger_record("A1/a/seed0", 3.0), _ledger_record("A1/a/seed0", 1.0, call="fc-0")]
    view = FakeVolumeView({f"ledger/{i}.json": json.dumps(dataclasses.asdict(r))
                           for i, r in enumerate(recs)})
    got = launch.load_ledger(view)
    assert [r.ts for r in got] == [1.0, 3.0]
    assert launch.latest_ledger(got)["A1/a/seed0"].call_id == "fc-1"
    assert read_ledger(lambda d: [], lambda p: b"") == []


def test_dirty_tree_is_refused(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f").write_text("a")
    subprocess.run(["git", "add", "f"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
                   cwd=tmp_path, check=True)
    sha = require_clean_sha(tmp_path)
    assert len(sha) == 40
    (tmp_path / "f").write_text("b")
    monkeypatch.delenv("LDT_ALLOW_DIRTY", raising=False)
    with pytest.raises(RuntimeError, match="uncommitted"):
        require_clean_sha(tmp_path)
    monkeypatch.setenv("LDT_ALLOW_DIRTY", "1")
    assert require_clean_sha(tmp_path) == f"{sha}-dirty"


# --- status and launch planning ------------------------------------------------

def _summary(n, correct, wrong=0, timeout=None):
    timeout = n - correct - wrong if timeout is None else timeout
    calls = {"mean": 5.0, "p50": 4.0, "p75": 6.0, "p90": 9.0, "p95": 12.0, "p99": 20.0}
    return json.dumps({
        "n": n, "stop_reason": "fixed_n", "correct": correct, "wrong": wrong,
        "timeout": timeout, "accuracy": correct / n, "accuracy_cs": [0.9, 1.0],
        "wrong_rate_upper": 0.003, "failure_rate": (n - correct) / n,
        "calls_all": calls, "calls_correct": calls, "calls_per_solve": 7.0,
        "soundness": {"unsound_rate": 0.01},
        "gpu_seconds": 10.0, "startup_seconds": 5.0, "wall_seconds": 20.0, "n_evaluated": n,
        "model_calls_total": 100, "mean_rounds_correct": 3.0, "mean_resets": 0.5,
    })


def test_status_of_each_run_state():
    runs = _by_key(_runs("appendix_d_soundness"))
    done_run, trained_run = runs["A0/default/seed0"], runs["A7/sym/seed0"]
    running, failed = runs["A7/ratio2/seed0"], runs["A7/ratio32/seed0"]
    partial = runs["A7/nocls/seed0"]
    files = {
        volume_path(done_run.checkpoint): b"",
        volume_path(done_run.summary_file): _summary(1000, 990),
        volume_path(trained_run.checkpoint): b"",
        volume_path(partial.checkpoint): b"",
        volume_path(partial.run_dir) + "/blocks/000000.jsonl": b"{}\n",
    }
    ledger = [_ledger_record(running.key, 1.0, call="fc-run"), _ledger_record(failed.key, 1.0, call="fc-bad")]
    states = launch.statuses(list(runs.values()), FakeVolumeView(files), ledger,
                        {"fc-run": "running", "fc-bad": "failed"}.get)
    assert states[done_run.key] == "done"
    assert states[trained_run.key] == "trained"
    assert states[running.key] == "running"
    assert states[failed.key] == "failed"
    assert states[partial.key] == "partial"
    assert states["A7/sym_matched/seed0"] == "trained"      # its checkpoint exists
    assert states["A7/ratio2_matched/seed0"] == "blocked"   # training still running
    assert states["A0/default/seed1"] == "not started"
    offline = launch.statuses(list(runs.values()), FakeVolumeView(files), ledger, None)
    assert offline[running.key] == "launched"


def test_plan_trains_missing_runs_and_evaluates_existing_checkpoints():
    runs = _runs("portfolio")
    states = {r.key: "not started" for r in runs}
    plan = {p.run.key: (p.cmd, p.reason) for p in launch.plan_launches(runs, states)}
    cmd, _ = plan["X1/anchor/seed0"]
    assert cmd[:6] == ["uv", "run", "modal", "run", "--detach", "-n"]
    assert "experiments/sudoku/run.py" in cmd and "--no-wait" in cmd
    assert "--no-evaluate" not in cmd
    assert "--no-evaluate" in plan["X1/models/seed7"][0]
    assert plan["X1/rerun_models/seed7"][0] is None          # no failure set yet
    # A checkpoint that exists is only evaluated.
    t1 = _runs("table1_sudoku")
    states = {r.key: "trained" for r in t1}
    planned = {p.run.key: p for p in launch.plan_launches(t1, states)}["M1/steps4k/seed0"]
    cmd, why = planned.cmd, planned.reason
    assert "experiments/evaluate.py" in cmd and why == "evaluate existing checkpoint"
    assert cmd[cmd.index("--checkpoint") + 1].startswith("/checkpoints/sudoku/paper-")


def test_rerun_entries_resolve_once_the_failure_set_exists(monkeypatch):
    monkeypatch.setattr(manifest, "_read_indices", lambda path: [5, 17])
    runs = _by_key(_runs("portfolio"))
    rerun = runs["X1/rerun_models/seed5/eval2"]
    assert rerun.spec.explicit_indices == [5, 17] and rerun.spec.eval_seed == 2
    assert rerun.checkpoint == runs["X1/models/seed5"].checkpoint
    cmd = launch.eval_command(rerun)
    assert cmd[cmd.index("--indices-file") + 1] == "paper/results/portfolio_failures.txt"


# --- collect -----------------------------------------------------------------

def test_collect_writes_csv_and_reported_rows(tmp_path, monkeypatch):
    runs = [r for r in _runs("appendix_b_supervision") if r.seed == 0]
    run = runs[0]
    rows = [{"idx": 100 + p, "pos": p, "status": "correct" if p % 3 else "timeout",
             "calls": p} for p in range(6)]
    files = {
        volume_path(run.checkpoint): b"",
        volume_path(run.train_record): json.dumps({"train_seconds": 420.0}),
        volume_path(run.summary_file): _summary(4, 2, 0, 2),
        volume_path(run.run_dir) + "/blocks/000000.jsonl":
            "".join(json.dumps(r) + "\n" for r in rows),
    }
    view = FakeVolumeView(files)
    monkeypatch.setattr(launch, "REPO", tmp_path)
    states = launch.statuses(runs, view, [], None)
    out = launch.collect("appendix_b_supervision", runs, view, states, tmp_path, with_rows=True)
    with out.open() as fh:
        recs = {r["key"]: r for r in csv.DictReader(fh)}
    rec = recs[run.key]
    assert rec["status"] == "done" and rec["n"] == "4" and rec["train_seconds"] == "420.0"
    assert rec["model_loops"] == "16" and rec["train_steps"] == "2000"
    assert recs[runs[1].key]["status"] == "not started"
    got = [json.loads(line)
           for line in (tmp_path / rec["rows_path"]).read_text().splitlines()]
    assert [r["pos"] for r in got] == [0, 1, 2, 3]
