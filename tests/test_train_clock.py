import time

from lattice_diffusion.training.utils.clock import TrainClock


def test_clock_splits_compile_step_and_monitoring():
    clock = TrainClock()
    for step in range(3):
        clock.tick()
        time.sleep(0.02 if step == 0 else 0.005)
        if step == 1:
            with clock.monitoring():
                time.sleep(0.01)
    t = clock.finish()
    assert t["step1_compile_secs"] >= 0.02
    assert t["monitor_secs"] >= 0.01
    assert abs(t["total_secs"] - t["step1_compile_secs"] - t["monitor_secs"]
               - t["post_compile_secs"]) < 1e-9


def test_resumed_clock_adds_the_prior_legs():
    prior = {"total_secs": 10.0, "step1_compile_secs": 2.0, "monitor_secs": 1.0,
             "post_compile_secs": 7.0}
    clock = TrainClock(prior)
    clock.tick()
    clock.tick()
    t = clock.finish()
    assert all(t[k] >= prior[k] for k in prior)
    assert t["total_secs"] < 11.0


def test_maze_resumes_from_a_snapshot(tmp_path):
    import shutil

    import torch

    from tests.equivalence import train_runs

    first = tmp_path / "first"
    first.mkdir()
    train_runs.maze(first, steps=4, checkpoint_every=2, keep_step_snapshots=True)
    second = tmp_path / "second"
    second.mkdir()
    shutil.copy(first / "maze.step0000002.pt", second / "maze.resume.pt")
    train_runs.maze(second, steps=4, checkpoint_every=2)
    ckpt = torch.load(second / "maze.pt", weights_only=False)
    snap = torch.load(first / "maze.step0000002.pt", weights_only=False)
    assert ckpt["train_wallclock"]["total_secs"] >= snap["train_wallclock"]["total_secs"]
    assert snap["benchmark"] == "maze" and snap["train_hash"] == ckpt["train_hash"]  # evaluable directly
