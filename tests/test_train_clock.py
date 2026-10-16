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
