import torch

from experiments.sudoku.eval_task import is_valid_solution


def _solved_grid() -> str:
    return "".join(str((r * 3 + r // 3 + c) % 9 + 1) for r in range(9) for c in range(9))


def _puzzle(grid: str, givens) -> torch.Tensor:
    x = torch.ones(81, 9)
    for i in givens:
        x[i] = 0
        x[i, int(grid[i]) - 1] = 1
    return x


def test_valid_grid_passes():
    g = _solved_grid()
    assert is_valid_solution(g, _puzzle(g, range(0, 81, 3)))


def test_invalid_grids_fail():
    g = _solved_grid()
    puzzle = _puzzle(g, [0, 40])
    swapped = g[1] + g[0] + g[2:]       # breaks columns
    assert not is_valid_solution(swapped, puzzle)
    assert not is_valid_solution("0" + g[1:], puzzle)
    relabeled = g.translate(str.maketrans("12", "21"))   # valid sudoku, wrong given
    assert not is_valid_solution(relabeled, _puzzle(g, [i for i in range(81) if g[i] == "1"]))


def test_sequential_estimate_flags_reach_the_solver_config():
    from experiments.sudoku.eval_task import build_solve_config, make_eval_spec

    default = build_solve_config(make_eval_spec("ckpt.pt").solve)
    on = build_solve_config(make_eval_spec("ckpt.pt", estimate_sequential=True,
                                           seq_drain_max_rounds=50).solve)
    assert not default.estimate_sequential
    assert on.estimate_sequential and on.seq_drain_max_rounds == 50
