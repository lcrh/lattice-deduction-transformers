# Equivalence fixtures

`fixtures/*.pt` hold reference outputs captured on CPU from the code at
commit 28dbc3b (the shared evaluator, before the config refactor):

- `dpll.pt`: `dpll_step` outputs for Sudoku-, Snowflake- and Maze-shaped
  inputs, with and without per-step augmentation, from a seeded tiny model
  in train mode (so eval dropout consumes RNG).
- `solve.pt`: full `solve()` results on six near-complete synthetic Sudokus.
- `train.pt`: model weights after three training steps of each benchmark's
  trainer on synthetic data (`train_runs.py`).

`test_equivalence.py` reruns the same inputs through the current code and
requires identical tensors. Changes that are meant to alter these outputs
must say so and recapture the fixtures.
