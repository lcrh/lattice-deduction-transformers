# E7 — Maze-Hard soundness: why suboptimal paths get returned, and verification

**Question.** On Maze-Hard, the few LDT failures are not abstentions —
the model *returns* a valid start→goal path of slightly suboptimal length
(7/1000 at K=1, 1/1000 at K=512). Sudoku failures, by contrast, are
timeouts, never wrong answers. Why does the conflict machinery miss
suboptimality, and what restores empirical soundness?

**Hypothesis to test.** The two conflict detectors are structurally blind
to suboptimality: (a) the *empty-cell test* can't fire, because a
suboptimal path is a perfectly consistent per-cell assignment — the
lattice encodes "on-path/off-path per cell", not path length; (b) the
*CLS head* is trained on states inconsistent with the sampled shortest
paths, and a suboptimal path may stay consistent-looking until very late
(or the CLS fires below θ_CLS = 0.53). Which of (a)/(b) dominates is
measurable.

**Design.** Three parts, almost all analysis/eval on existing Maze-Hard
checkpoints (K=1 and K=512 runs):

1. **Failure forensics.** What actually happens, round by round, on the
   solves that return a wrong path? For each wrong-returned puzzle: replay
   the winning chain, log the CLS sigmoid per round, and locate the first
   round where the state became inconsistent with *every* optimal path
   (BFS gives ground truth). Report: CLS trajectory vs. that
   commitment point, how far below threshold the CLS peaked, and whether
   suboptimality was decided by a branch or by an (unsound) deduction.
   Also: are wrong paths longer by exactly 2 (one detour) or more?
2. **Cheap-verifier inference.** Does checking the answer before
   accepting it restore soundness at acceptable cost? Add an optional
   accept-time check:
   a returned path is accepted only if it is a valid simple path AND its
   length equals the BFS-optimal length (BFS on a 30×30 maze is
   microseconds — but it uses the *problem definition*, so report it as a
   separate "LDT + verifier" row, not as the base model). Rejected
   accepts → chain reset, search continues. Measures: accuracy /
   abstention with the same round budget; how often the verifier fires.
   A weaker, definition-free variant to include: accept only if ≥2 chains
   independently produce *some* accept in the same round budget and take
   the shortest (self-consistency, no oracle).
3. **Training-side fix (stretch, one run).** Does the training pool ever
   even visit the near-miss states that would teach the conflict head
   about suboptimality? The supervision mechanism exists already: states
   consistent with a valid-but-suboptimal
   path but not with any sampled shortest path already produce
   `gt_conflict=True` under the α machinery; check whether such states
   are *rare in the training pool* (measure their frequency first — if
   ~never visited, that explains the miss, and a pool-seeding fix
   follows naturally). Only train if the measurement supports it.

**Deliverable.** A small table (base / +verifier / +self-consistency:
correct / wrong / abstain), a CLS-trajectory figure for the failure cases,
and the part-1 forensics data (per-failure: first-inconsistent round, CLS
peak, branch-vs-deduction attribution, path-length excess).

**Cost.** Forensics + verifier evals: a few B200-hours of eval (Maze eval
is the expensive one — fan out with the existing
`experiments/maze/eval_only.py` worker pattern). Part 3 adds one ~10
B200-h training run only if justified.

## TODO(worker)

- [ ] `experiments/maze/eval_only.py`: per-round CLS logging for a
      designated puzzle subset (the known failures) — needs a
      trajectory-capture mode in `solve()` keyed off the existing
      `log_per_round_fill` machinery.
- [ ] Failure replayer: identify wrong puzzles from existing
      `.eval.jsonl`, re-run each solo with trajectory capture, compute
      first-inconsistent-round against the all-shortest-paths DAG
      (reuse `lattice_diffusion/data/maze_hard.py` BFS/DAG utilities).
- [ ] Verifier hook in maze eval: `--verify {none,bfs_optimal,self_consistency}`;
      on reject, treat as conflict (chain reset), keep round budget fixed.
- [ ] Pool-frequency measurement for part 3: during a short training run
      (or from a saved pool snapshot), count states consistent with a
      suboptimal-but-valid path and inconsistent with all sampled optimal
      paths.
- [ ] Table + figure script.
