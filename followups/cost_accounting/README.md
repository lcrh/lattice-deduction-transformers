# E8 — Normalized cost accounting

**Question.** Published training-cost numbers for LDT, Sotaku, TRM, and
HRM mix hardware generations (B200 / H100 / L40S wall-clock, each as
reported by its authors), which conflates method efficiency with GPU
generation. Restate every comparison in common units, and account for the
offline costs the wall-clock numbers don't show.

**This is mostly bookkeeping, and partially done.** The
[`../../repro/`](../../repro/) study already re-trained HRM and TRM on
Maze-Hard on the *same* B200 hardware and measured steady-state
throughput (TRM ~1.46 s/step, HRM ~0.61 s/step at batch 768 — see
`repro/README.md`), which is exactly the same-silicon datapoint the
Maze-Hard comparison needs. What remains:

1. **GPU-hours normalization.** One table: model × task → reported
   wall-clock, hardware, and converted B200-equivalent GPU-hours. For
   TRM/HRM Maze-Hard use the measured repro throughput directly (steps ×
   s/step), not a hardware conversion factor. For Sudoku TRM (reported:
   4× L40S, 36h) either run the same measurement (TRM sudoku training is cheap to
   sample for 100 steps — measure s/step on B200, don't retrain to
   convergence) or clearly label the row as author-reported.
2. **FLOPs estimates.** Analytic forward+backward FLOPs per training step
   for LDT / TRM / HRM / Sotaku from architecture constants (params,
   seq-len, unroll counts), × steps. Cross-check the LDT estimate against
   measured wall-clock and B200 peak throughput (report achieved MFU as a
   sanity number). FLOPs is the hardware-independent column; GPU-hours is
   the reader-friendly one; report both.
3. **Offline α-operator / data costs.** The wall-clock tables start at
   training; the pipeline doesn't. Measure and report:
   - Maze K-path sampling (`experiments/maze/pregen.py`) CPU-time at
     K ∈ {1, 512} for the 1K-puzzle split,
   - Snowflake CVC5 generation cost per order (instrument
     `experiments/snowflake/gen_data.py`; also feeds E4's order 9–10 gen),
   - Sudoku: trivial (unique solutions ship with the dataset) — state it.
   Report alongside: HRM/TRM's own offline costs on the same tasks
   (dataset build + augmentation expansion) so the comparison stays
   symmetric.
4. **Inference cost column.** Per-puzzle inference cost exists only for
   LDT today; add measured per-puzzle inference cost for TRM/HRM from
   the repro checkpoints (forwards × per-forward cost on the same B200) so
   train/test compute trade-off claims are also same-silicon.

**Deliverable.** One normalized cost table (train GPU-h_B200 + PFLOPs +
offline CPU-h + inference cost per puzzle, per model × task):
`results/cost_table.csv` plus a rendered version.

**Cost.** Bookkeeping + a few short measurement jobs (100-step throughput
samples, pregen timing): well under 1 B200-hour + some CPU-hours.

## TODO(worker)

- [ ] `measure_throughput.py`: 100-step s/step samples on B200 for TRM
      (sudoku config) and any missing model×task cell; reuse the
      `repro/*/modal_train.py` harnesses with a step-cap flag.
- [ ] `flops.py`: analytic per-step FLOPs for the four architectures from
      their configs; unit-test against LDT measured wall-clock (MFU in a
      plausible 20–60% band, else the formula is wrong).
- [ ] Instrument `pregen.py` and `gen_data.py` with total CPU-seconds
      reporting (they run on parallel workers — sum worker time, don't
      report wall-clock).
- [ ] `cost_table.py`: assemble `results/cost_table.csv` + a
      markdown/LaTeX table emitter.
- [ ] Keep every author-reported (non-measured) number flagged in a
      `source` column — measured vs. reported must stay distinguishable.
