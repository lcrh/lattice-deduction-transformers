"""Timed snowflake generator wrapper for offline cost accounting.

Same generation as ``experiments/snowflake/gen_data.py``, plus
CPU timing summed across workers for the cost table.

Original: Modal-parallel snowflake puzzle generator.

Uses the `snowflake` package from https://github.com/santolucito/snowflakesudoku
(MIT-licensed) for hex-topology construction; CVC5 for SAT solving.

100 CPU workers each generate ~300 puzzles (5 n-values × 10 base × 6 variants),
write a parquet shard to the `lattice-diffusion-data` volume at
`/data/snowflake_shards/shard_{id:03d}.parquet`. A consolidation function
then reads all shards, deterministically reorders by (n, code), assigns
final ids, and splits 90/10 into train/test parquet files.

Schema (per row):
  - id        : int64
  - n         : int32   (topology size, in [n_min, n_max])
  - code      : string  (3-char base puzzle id, optionally suffixed -nXX-Gv{var_idx})
  - puzzle    : list<int8>  (per-cell input: 1-6 = given digit, 7 = blank)
  - solution  : list<int8>  (per-cell GT digit, 1-6)
  - givens    : int32   (count of non-7 entries in `puzzle`)
  - topology  : string  (JSON-encoded {n_cells, hex_coords, cell_positions, constraints})

Usage:
    uv run modal run --detach followups/cost_accounting/timed_snowflake_gen.py
    uv run modal run --detach followups/cost_accounting/timed_snowflake_gen.py --n-shards 100 --count-per-shard 10
    # E4 order-transfer OOD: extend the generated range up to order 10. This
    # regenerates the SINGLE consolidated snowflake_train/test.parquet that
    # run.py reads, so orders 9-10 become available to both training (via
    # --train-orders) and eval (--eval-orders). run.py has no separate-file
    # data path, so generate into the main split rather than a side file.
    uv run modal run --detach followups/cost_accounting/timed_snowflake_gen.py --n-max 10
"""

from __future__ import annotations

import json
import random
import string
import sys
import time
from pathlib import Path

import modal


# Build a CPU-only image with cvc5 + snowflakesudoku helper repo.
SNOWFLAKE_REPO_DIR = "/opt/snowflakesudoku"
gen_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("cvc5==1.3.3", "pyarrow==23.0.1", "numpy>=1.24")
    .run_commands(
        f"git clone https://github.com/santolucito/snowflakesudoku {SNOWFLAKE_REPO_DIR}",
    )
)

data_volume = modal.Volume.from_name("lattice-diffusion-data", create_if_missing=True)
DATA_MOUNT = "/data"

app = modal.App("cost-accounting-snowflake-gen")

SHARDS_DIR = "snowflake_shards"  # subdir on the volume


@app.function(image=gen_image, cpu=1.0, timeout=3600,
              volumes={DATA_MOUNT: data_volume})
def gen_shard(
    shard_id: int,
    n_min: int = 4,
    n_max: int = 8,
    count_per_shard: int = 10,
    variants: int = 6,
    seed_base: int = 42,
) -> dict:
    """Generate one parquet shard. Returns a small status dict for logging."""
    import cvc5
    from cvc5 import Kind
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, SNOWFLAKE_REPO_DIR)
    from snowflake.parametric_topology import (
        build_snowflake, HEX_COORDS_BY_N, get_cell_positions,
    )

    # ---- inline snowflake puzzle generation ----
    def _new_solver(rand_seed: int | None = None):
        tm = cvc5.TermManager()
        s = cvc5.Solver(tm)
        s.setLogic("QF_LIA")
        s.setOption("produce-models", "true")
        # Without this, cvc5 returns the same deterministic model for the
        # same constraint set on every call — meaning every shard's
        # `solve_full(...)` for a given n returns identical solutions, and
        # the dataset's only true variation is the minimal-puzzle removal
        # order. We pass a per-call random seed to actually diversify.
        if rand_seed is not None:
            s.setOption("seed", str(rand_seed))
            s.setOption("sat-random-seed", str(rand_seed))
        return tm, s

    def _build_base_constraints(tm, solver, n_cells, constraints):
        int_sort = tm.getIntegerSort()
        cells = [tm.mkConst(int_sort, f"cell{i}") for i in range(n_cells)]
        one = tm.mkInteger(1)
        six = tm.mkInteger(6)
        for c in cells:
            solver.assertFormula(tm.mkTerm(Kind.GEQ, c, one))
            solver.assertFormula(tm.mkTerm(Kind.LEQ, c, six))
        for group in constraints:
            terms = [cells[i] for i in group]
            if len(terms) >= 2:
                solver.assertFormula(tm.mkTerm(Kind.DISTINCT, *terms))
        return cells

    def solve_full(n_cells, constraints, rand_seed=None):
        tm, s = _new_solver(rand_seed=rand_seed)
        cells = _build_base_constraints(tm, s, n_cells, constraints)
        if not s.checkSat().isSat():
            return None
        return [int(s.getValue(c).getIntegerValue()) for c in cells]

    def is_unique(solution, given_set, n_cells, constraints):
        # Uniqueness check: deterministic is fine — we only care about SAT/UNSAT
        # of "does another solution exist?", not which one cvc5 finds.
        tm, s = _new_solver()
        cells = _build_base_constraints(tm, s, n_cells, constraints)
        for i in given_set:
            s.assertFormula(tm.mkTerm(Kind.EQUAL, cells[i], tm.mkInteger(solution[i])))
        # Disallow the known solution: at least one cell differs.
        diff_terms = [
            tm.mkTerm(Kind.NOT, tm.mkTerm(Kind.EQUAL, cells[i], tm.mkInteger(solution[i])))
            for i in range(n_cells) if i not in given_set
        ]
        if not diff_terms:
            return True
        s.assertFormula(tm.mkTerm(Kind.OR, *diff_terms) if len(diff_terms) > 1 else diff_terms[0])
        return not s.checkSat().isSat()

    def _random_distinct_solution(n_cells, constraints, rng):
        # Pass a fresh per-call random seed to cvc5 so each shard produces
        # genuinely different solutions for the same topology. Without this,
        # cvc5 is deterministic on the constraint set and every call returns
        # the same model.
        rand_seed = rng.randint(0, 2**31 - 1)
        return solve_full(n_cells, constraints, rand_seed=rand_seed)

    def minimal_puzzle(solution, n_cells, constraints, rng):
        given = set(range(n_cells))
        order = list(range(n_cells))
        rng.shuffle(order)
        removed = []
        for i in order:
            given.discard(i)
            if is_unique(solution, given, n_cells, constraints):
                removed.append(i)
            else:
                given.add(i)
        return len(given), removed

    def puzzle_variants(solution, n_cells, min_givens, removed_indices, rng, n_variants):
        variants_out = []
        min_puzzle = [solution[i] if i not in removed_indices else 7 for i in range(n_cells)]
        variants_out.append((min_puzzle, min_givens))
        remaining = list(removed_indices)
        rng.shuffle(remaining)
        step = max(1, len(remaining) // 4)
        added_back = 0
        while added_back < len(remaining) and len(variants_out) < n_variants:
            added_back = min(added_back + step, len(remaining))
            still_removed = set(remaining[added_back:])
            puzzle = [solution[i] if i not in still_removed else 7 for i in range(n_cells)]
            givens = sum(1 for v in puzzle if v != 7)
            if givens > min_givens:
                variants_out.append((puzzle, givens))
        return variants_out[:n_variants]

    def three_char_code(rng):
        chars = string.ascii_uppercase + string.digits
        return "".join(rng.choice(chars) for _ in range(3))

    # Pre-cache topology per n. HEX_COORDS_BY_N supports orders up to 19 (the
    # covering grid embeds all of them), so E4's orders 9-10 are in range;
    # fail loudly if a requested order is unsupported by the topology module.
    max_supported = max(HEX_COORDS_BY_N.keys())
    if n_max > max_supported:
        raise ValueError(
            f"n_max={n_max} exceeds HEX_COORDS_BY_N max order {max_supported}"
        )
    topo = {}
    for n in range(n_min, n_max + 1):
        constraints, n_cells = build_snowflake(n)
        topo[n] = {
            "n_cells": n_cells,
            "constraints": constraints,
            "hex_coords": HEX_COORDS_BY_N[n],
            "cell_positions": get_cell_positions(n),
        }

    # Per-shard seed: distinct across shards so puzzles don't duplicate.
    base_seed = seed_base + shard_id * 100_000
    records: list[dict] = []
    t0 = time.time()
    # Offline cost accounting: also track CPU-seconds (user+sys of this
    # process) so `main` can sum TOTAL CPU across parallel workers, not just
    # per-shard wall-clock. Each worker runs at cpu=1.0 so cpu~=wall, but CPU
    # time is the hardware-independent quantity the cost table needs.
    cpu_t0 = time.process_time()
    for n in range(n_min, n_max + 1):
        for base_idx in range(count_per_shard):
            rng = random.Random(base_seed + n * 1000 + base_idx * 100)
            t = topo[n]
            solution = _random_distinct_solution(t["n_cells"], t["constraints"], rng)
            if solution is None:
                continue
            min_givens, removed = minimal_puzzle(solution, t["n_cells"], t["constraints"], rng)
            vlist = puzzle_variants(solution, t["n_cells"], min_givens, removed, rng, variants)
            code = three_char_code(rng)
            for var_idx, (pv, givens) in enumerate(vlist):
                tag = f"{code}-{n}-{givens}" + (f"v{var_idx}" if var_idx > 0 else "")
                records.append({
                    "id": -1,  # assigned in consolidation
                    "n": n,
                    "code": tag,
                    "puzzle": pv,
                    "solution": solution,
                    "givens": givens,
                    "topology": json.dumps({
                        "n_cells": t["n_cells"],
                        "hex_coords": [{"q": q, "r": r} for q, r in t["hex_coords"]],
                        "cell_positions": t["cell_positions"],
                        "constraints": [{"cells": list(c)} for c in t["constraints"]],
                    }),
                })

    elapsed = time.time() - t0
    cpu_secs = time.process_time() - cpu_t0
    # Per-order CPU breakdown is not tracked per-record (CVC5 solve time
    # dominates and grows with n); the (n_min, n_max) span is recorded so the
    # consolidation can attribute cost to the order range this shard covered.
    print(f"shard {shard_id}: generated {len(records)} puzzles in {elapsed:.1f}s "
          f"(cpu={cpu_secs:.1f}s, orders {n_min}-{n_max})", flush=True)

    # Write parquet shard.
    shards_dir = Path(DATA_MOUNT) / SHARDS_DIR
    shards_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "id":       pa.array([r["id"] for r in records], type=pa.int64()),
        "n":        pa.array([r["n"] for r in records], type=pa.int32()),
        "code":     pa.array([r["code"] for r in records], type=pa.string()),
        "puzzle":   pa.array([r["puzzle"] for r in records], type=pa.list_(pa.int8())),
        "solution": pa.array([r["solution"] for r in records], type=pa.list_(pa.int8())),
        "givens":   pa.array([r["givens"] for r in records], type=pa.int32()),
        "topology": pa.array([r["topology"] for r in records], type=pa.string()),
    })
    out_path = shards_dir / f"shard_{shard_id:03d}.parquet"
    pq.write_table(table, str(out_path), compression="zstd")
    data_volume.commit()
    return {"shard_id": shard_id, "n_records": len(records), "elapsed_s": elapsed,
            "cpu_secs": cpu_secs, "n_min": n_min, "n_max": n_max,
            "path": str(out_path)}


@app.function(image=gen_image, cpu=2.0, timeout=600,
              volumes={DATA_MOUNT: data_volume})
def consolidate(test_frac: float = 0.1, seed: int = 0,
                out_suffix: str = "") -> dict:
    """Read all parquet shards, dedupe by (puzzle, solution) fingerprint,
    sort by (n, code), assign final ids, then hash-split disjointly into
    train/test parquet files at
    /data/snowflake_train{out_suffix}.parquet and /data/snowflake_test{out_suffix}.parquet.

    Hash-split: each fingerprint deterministically maps to train or test
    via `sha256(fingerprint)` mod 1000 < test_frac*1000`. This is robust
    to any future duplicate that slips through (it would map to the same
    side both times) and is independent of split-time randomness.
    """
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq

    shards_dir = Path(DATA_MOUNT) / SHARDS_DIR
    shard_files = sorted(shards_dir.glob("shard_*.parquet"))
    print(f"reading {len(shard_files)} shards from {shards_dir}", flush=True)
    if not shard_files:
        raise RuntimeError(f"no shards found in {shards_dir}")

    table = pa.concat_tables([pq.read_table(str(p)) for p in shard_files])
    print(f"  total rows before dedup: {table.num_rows}", flush=True)

    # Dedup by (puzzle bytes, solution bytes). Within a duplicate group we
    # keep the first row encountered after sort-by-(n, code).
    df = table.to_pylist()
    df.sort(key=lambda r: (r["n"], r["code"]))
    seen: set[bytes] = set()
    deduped: list[dict] = []
    for r in df:
        fp = bytes(bytearray(r["puzzle"])) + b"|" + bytes(bytearray(r["solution"]))
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(r)
    n_dropped = len(df) - len(deduped)
    print(f"  dedup: {n_dropped} rows dropped, {len(deduped)} unique fingerprints", flush=True)

    # Hash-based deterministic split. Same fingerprint -> same side, always.
    test_threshold = int(round(test_frac * 1000))
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    salt = f"snowflake-split-seed{seed}".encode()
    for i, r in enumerate(deduped):
        r["id"] = i
        fp = bytes(bytearray(r["puzzle"])) + b"|" + bytes(bytearray(r["solution"]))
        h = hashlib.sha256(salt + fp).digest()
        bucket = int.from_bytes(h[:4], "big") % 1000
        if bucket < test_threshold:
            test_rows.append(r)
        else:
            train_rows.append(r)
    print(f"  hash-split: {len(train_rows)} train / {len(test_rows)} test", flush=True)

    def _to_table(rows):
        return pa.table({
            "id":       pa.array([r["id"] for r in rows], type=pa.int64()),
            "n":        pa.array([r["n"] for r in rows], type=pa.int32()),
            "code":     pa.array([r["code"] for r in rows], type=pa.string()),
            "puzzle":   pa.array([r["puzzle"] for r in rows], type=pa.list_(pa.int8())),
            "solution": pa.array([r["solution"] for r in rows], type=pa.list_(pa.int8())),
            "givens":   pa.array([r["givens"] for r in rows], type=pa.int32()),
            "topology": pa.array([r["topology"] for r in rows], type=pa.string()),
        })

    train_table = _to_table(train_rows)
    test_table = _to_table(test_rows)

    train_path = Path(DATA_MOUNT) / f"snowflake_train{out_suffix}.parquet"
    test_path = Path(DATA_MOUNT) / f"snowflake_test{out_suffix}.parquet"
    pq.write_table(train_table, str(train_path), compression="zstd")
    pq.write_table(test_table, str(test_path), compression="zstd")
    data_volume.commit()

    # Sanity: assert disjoint by fingerprint.
    train_fp = {bytes(bytearray(r["puzzle"])) + b"|" + bytes(bytearray(r["solution"]))
                for r in train_rows}
    test_fp = {bytes(bytearray(r["puzzle"])) + b"|" + bytes(bytearray(r["solution"]))
               for r in test_rows}
    overlap = len(train_fp & test_fp)
    print(f"  sanity: train ∩ test fingerprints = {overlap} (should be 0)", flush=True)
    if overlap != 0:
        raise RuntimeError(f"BUG: dedup+hash-split produced {overlap} overlap")

    return {
        "n_rows_before_dedup": len(df),
        "n_unique": len(deduped),
        "n_dropped_dups": n_dropped,
        "n_train": train_table.num_rows,
        "n_test": test_table.num_rows,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }


@app.local_entrypoint()
def main(
    n_shards: int = 100,
    n_min: int = 4,
    n_max: int = 8,
    count_per_shard: int = 10,
    variants: int = 6,
    seed_base: int = 42,
    test_frac: float = 0.1,
    split_seed: int = 0,
    out_suffix: str = "",
):
    print(f"Spawning {n_shards} shard workers in parallel "
          f"(n in [{n_min}, {n_max}], count_per_shard={count_per_shard}, "
          f"variants={variants}). Each shard targets ~"
          f"{(n_max - n_min + 1) * count_per_shard * variants} puzzles.",
          flush=True)
    args = [
        (i, n_min, n_max, count_per_shard, variants, seed_base)
        for i in range(n_shards)
    ]
    total = 0
    total_cpu_secs = 0.0
    total_wall_secs = 0.0
    for result in gen_shard.starmap(args):
        total += result["n_records"]
        total_cpu_secs += float(result.get("cpu_secs", 0.0))
        total_wall_secs += float(result.get("elapsed_s", 0.0))
        print(f"  shard {result['shard_id']:>3d}: {result['n_records']} puzzles  "
              f"(wall={result['elapsed_s']:.1f}s cpu={result.get('cpu_secs', 0.0):.1f}s)  "
              f"→ {result['path']}",
              flush=True)
    # Offline cost accounting: TOTAL CPU-seconds summed across the parallel
    # workers (not wall-clock). This is the hardware-independent CVC5 generation
    # cost for the cost table. Also feeds E4's order 9-10 generation cost.
    cpu_per_puzzle = total_cpu_secs / max(total, 1)
    print(f"\nGenerated {total} puzzles across {n_shards} shards "
          f"(orders {n_min}-{n_max}).", flush=True)
    print(f"OFFLINE CPU COST (summed across {n_shards} workers): "
          f"{total_cpu_secs:.1f} CPU-s ({total_cpu_secs / 3600:.3f} CPU-h); "
          f"{cpu_per_puzzle * 1000:.2f} CPU-ms/puzzle. Consolidating…",
          flush=True)

    summary = consolidate.remote(
        test_frac=test_frac, seed=split_seed, out_suffix=out_suffix,
    )
    print(f"\nConsolidation: {summary['n_rows_before_dedup']} rows → "
          f"{summary['n_unique']} unique → "
          f"{summary['n_train']} train / {summary['n_test']} test  "
          f"(dropped {summary['n_dropped_dups']} duplicates)",
          flush=True)
    print(f"  train: {summary['train_path']}", flush=True)
    print(f"  test:  {summary['test_path']}", flush=True)
