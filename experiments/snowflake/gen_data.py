"""Modal-parallel snowflake puzzle generator.

Uses the `snowflake` package from https://github.com/lcrh/snowflakesudoku
(an MIT-licensed fork of santolucito/snowflakesudoku, pinned below) for
hex-topology construction; CVC5 for SAT solving. Every base puzzle gets its
own deterministic, randomly grown connected layout, translated into the
covering grid. Orders 4-10 by default, so order-transfer experiments can hold
out the larger orders.

100 CPU workers each generate a parquet shard (orders × count_per_shard base
puzzles × up to `variants` given-count variants) on the
`ldt-data` volume. A consolidation function reads all shards,
deterministically reorders by (n, code), assigns final ids, and hash-splits
rows into test / val / train (10% / 5% / rest). Files are
`/data/snowflake_<split>.parquet`; each file's parquet metadata records the
generator settings and a content hash.

Schema (per row):
  - id        : int64
  - n         : int32   (topology size, in [n_min, n_max])
  - code      : string  (3-char base puzzle id, optionally suffixed -nXX-Gv{var_idx})
  - puzzle    : list<int8>  (per-cell input: 1-6 = given digit, 7 = blank)
  - solution  : list<int8>  (per-cell GT digit, 1-6)
  - givens    : int32   (count of non-7 entries in `puzzle`)
  - topology  : string  (JSON-encoded {n_cells, hex_coords, cell_positions, constraints})

Usage:
    uv run modal run --detach experiments/snowflake/gen_data.py      # orders 4-10

After generation, the entrypoint prints the index-list entries to pin in
src/ldt/eval/indices.py.
"""

from __future__ import annotations

import hashlib
import json
import random
import string
import sys
import time
from pathlib import Path

import modal


# Build a CPU-only image with cvc5 + the pinned snowflakesudoku fork.
SNOWFLAKE_REPO_DIR = "/opt/snowflakesudoku"
SNOWFLAKE_REPO_URL = "https://github.com/lcrh/snowflakesudoku"
SNOWFLAKE_REPO_COMMIT = "78394dfa8ce77c5429fe411547c866e704a18d68"
gen_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("cvc5==1.3.3", "pyarrow==23.0.1", "numpy>=1.24")
    .run_commands(
        f"git clone {SNOWFLAKE_REPO_URL} {SNOWFLAKE_REPO_DIR} && "
        f"git -C {SNOWFLAKE_REPO_DIR} checkout {SNOWFLAKE_REPO_COMMIT}",
    )
)
# Covering-grid bounds; must match experiments/snowflake/data.py.
COVER_Q_MIN, COVER_Q_MAX = -2, 2
COVER_R_MIN, COVER_R_MAX = -2, 2
SHARDS_DIR = "snowflake_shards"

data_volume = modal.Volume.from_name("ldt-data", create_if_missing=True)
DATA_MOUNT = "/data"

app = modal.App("snowflake-data-gen")


def split_file(split: str) -> str:
    """Mirrors experiments/snowflake/data.py:split_file (this image has no torch)."""
    return f"snowflake_{split}.parquet"


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
        build_snowflake, HEX_COORDS_BY_N, get_cell_positions, get_hex_coords,
        translate_hex_coords,
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

    def grown_topology(n: int, seed: int) -> dict:
        # A grown layout can be too long for the covering grid; resample with
        # derived seeds until one fits. The topology seed is namespaced away
        # from the puzzle RNG seed so geometry and digits are not coupled.
        for attempt in range(64):
            topology_seed = ((seed ^ 0xA5A5A5A5) + attempt * 7919) & 0x7FFFFFFF
            try:
                coords = translate_hex_coords(
                    get_hex_coords(n, topology_seed=topology_seed),
                    q_min=COVER_Q_MIN, q_max=COVER_Q_MAX, r_min=COVER_R_MIN, r_max=COVER_R_MAX)
            except ValueError:
                continue
            constraints, n_cells = build_snowflake(n, coords)
            return {
                "n_cells": n_cells,
                "constraints": constraints,
                "hex_coords": coords,
                "cell_positions": get_cell_positions(n, coords),
            }
        raise RuntimeError(f"no varied layout of order {n} fits the covering grid")

    if n_max > max(HEX_COORDS_BY_N):
        raise ValueError(f"n_max={n_max} exceeds the largest supported order "
                         f"{max(HEX_COORDS_BY_N)}")

    # Per-shard seed: distinct across shards so puzzles don't duplicate.
    base_seed = seed_base + shard_id * 100_000
    records: list[dict] = []
    t0 = time.time()
    for n in range(n_min, n_max + 1):
        for base_idx in range(count_per_shard):
            task_seed = base_seed + n * 1000 + base_idx * 100
            rng = random.Random(task_seed)
            t = grown_topology(n, task_seed)
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
    print(f"shard {shard_id}: generated {len(records)} puzzles in {elapsed:.1f}s",
          flush=True)

    # Write parquet shard.
    out_dir = Path(DATA_MOUNT) / SHARDS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "id":       pa.array([r["id"] for r in records], type=pa.int64()),
        "n":        pa.array([r["n"] for r in records], type=pa.int32()),
        "code":     pa.array([r["code"] for r in records], type=pa.string()),
        "puzzle":   pa.array([r["puzzle"] for r in records], type=pa.list_(pa.int8())),
        "solution": pa.array([r["solution"] for r in records], type=pa.list_(pa.int8())),
        "givens":   pa.array([r["givens"] for r in records], type=pa.int32()),
        "topology": pa.array([r["topology"] for r in records], type=pa.string()),
    })
    out_path = out_dir / f"shard_{shard_id:03d}.parquet"
    pq.write_table(table, str(out_path), compression="zstd")
    data_volume.commit()
    return {"shard_id": shard_id, "n_records": len(records), "elapsed_s": elapsed,
            "path": str(out_path)}


def assign_split(key: bytes, test_frac: float, val_frac: float, seed: int) -> str:
    """Split from a row's split key: test is buckets [0, test_frac), val the next val_frac."""
    salt = f"snowflake-split-seed{seed}".encode()
    bucket = int.from_bytes(hashlib.sha256(salt + key).digest()[:4], "big") % 1000
    test_hi = int(round(test_frac * 1000))
    val_hi = test_hi + int(round(val_frac * 1000))
    return "test" if bucket < test_hi else ("val" if bucket < val_hi else "train")


def fingerprint(row: dict) -> bytes:
    """Dedup key: puzzle, solution and layout."""
    return bytes(bytearray(row["puzzle"])) + b"|" + split_key(row)


def split_key(row: dict) -> bytes:
    """Split key: solution and layout, so every given-count variant of a puzzle shares a split."""
    return bytes(bytearray(row["solution"])) + b"|" + row["topology"].encode()


def content_sha256(rows: list[dict]) -> str:
    """Hash of a split's rows (ids and fingerprints, in file order)."""
    h = hashlib.sha256()
    for r in rows:
        h.update(str(r["id"]).encode() + b":" + fingerprint(r) + b"\n")
    return h.hexdigest()


def split_rows(rows: list[dict], test_frac: float, val_frac: float,
               seed: int) -> tuple[dict[str, list[dict]], int]:
    """Dedup, order by (n, code), assign ids and split; returns (splits, duplicates dropped)."""
    rows = sorted(rows, key=lambda r: (r["n"], r["code"]))
    seen: set[bytes] = set()
    deduped = []
    for r in rows:
        fp = fingerprint(r)
        if fp not in seen:
            seen.add(fp)
            deduped.append(r)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for i, r in enumerate(deduped):
        r["id"] = i
        splits[assign_split(split_key(r), test_frac, val_frac, seed)].append(r)
    return splits, len(rows) - len(deduped)


@app.function(image=gen_image, cpu=2.0, timeout=600,
              volumes={DATA_MOUNT: data_volume})
def consolidate(generator: dict, test_frac: float = 0.1, val_frac: float = 0.05,
                seed: int = 0) -> dict:
    """Merge the shards into one parquet file per split (see split_rows).

    Each file's metadata records the generator settings and a content hash.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    in_dir = Path(DATA_MOUNT) / SHARDS_DIR
    shard_files = sorted(in_dir.glob("shard_*.parquet"))
    print(f"reading {len(shard_files)} shards from {in_dir}", flush=True)
    if not shard_files:
        raise RuntimeError(f"no shards found in {in_dir}")
    rows = pa.concat_tables([pq.read_table(str(p)) for p in shard_files]).to_pylist()
    splits, n_dropped = split_rows(rows, test_frac, val_frac, seed)
    print(f"  {len(rows)} rows, {n_dropped} duplicates dropped; "
          + ", ".join(f"{k}={len(v)}" for k, v in splits.items()), flush=True)

    def _to_table(rows, meta):
        table = pa.table({
            "id":       pa.array([r["id"] for r in rows], type=pa.int64()),
            "n":        pa.array([r["n"] for r in rows], type=pa.int32()),
            "code":     pa.array([r["code"] for r in rows], type=pa.string()),
            "puzzle":   pa.array([r["puzzle"] for r in rows], type=pa.list_(pa.int8())),
            "solution": pa.array([r["solution"] for r in rows], type=pa.list_(pa.int8())),
            "givens":   pa.array([r["givens"] for r in rows], type=pa.int32()),
            "topology": pa.array([r["topology"] for r in rows], type=pa.string()),
        })
        return table.replace_schema_metadata({b"snowflake": json.dumps(meta).encode()})

    out = {"n_rows": len(rows), "n_dropped_dups": n_dropped, "splits": {}}
    for split, split_rows_ in splits.items():
        meta = {**generator, "split": split,
                "split_rule": {"test_frac": test_frac, "val_frac": val_frac, "seed": seed},
                "generator_repo": SNOWFLAKE_REPO_URL, "generator_commit": SNOWFLAKE_REPO_COMMIT,
                "n_rows": len(split_rows_), "content_sha256": content_sha256(split_rows_)}
        path = Path(DATA_MOUNT) / split_file(split)
        pq.write_table(_to_table(split_rows_, meta), str(path), compression="zstd")
        out["splits"][split] = {"path": str(path), "n_rows": len(split_rows_),
                                "content_sha256": meta["content_sha256"]}
    data_volume.commit()
    return out


@app.local_entrypoint()
def main(
    n_shards: int = 100,
    n_min: int = 4,
    n_max: int = 10,
    count_per_shard: int = 10,
    variants: int = 6,
    seed_base: int = 42,
    test_frac: float = 0.1,
    val_frac: float = 0.05,
    split_seed: int = 0,
):
    print(f"Spawning {n_shards} shard workers ("
          f"n in [{n_min}, {n_max}], count_per_shard={count_per_shard}, "
          f"variants={variants}). Each shard targets ~"
          f"{(n_max - n_min + 1) * count_per_shard * variants} puzzles.",
          flush=True)
    args = [
        (i, n_min, n_max, count_per_shard, variants, seed_base)
        for i in range(n_shards)
    ]
    total = 0
    for result in gen_shard.starmap(args):
        total += result["n_records"]
        print(f"  shard {result['shard_id']:>3d}: {result['n_records']} puzzles  "
              f"({result['elapsed_s']:.1f}s)  → {result['path']}",
              flush=True)
    print(f"\nGenerated {total} puzzles across {n_shards} shards. Consolidating…",
          flush=True)

    generator = {"n_shards": n_shards, "n_min": n_min, "n_max": n_max,
                 "count_per_shard": count_per_shard, "variants": variants,
                 "seed_base": seed_base}
    summary = consolidate.remote(generator, test_frac=test_frac, val_frac=val_frac,
                                 seed=split_seed)
    print(f"\nConsolidation: {summary['n_rows']} rows, "
          f"{summary['n_dropped_dups']} duplicates dropped", flush=True)
    for split, info in summary["splits"].items():
        print(f"  {split}: {info['n_rows']} rows → {info['path']}  "
              f"(content sha256 {info['content_sha256'][:16]}…)", flush=True)
    print_index_pins({k: v["n_rows"] for k, v in summary["splits"].items()})


def print_index_pins(sizes: dict[str, int]) -> None:
    """Print the split sizes and checksums to pin in src/ldt/eval/indices.py."""
    from ldt.eval import indices as index_lists

    print("\nPin these in src/ldt/eval/indices.py:", flush=True)
    for split in ("test", "val"):
        name = f"snowflake_{split}"
        spec = index_lists.SPECS.get(name)
        if spec is None:
            print(f"  (no index list named {name})", flush=True)
            continue
        lst = index_lists.load(name, split_size=sizes[split]) if spec.split_size is None else None
        digest = lst.digest if lst is not None else spec.sha256
        print(f"  {name}: split_size={sizes[split]}, sha256=\"{digest}\"", flush=True)
