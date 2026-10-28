"""Training-input mix: each arm of manifest `ablation_training_inputs` against
the Table 1 4K-step models on the same test puzzles and seeds.

The reference answers are those of the same checkpoints in manifest
`portfolio` (entry `same_data_test`, every test puzzle at 1,000 rounds), so
the comparison is paired by puzzle; the interval is a normal 95% interval on
the mean difference in correctness.

    uv run python -m paper.tables.training_inputs [--copy-to DIR]
"""

from __future__ import annotations

from paper.portfolio import TIMEOUT, test_failures
from paper.tables.common import Table, by_entry, cli, done, load_rows, num, read_results, write_output

ARMS = (("Raw and correct hints", "no_error_hints"), ("Raw only", "raw_only"))


def compare(arm: list[dict], reference: dict[int, dict[int, int]]) -> list[str]:
    """Puzzles, paired difference, wrong and timeouts (arm / reference)."""
    diffs, counts = [], {"aw": 0, "rw": 0, "at": 0, "rt": 0}
    for rec in done(arm):
        ref = reference[rec["seed"]]
        for row in load_rows(rec):
            r = ref.get(row["idx"], 0)
            diffs.append(int(row["status"] == "correct") - int(r == 0))
            counts["aw"] += row["status"] == "wrong"
            counts["at"] += row["status"] == "timeout"
            counts["rw"] += r > 0
            counts["rt"] += r == TIMEOUT
    n = len(diffs)
    mean = sum(diffs) / n
    half = 1.96 * (sum((d - mean) ** 2 for d in diffs) / (n - 1) / n) ** 0.5
    return [num(n, 0), f"{100 * mean:+.2f} [{100 * (mean - half):+.2f}, {100 * (mean + half):+.2f}]",
            f"{counts['aw']} / {counts['rw']}", f"{counts['at']} / {counts['rt']}"]


def main() -> None:
    args = cli("Training-input ablation.", default_manifest="ablation_training_inputs")
    entries = by_entry(read_results(args.manifest, args.results))
    reference, _ = test_failures(read_results("portfolio", args.results))
    body = [[label, *compare(entries[name], reference)] for label, name in ARMS]
    table = Table(columns=["Training inputs", "Puzzles", "Delta (pp)", "Wrong", "Timeouts"],
                  rows=body, label="tab:abl-inputs",
                  caption="Training-input mix on Sudoku-Extreme (4K steps, 3 seeds) against the "
                          "Table 1 models on the same test puzzles (arm / reference).")
    print(f"wrote {write_output('training_inputs.tex', table.to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
