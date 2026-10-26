"""Snowflake Sudoku on hard puzzles: LDT, the same model in one shot, and direct
prediction, in distribution and under order transfer (manifests
`snowflake_extreme` and `snowflake_extreme_ood`). Cells: accuracy / wrong
answers (%), mean over seeds.

    uv run python -m paper.tables.table2_snowflake [--copy-to DIR]
"""

from __future__ import annotations

from paper.tables.common import Table, by_entry, cli, done, read_results, write_output

# (column heading, manifest, entry of each method: LDT, LDT one shot, direct prediction)
COLUMNS = [
    ("In distribution, 1K", "snowflake_extreme", ("ldt_n_1000", "ldt_n_1000_one_shot", "direct_n_1000")),
    ("In distribution, 2K", "snowflake_extreme", ("ldt", "ldt_one_shot", "direct")),
    ("6--8 $\\to$ 9--10", "snowflake_extreme_ood", ("ldt_6to8", "ldt_6to8_one_shot", "direct_6to8")),
    ("4--6 $\\to$ 7--10", "snowflake_extreme_ood", ("ldt_4to6", "ldt_4to6_one_shot", "direct_4to6")),
]
METHODS = ("LDT", "LDT, one shot", "Direct prediction")


def cell(records: list[dict]) -> str:
    recs = done(records)
    if not recs:
        return "--"
    acc = sum(r["accuracy"] for r in recs) / len(recs)
    wrong = sum(r["wrong"] / r["n"] for r in recs) / len(recs)
    return f"{100 * acc:.2f} / {100 * wrong:.2f}"


def build(results: dict[str, list[dict]]) -> Table:
    entries = {m: by_entry(recs) for m, recs in results.items()}
    body = [[method] + [cell(entries[m].get(names[i], [])) for _, m, names in COLUMNS]
            for i, method in enumerate(METHODS)]
    return Table(columns=["Method"] + [c for c, _, _ in COLUMNS], rows=body, label="tab:snowflake",
                 caption=("Snowflake Sudoku, minimal puzzles that singles cannot solve: accuracy / "
                          "wrong answers (\\%), mean over 3 seeds. In distribution: orders 4 to 8, "
                          "1K or 2K training puzzles. Order transfer: training orders $\\to$ test "
                          "orders."))


def main() -> None:
    args = cli("Table 2 (Snowflake Sudoku, hard puzzles).", default_manifest="snowflake_extreme")
    results = {m: read_results(m, args.results) for m in ("snowflake_extreme", "snowflake_extreme_ood")}
    print(f"wrote {write_output('table2_snowflake.tex', build(results).to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
