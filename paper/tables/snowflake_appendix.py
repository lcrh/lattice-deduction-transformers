"""Snowflake appendix tables: order transfer by test order (manifest
`snowflake_extreme_ood`) and training-set size (manifest `snowflake_extreme`).
Cells: accuracy / wrong answers (%), mean over seeds; per-order cells come from
the per-puzzle rows, which record each puzzle's order.

    uv run python -m paper.tables.snowflake_appendix [--copy-to DIR]
"""

from __future__ import annotations

from paper.tables.common import Table, by_entry, cli, done, load_rows, read_results, write_output

METHODS = (("LDT", ""), ("LDT, one shot", "_one_shot"), ("Direct prediction", None))
TRANSFER = (("4--6", "4to6", (7, 8, 9, 10)), ("6--8", "6to8", (9, 10)))
SIZES = (("1,000", "ldt_n_1000", "direct_n_1000"), ("2,000", "ldt", "direct"),
         ("2,000, 16K steps", "ldt_16k", "direct_16k"), ("all 3,957", "ldt_n_4000", "direct_n_4000"))


def cell(recs: list[dict], order: int | None = None) -> str:
    """Mean over seeds of accuracy / wrong (%), on all rows or on one order."""
    acc, wrong = [], []
    for rec in done(recs):
        rows = [r for r in load_rows(rec) if order is None or r["order"] == order]
        acc.append(sum(r["status"] == "correct" for r in rows) / len(rows))
        wrong.append(sum(r["status"] == "wrong" for r in rows) / len(rows))
    if not acc:
        return "--"
    return f"{100 * sum(acc) / len(acc):.2f} / {100 * sum(wrong) / len(wrong):.2f}"


def transfer_table(records: list[dict]) -> Table:
    entries = by_entry(records)
    body = []
    for train, tag, orders in TRANSFER:
        for i, (method, suffix) in enumerate(METHODS):
            name = f"direct_{tag}" if suffix is None else f"ldt_{tag}{suffix}"
            cells = [cell(entries[name], o) if o in orders else "---" for o in (7, 8, 9, 10)]
            body.append([train if i == 0 else "", method, *cells, cell(entries[name])])
    return Table(columns=["Train", "Method", "7", "8", "9", "10", "All"], rows=body,
                 label="tab:abl-ood",
                 caption="Snowflake Sudoku order transfer (hard puzzles): accuracy / wrong answers "
                         "(\\%) on test puzzles of orders not seen in training, mean over 3 seeds.")


def size_table(records: list[dict]) -> Table:
    entries = by_entry(records)
    body = [[size, cell(entries[ldt]), cell(entries[f"{ldt}_one_shot"]), cell(entries[direct])]
            for size, ldt, direct in SIZES]
    return Table(columns=["Training puzzles", "LDT", "LDT, one shot", "Direct prediction"],
                 rows=body, label="tab:snowflake-data",
                 caption="Snowflake Sudoku training set size: accuracy / wrong answers (\\%) on hard "
                         "test puzzles of orders 4 to 8, mean over 3 seeds.")


def main() -> None:
    args = cli("Snowflake appendix tables.", default_manifest="snowflake_extreme")
    for name, table in (
            ("snowflake_transfer.tex", transfer_table(read_results("snowflake_extreme_ood", args.results))),
            ("snowflake_size.tex", size_table(read_results("snowflake_extreme", args.results)))):
        print(f"wrote {write_output(name, table.to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
