"""Appendix tables: ablations, search policies, deduction and latent carry.

One row per entry: accuracy (mean +- std over seeds), the paired difference
from the reference entry (same seed, same puzzles, pooled over seeds),
calls/solve, wrong answers, timeouts and the unsound-elimination rate. The
manifest's `table` hints name the reference entry, optional row groups, and
for Snowflake order transfer the per-order columns.

    uv run python -m paper.tables.appendix ablation_recurrence [--copy-to DIR]
"""

from __future__ import annotations

from pathlib import Path

from paper.manifest import REPO, TableHints, table_hints
from paper.tables.common import (
    Table, by_entry, cli, delta_text, done, escape, eval_rule, load_rows, mean, mean_std, num,
    paired_vs, pct, pm, read_results, seeds_cell, write_output,
)


def _per_order(records: list[dict], orders: list[int], root: Path) -> list[str]:
    cells = []
    for order in orders:
        accs = []
        for rec in records:
            rows = [r for r in load_rows(rec, root) if r.get("order") == order]
            if rows:
                accs.append(sum(r["status"] == "correct" for r in rows) / len(rows))
        cells.append(pm(*mean_std(accs)))
    return cells


def build(manifest: str, records: list[dict], hints: TableHints, root: Path = REPO,
          label: str | None = None) -> Table:
    finished = done(records)
    entries = by_entry(finished)
    seeds = {name: seeds_cell(recs) for name, recs in by_entry(records).items()}
    reference = hints.get("reference")
    orders = hints.get("per_order")
    ref_records = entries.get(reference, [])
    groups = hints.get("groups") or {}

    columns = ["Configuration", "Seeds", "Acc. (\\%)"]
    if reference:
        columns.append("$\\Delta$ vs.\\ ref. (pp)")
    if orders:
        columns += [f"Order {o}" for o in orders]
    columns += ["Calls/solve", "Wrong", "Timeouts", "Unsound (\\%)"]

    def row(name: str, recs: list[dict]) -> list[str]:
        cells = [escape(recs[0]["label"]), seeds[name],
                 pm(*mean_std([r["accuracy"] for r in recs]))]
        if reference:
            cells.append("(ref.)" if name == reference or not ref_records
                         else delta_text(paired_vs(ref_records, recs, root)))
        if orders:
            cells += _per_order(recs, orders, root)
        cells += [num(mean([r["calls_per_solve"] for r in recs])),
                  num(sum(r["wrong"] for r in recs)), num(sum(r["timeout"] for r in recs)),
                  pct(mean([r.get("unsound_rate") for r in recs]), digits=2)]
        return cells

    body: list[list[str]] = []
    ungrouped = [n for n, recs in entries.items() if not recs[0].get("group")]
    body += [row(n, entries[n]) for n in ungrouped]
    for group, title in groups.items():
        names = [n for n, recs in entries.items() if recs[0].get("group") == group]
        if names:
            body.append([escape(title)])
            body += [row(n, entries[n]) for n in names]
    leftover = [n for n, recs in entries.items()
                if recs[0].get("group") and recs[0]["group"] not in groups]
    body += [row(n, entries[n]) for n in leftover]

    caption = (f"{escape(hints.get('caption', manifest.replace('_', ' ')))} "
               f"{eval_rule(finished)}. Accuracy is mean $\\pm$ std over seeds"
               + ("; $\\Delta$ is paired by puzzle against the reference row, pooled over "
                  "seeds, with a 95\\% interval" if reference else "") + ".")
    return Table(columns=columns, rows=body, caption=caption, label=label or f"tab:{manifest}")


def main() -> None:
    args = cli("Appendix ablation table from collected results.")
    table = build(args.manifest, read_results(args.manifest, args.results),
                  table_hints(args.manifest))
    path = write_output(f"{args.manifest}.tex", table.to_latex(), args.out, args.copy_to)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
