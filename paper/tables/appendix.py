"""Appendix ablation tables (manifests appendix_a ... appendix_h).

One row per entry: accuracy (mean +- std over seeds), the paired difference
from the reference entry (same seed, same puzzles, pooled over seeds),
calls/solve, wrong answers, timeouts and the unsound-elimination rate. The
manifest's `table` hints name the reference entry, optional row groups, and
for Snowflake order transfer the per-order columns.

    uv run python -m paper.tables.appendix appendix_a_recurrence [--copy-to DIR]
"""

from __future__ import annotations

from pathlib import Path

from paper.manifest import REPO, load_manifest
from paper.tables.common import (
    Table, by_entry, cli, delta_text, done, escape, eval_rule, load_rows, mean_std, num, paired_vs,
    pm, read_results, write_output,
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


def build(manifest: str, records: list[dict], hints: dict, root: Path = REPO) -> Table:
    entries = by_entry(done(records))
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
        cells = [escape(recs[0]["label"]), str(len(recs)),
                 pm(*mean_std([r["accuracy"] for r in recs]))]
        if reference:
            cells.append("(ref.)" if name == reference or not ref_records
                         else delta_text(paired_vs(ref_records, recs, root)))
        if orders:
            cells += _per_order(recs, orders, root)
        unsound = mean_std([r.get("unsound_rate") for r in recs])[0]
        cells += [num(mean_std([r["calls_mean"] for r in recs])[0]),
                  num(sum(r["wrong"] for r in recs)), num(sum(r["timeout"] for r in recs)),
                  "--" if unsound != unsound else f"{100 * unsound:.2f}"]
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

    all_done = done(records)
    caption = (f"{escape(hints.get('caption', manifest.replace('_', ' ')))} "
               f"{eval_rule(all_done)}. Accuracy is mean $\\pm$ std over seeds"
               + ("; $\\Delta$ is paired by puzzle against the reference row, pooled over "
                  "seeds, with a 95\\% interval" if reference else "") + ".")
    return Table(columns=columns, rows=body, caption=caption, label=f"tab:{manifest}")


def main() -> None:
    args = cli("Appendix ablation table from collected results.")
    m, _ = load_manifest(args.manifest)
    table = build(args.manifest, read_results(args.manifest, args.results), m.get("table") or {})
    path = write_output(f"{args.manifest}.tex", table.to_latex(), args.out, args.copy_to)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
