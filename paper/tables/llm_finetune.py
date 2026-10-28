"""Appendix I: fine-tuned Qwen3.5-0.8B, pass@k at each run's final evaluated epoch.

    uv run python -m paper.tables.llm_finetune [--copy-to DIR]
"""

from __future__ import annotations

import json

from paper.manifest import REPO
from paper.tables.common import Table, cli, escape, pct, read_results, write_output

PASS_KS = (1, 8, 32)


def build(records: list[dict]) -> Table:
    body = []
    for rec in records:
        if rec.get("status") != "done" or not rec.get("result_path"):
            continue
        summaries = json.loads((REPO / rec["result_path"]).read_text())
        last = max(summaries, key=lambda s: s["epoch"])
        body.append([escape(rec["label"]), f"{last['epoch']:g}",
                     *(pct(last.get(f"pass_at_{k}")) for k in PASS_KS)])
    return Table(
        columns=["Training puzzles", "Epoch", *(f"pass@{k} (\\%)" for k in PASS_KS)],
        rows=body, label="tab:llm-finetune",
        caption=("Qwen3.5-0.8B fully fine-tuned on 1,000 Sudoku-Extreme training puzzles, evaluated "
                 "on the first 256 puzzles of the Sudoku-Extreme test ordering (or their "
                 "$k$-blank versions), 32 samples per puzzle; unbiased pass@$k$."))


def main() -> None:
    args = cli("Appendix I (fine-tuned LLM).", default_manifest="llm_finetune")
    table = build(read_results(args.manifest, args.results))
    print(f"wrote {write_output('llm_finetune.tex', table.to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
