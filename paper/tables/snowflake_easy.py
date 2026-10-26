"""Snowflake Sudoku accuracy by order on the easier dataset (appendix).

    uv run python -m paper.tables.snowflake_easy [--copy-to DIR]
"""

from __future__ import annotations

from paper.manifest import table_hints
from paper.tables.appendix import build
from paper.tables.common import cli, read_results, write_output


def main() -> None:
    args = cli("Snowflake Sudoku, easier dataset.", default_manifest="snowflake_easy")
    table = build(args.manifest, read_results(args.manifest, args.results),
                  table_hints(args.manifest), label="tab:snowflake-easy")
    print(f"wrote {write_output('snowflake_easy.tex', table.to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
