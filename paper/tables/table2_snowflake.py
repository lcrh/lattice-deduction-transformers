"""Table 2: Snowflake Sudoku accuracy by order (varied-topology generator).

    uv run python -m paper.tables.table2_snowflake [--copy-to DIR]
"""

from __future__ import annotations

from paper.manifest import load_manifest
from paper.tables.appendix import build
from paper.tables.common import cli, read_results, write_output


def main() -> None:
    args = cli("Table 2 (Snowflake Sudoku).", default_manifest="table2_snowflake")
    m, _ = load_manifest(args.manifest)
    hints = m.get("table") or {}
    table = build(args.manifest, read_results(args.manifest, args.results),
                  {"caption": hints.get("caption", ""), "per_order": hints.get("orders")})
    table.label = "tab:snowflake"
    print(f"wrote {write_output('table2_snowflake.tex', table.to_latex(), args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
