"""Snowflake Sudoku figure: puzzles of orders 3, 4 and 5, and the order-5
puzzle in the covering grid, as TikZ files.

Hexagons are flat-topped at axial coordinates (q, r); a cell is one of the six
triangles of a hexagon, named by its direction. Constraint groups are the six
cells of each hexagon and the six cells around each point where three hexagons
meet. All puzzles are minimal, have a unique solution, and are solved by naked
and hidden singles alone.

The order-4 and order-5 layouts are those of puzzles 056-4-14 and 0OQ-5-14 of
the Snowflake generator, with givens re-minimized for singles; the order-3
puzzle is built by hand.

    uv run python -m paper.figures.snowflake_puzzles [--copy-to DIR]
"""

from __future__ import annotations

from math import cos, pi, radians, sin, sqrt

from paper.tables.common import cli, write_output

DIRS = ["E", "NE", "NW", "W", "SW", "SE"]
COVER = range(-2, 3)          # covering grid: q, r in [-2, 2]
UNIT = 0.69                   # cm per hexagon radius, shared by all puzzles

# Givens per hexagon, as {direction: digit}; omitted cells are blank.
ORDER3 = {(0, 0): {"E": 2, "NW": 3, "SW": 1, "SE": 5},
          (1, 0): {"E": 6, "NE": 3, "W": 2, "SE": 5},
          (0, 1): {"E": 1, "NE": 5, "NW": 3, "SW": 6}}
ORDER4 = {(0, 0): {"NE": 3, "E": 4},
          (-1, 0): {"SE": 1, "SW": 2},
          (-1, 1): {"E": 3, "SE": 4, "SW": 6, "W": 1},
          (0, -1): {"NE": 6, "SE": 4, "SW": 1, "W": 2}}
ORDER5 = {(-1, 1): {"SE": 2, "SW": 3, "W": 6},
          (0, 0): {"NE": 4},
          (-1, 0): {"SE": 1, "SW": 2},
          (0, 1): {"E": 6, "SE": 1, "SW": 5, "NW": 3},
          (0, -1): {"NE": 1, "E": 6, "W": 3, "NW": 4}}


def centre(q, r):
    return 1.5 * q, -sqrt(3) * (r + q / 2)


def corner(q, r, k):
    """Corner k of a flat-topped hexagon, at angle 60k degrees."""
    x, y = centre(q, r)
    return x + cos(radians(60 * k)), y + sin(radians(60 * k))


def triangle(q, r, d):
    """Corners of cell `d`; its outer edge spans corners at its angle -/+ 30 degrees."""
    k = DIRS.index(d)
    a = radians(60 * k - 90)
    x, y = centre(q, r)
    return [(x, y)] + [(x + cos(a + s), y + sin(a + s)) for s in (-pi / 6, pi / 6)]


def meeting_points(hexes):
    """Points shared by three hexagons of the layout."""
    count = {}
    for q, r in hexes:
        for k in range(6):
            p = tuple(round(c, 4) for c in corner(q, r, k))
            count[p] = count.get(p, 0) + 1
    return [p for p, n in count.items() if n == 3]


def pt(p):
    return f"({p[0]:.3f},{p[1]:.3f})"


def puzzle_tikz(givens):
    hexes = list(givens)
    out = [f"\\begin{{tikzpicture}}[x={UNIT}cm, y={UNIT}cm, font=\\footnotesize]"]
    for p in meeting_points(hexes):
        ring = [(p[0] + cos(radians(60 * k)), p[1] + sin(radians(60 * k))) for k in range(6)]
        out.append(f"\\fill[blue!25, fill opacity=0.6] {'--'.join(map(pt, ring))}--cycle;")
        out.append(f"\\draw[blue!60, dashed] {'--'.join(map(pt, ring))}--cycle;")
    for q, r in hexes:
        c = pt(centre(q, r))
        for k in range(6):
            out.append(f"\\draw[gray!60] {c}--{pt(corner(q, r, k))};")
        out.append(f"\\draw[thick] {'--'.join(pt(corner(q, r, k)) for k in range(6))}--cycle;")
        for d, v in givens[(q, r)].items():
            t = triangle(q, r, d)
            g = (sum(x for x, _ in t) / 3, sum(y for _, y in t) / 3)
            out.append(f"\\node at {pt(g)} {{{v}}};")
    out.append("\\end{tikzpicture}")
    return "\n".join(out) + "\n"


def cover_tikz(hexes):
    out = ["\\begin{tikzpicture}"]
    for q in COVER:
        for r in COVER:
            used = (q, r) in hexes
            style = "fill=gray!30, draw=black, thick" if used else "draw=gray!70"
            out.append(f"\\path[{style}] {'--'.join(pt(corner(q, r, k)) for k in range(6))}--cycle;")
    out.append("\\end{tikzpicture}")
    return "\n".join(out) + "\n"


def main() -> None:
    args = cli("Snowflake Sudoku puzzle figure.", default_manifest="snowflake_extreme")
    for name, tikz in (("snowflake_order3.tikz", puzzle_tikz(ORDER3)),
                       ("snowflake_order4.tikz", puzzle_tikz(ORDER4)),
                       ("snowflake_order5.tikz", puzzle_tikz(ORDER5)),
                       ("snowflake_cover.tikz", cover_tikz(set(ORDER5)))):
        print(f"wrote {write_output(name, tikz, args.out, args.copy_to)}")


if __name__ == "__main__":
    main()
