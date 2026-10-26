"""Summarize results.json: solve rate, output tokens and cost per puzzle, and a
failure category for every unsolved answer.

usage (from the lattice_deduction_transformers repo):
    uv run --with pyarrow python <this file>
"""
import collections
import importlib.util
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

TASKS = {(t["bench"], t["idx"]): t for t in run.sudoku_tasks() + run.snowflake_tasks() + run.maze_tasks()}
BENCHES = ("sudoku", "snowflake", "maze")


def grid_cells(task, text):
    """Digits per cell from a Sudoku or Snowflake answer, or None if malformed."""
    if task["bench"] == "sudoku":
        s = "".join(line.strip() for line in run.lines(text))
        return [int(c) for c in s] if len(s) == 81 and s.isdigit() else None
    got = {}
    for line in run.lines(text):
        if ":" in line:
            c, v = line.strip().split(":", 1)
            if c.isdigit() and v.strip().isdigit():
                got[int(c)] = int(v)
    n = len(task["answer"])
    return [got[c] for c in range(n)] if set(got) == set(range(n)) else None


def sudoku_groups():
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    cols = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = [[(br + r) * 9 + bc + c for r in range(3) for c in range(3)]
             for br in (0, 3, 6) for bc in (0, 3, 6)]
    return rows + cols + boxes


def failure(task, r):
    if r.get("text") is None:
        return "no response (error)"
    if r.get("finish") == "length":
        return "hit the output-token cap"
    if task["bench"] in ("sudoku", "snowflake"):
        cells = grid_cells(task, r["text"])
        if cells is None:
            return "malformed output"
        if task["bench"] == "sudoku":
            q = task["user"].split("\n", 1)[1].replace("\n", "")
            givens = [int(c) for c in q]
            groups = sudoku_groups()
        else:
            givens = [0 if v == 7 else v for v in task["puzzle"]]
            groups = task["groups"]
        if any(g and g != c for g, c in zip(givens, cells)):
            return "changed a given"
        if any(len({cells[i] for i in g}) < len(g) for g in groups):
            return "violates a constraint"
        return "valid grid, wrong solution"
    q = task["question"]
    rows = [line.rstrip("\r") for line in run.lines(r["text"])]
    if len(rows) != 30 or any(len(x) != 30 for x in rows):
        return "malformed output"
    a = "".join(rows)
    s, g = q.index("S"), q.index("G")

    def bfs(ok):
        dist, dq = {s: 0}, deque([s])
        while dq:
            u = dq.popleft()
            r_, c_ = divmod(u, 30)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r_ + dr, c_ + dc
                v = rr * 30 + cc
                if 0 <= rr < 30 and 0 <= cc < 30 and v not in dist and ok(v):
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return dist.get(g)

    path = {k for k in range(900) if a[k] == "o"}
    if any(q[k] == "#" for k in path):
        return "path crosses a wall"
    reach = bfs(lambda v: v in path or v == g)
    if reach is None:
        return "path does not connect S to G"
    if reach > bfs(lambda v: q[v] != "#"):
        return "path longer than shortest"
    return "altered walls or S/G (path itself shortest)"


if __name__ == "__main__":
    results = json.load(open(HERE / "results.json"))
    by = collections.defaultdict(list)
    for r in results:
        by[(r["model"], r["bench"])].append(r)
    print(f"{'model':28s} {'bench':9s} {'n':>3s} {'solved':>7s} {'out tok':>8s} {'$/puzzle':>9s} {'sec':>5s}")
    for (m, b), rs in sorted(by.items()):
        ok = [r for r in rs if r.get("usage")]
        print(f"{m:28s} {b:9s} {len(rs):3d} {100 * sum(r['correct'] for r in rs) / len(rs):6.0f}% "
              f"{sum(r['usage']['completion_tokens'] for r in ok) / max(len(ok), 1):8.0f} "
              f"{sum(r['usage']['cost'] for r in ok) / max(len(ok), 1):9.3f} "
              f"{sum(r['seconds'] for r in rs) / len(rs):5.0f}")
    print("\nfailures:")
    for (m, b), rs in sorted(by.items()):
        c = collections.Counter(failure(TASKS[(b, r["idx"])], r) for r in rs if not r["correct"])
        print(f"{m:28s} {b:9s} {dict(c)}")
    print("\ntotal cost $%.2f" % sum(r["usage"]["cost"] for r in results if r.get("usage")))
