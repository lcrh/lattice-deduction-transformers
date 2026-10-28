"""Zero-shot frontier LLMs on the first 50 puzzles of each pinned test ordering,
with the appendix prompts, via OpenRouter (key in OPENROUTER_API_KEY).

The puzzles are in puzzles.json, so a run needs no dataset; `export` rebuilds
that file from the Sudoku-Extreme and Maze-Hard test splits (Hugging Face) and
the Snowflake test split written by experiments/snowflake/gen_data.py.
Answers go to results/results.json; puzzles already answered there are skipped.

usage (from the repo root):
    uv run python baselines/frontier_llm/run.py
    uv run --with pyarrow python baselines/frontier_llm/run.py export --snowflake-test PATH
"""
import argparse
import concurrent.futures as cf
import csv
import glob
import json
import os
import time
import urllib.request
from collections import deque
from pathlib import Path

HERE = Path(__file__).parent
PUZZLES = HERE / "puzzles.json"
RESULTS = HERE / "results" / "results.json"
N = 50
MODELS = ["anthropic/claude-opus-5.5", "openai/gpt-6-astra", "openai/gpt-6.1-sol"]
HF = Path.home() / ".cache/huggingface/hub"

SUDOKU_SYSTEM = """You are a Sudoku solver. You will be given a 9 x 9 Sudoku puzzle. Solve it and respond ONLY with the completed 9 x 9 grid, one row per line, digits only, no spaces or other characters. Example response format:
534678912
672195348
...
345286179"""

SNOWFLAKE_SYSTEM = """You are solving a Snowflake Sudoku puzzle, a Sudoku variant played on a snowflake-shaped topology of N cells, each holding a digit from 1 to 6. The puzzle is defined by a list of constraint groups; each group is a set of cell IDs whose digits must all be distinct. You will be given (1) the total number of cells, (2) the list of constraint groups, and (3) the current cell values, where ? marks a cell you must solve and an integer 1-6 marks a given. Respond with ONLY the solution: one line per cell in order, formatted as CELL_ID:DIGIT with no spaces, comments, or other text. Output exactly N lines. Example:
0:3
1:5
2:1
3:6
4:2
5:4"""

MAZE_SYSTEM = """You are a maze solver. You will be given a 30 x 30 maze. Every cell of the maze is exactly one of the following four characters:
'#' - wall (impassable)
' ' - free cell
'S' - the start cell (exactly one)
'G' - the goal cell (exactly one)
Find a path of orthogonally adjacent (up/down/left/right) free cells that connects S to G without passing through any wall. Diagonal moves are not allowed. Mark every free cell that lies on the path with the character o. The cells #, S, and G are unchanged, and free cells that are not on the path remain ' '. Respond with ONLY the completed 30 x 30 grid: exactly 30 lines, each containing exactly 30 characters drawn from the set {#, ' ', S, G, o}. Do not include row numbers, column markers, blank lines, code fences, prose, or any other text. Whitespace inside each row is significant: a leading or trailing space is itself a free cell, not formatting. Example response format (shown for a 6 x 6 maze for brevity; your output must be 30 rows of 30 characters):
######
#Sooo#
#  #o#
#  #o#
## ooG
######"""


def hf_test(name):
    return list(csv.DictReader(open(glob.glob(str(HF / f"datasets--sapientinc--{name}/snapshots/*/test.csv"))[0])))


def sudoku_tasks():
    from ldt.eval.indices import load

    rows = hf_test("sudoku-extreme")
    out = []
    for i in load("sudoku_test_perm").indices[:N]:
        q = rows[i]["question"].replace(".", "0")
        text = "\n".join(q[r * 9:(r + 1) * 9] for r in range(9))
        out.append(dict(bench="sudoku", idx=int(i),
                        user=f"Solve this Sudoku:\n{text}", answer=rows[i]["answer"]))
    return out


def snowflake_tasks(test_parquet: str):
    import pyarrow.parquet as pq

    rows = pq.read_table(test_parquet).to_pylist()
    out = []
    from ldt.eval.indices import load

    for i in load("snowflake_extreme_test").indices[:N]:
        r = rows[i]
        t = json.loads(r["topology"])
        groups = "\n".join(f"  Group {g}: {c['cells']}" for g, c in enumerate(t["constraints"]))
        cells = "\n".join(f"  Cell {c}: {'?' if v == 7 else v}" for c, v in enumerate(r["puzzle"]))
        text = (f"Puzzle (n={r['n']}, {t['n_cells']} cells):\n\n"
                f"Constraint groups (each group must contain distinct digits):\n{groups}\n\n"
                f"Cell values (? = solve this):\n{cells}")
        out.append(dict(bench="snowflake", idx=int(i), order=r["n"],
                        user=f"Solve this Snowflake Sudoku:\n{text}",
                        answer=r["solution"], puzzle=r["puzzle"],
                        groups=[c["cells"] for c in t["constraints"]]))
    return out


def maze_tasks():
    from ldt.eval.indices import load

    rows = hf_test("maze-30x30-hard-1k")
    out = []
    for i in load("maze_test").indices[:N]:
        q = rows[i]["question"]
        text = "\n".join(q[r * 30:(r + 1) * 30] for r in range(30))
        out.append(dict(bench="maze", idx=int(i),
                        user=f"Solve this maze:\n{text}", question=q))
    return out


SYSTEM = {"sudoku": SUDOKU_SYSTEM, "snowflake": SNOWFLAKE_SYSTEM, "maze": MAZE_SYSTEM}


def load_tasks() -> list[dict]:
    return json.loads(PUZZLES.read_text())


def lines(text):
    text = text.strip("\n")
    if text.startswith("```"):
        text = "\n".join(line for line in text.split("\n") if not line.startswith("```"))
    return text.split("\n")


def snowflake_answer(text) -> dict[int, int]:
    """{cell: digit} from the CELL:DIGIT lines of a Snowflake answer."""
    got = {}
    for line in lines(text):
        if ":" in line:
            c, v = line.strip().split(":", 1)
            if c.isdigit() and v.strip().isdigit():
                got[int(c)] = int(v)
    return got


def maze_bfs(q: str, ok) -> dict[int, int]:
    """Distances from S over the 30 x 30 cells `ok` accepts."""
    s = q.index("S")
    dist, dq = {s: 0}, deque([s])
    while dq:
        u = dq.popleft()
        r, c = divmod(u, 30)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            v = rr * 30 + cc
            if 0 <= rr < 30 and 0 <= cc < 30 and v not in dist and ok(v):
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def score(task, text):
    """True if the response is a correct solution (any shortest path for Maze)."""
    if text is None:
        return False
    if task["bench"] == "sudoku":
        return "".join(line.strip() for line in lines(text)) == task["answer"]
    if task["bench"] == "snowflake":
        got = snowflake_answer(text)
        return [got.get(c) for c in range(len(task["answer"]))] == list(task["answer"])
    q = task["question"]
    rows = [line.rstrip("\r") for line in lines(text)]
    if len(rows) != 30 or any(len(r) != 30 for r in rows):
        return False
    a = "".join(rows)
    if any(q[k] in "#SG" and a[k] != q[k] for k in range(900)):
        return False
    if any(q[k] == " " and a[k] not in " o" for k in range(900)):
        return False
    g = q.index("G")
    best = maze_bfs(q, lambda v: q[v] != "#")[g]
    path = {k for k in range(900) if a[k] == "o"}
    return len(path) == best - 1 and maze_bfs(q, lambda v: v in path or v == g).get(g) == best


def ask(model, task):
    body = json.dumps({"model": model, "reasoning": {"effort": "high"}, "max_tokens": 100_000,
                       "messages": [{"role": "system", "content": SYSTEM[task["bench"]]},
                                    {"role": "user", "content": task["user"]}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                                          "Content-Type": "application/json"})
    t0 = time.time()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                d = json.load(resp)
            msg = d["choices"][0]["message"]
            return dict(text=msg.get("content"), usage=d.get("usage"),
                        finish=d["choices"][0].get("finish_reason"), seconds=time.time() - t0)
        except Exception as e:  # retry transient failures, then record the error
            err = repr(e)
            time.sleep(10)
    return dict(text=None, error=err, seconds=time.time() - t0)


def evaluate() -> None:
    tasks = load_tasks()
    old = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    results = [r for r in old if r["model"] in MODELS and r.get("text") is not None]
    done = {(r["model"], r["bench"], r["idx"]) for r in results}
    jobs = [(m, t) for m in MODELS for t in tasks if (m, t["bench"], t["idx"]) not in done]
    print(f"{len(results)} earlier results kept, {len(jobs)} new calls", flush=True)
    RESULTS.parent.mkdir(exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=50) as ex:
        futs = {ex.submit(ask, m, t): (m, t) for m, t in jobs}
        for f in cf.as_completed(futs):
            m, t = futs[f]
            r = f.result()
            ok = score(t, r.get("text"))
            results.append(dict(model=m, bench=t["bench"], idx=t["idx"], order=t.get("order"),
                                correct=ok, **r))
            print(f"{m:28s} {t['bench']:9s} {t['idx']:7d} {'OK ' if ok else 'no '} "
                  f"{r.get('finish')} {r.get('seconds', 0):6.0f}s {r.get('error', '')}", flush=True)
            RESULTS.write_text(json.dumps(results, indent=1))
    for m in MODELS:
        print(m, {b: sum(r["correct"] for r in results if r["model"] == m and r["bench"] == b)
                  for b in ("sudoku", "snowflake", "maze")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="evaluate", choices=["evaluate", "export"])
    ap.add_argument("--snowflake-test", help="snowflake_extreme_test.parquet (export only)")
    args = ap.parse_args()
    if args.command == "export":
        tasks = sudoku_tasks() + snowflake_tasks(args.snowflake_test) + maze_tasks()
        PUZZLES.write_text(json.dumps(tasks, indent=1))
        print(f"wrote {len(tasks)} puzzles to {PUZZLES}")
    else:
        evaluate()
