"""Neutral clean-prefix reporting helpers shared by sudoku / snowflake runners."""

from __future__ import annotations


def prefix_outcomes(res, n: int) -> tuple[dict[int, dict], range, int]:
    """Build outcomes map + maximal gap-free prefix [0..k] from a SolveResult."""
    outcomes: dict[int, dict] = {}
    for i in range(n):
        if int(res.puzzle_calls[i].item()) < 0:
            continue
        outcomes[i] = {
            "idx": i,
            "correct": bool(res.correct[i].item()),
            "wrong": bool(res.wrong[i].item()),
            "timeout": bool(res.timeouts[i].item()),
            "round_solved": int(res.round_solved[i].item()),
            "n_resets": int(res.n_resets[i].item()),
            "puzzle_calls": int(res.puzzle_calls[i].item()),
        }
    k = -1
    while (k + 1) in outcomes:
        k += 1
    prefix_idxs = range(0, k + 1)
    return outcomes, prefix_idxs, k + 1


def prefix_summary_metrics(outcomes: dict[int, dict], prefix_idxs) -> dict:
    n_prefix = len(list(prefix_idxs)) if not isinstance(prefix_idxs, range) else (
        prefix_idxs.stop - prefix_idxs.start)
    # range is fine: use length via stop
    if isinstance(prefix_idxs, range):
        n_prefix = prefix_idxs.stop - prefix_idxs.start
    n_correct = sum(1 for i in prefix_idxs if outcomes[i]["correct"])
    n_wrong = sum(1 for i in prefix_idxs if outcomes[i]["wrong"])
    n_timeout = sum(1 for i in prefix_idxs if outcomes[i]["timeout"])
    avg_resets = (
        float(sum(int(outcomes[i].get("n_resets", 0)) for i in prefix_idxs) / n_prefix)
        if n_prefix > 0 else 0.0)
    avg_puzzle_calls = (
        float(sum(int(outcomes[i]["puzzle_calls"]) for i in prefix_idxs) / n_prefix)
        if n_prefix > 0 else -1.0)
    prefix_correct_rounds = [
        int(outcomes[i]["round_solved"]) for i in prefix_idxs
        if outcomes[i]["correct"] and int(outcomes[i]["round_solved"]) >= 0
    ]
    avg_rounds_solved = (
        float(sum(prefix_correct_rounds) / len(prefix_correct_rounds))
        if prefix_correct_rounds else 0.0)
    return {
        "n_prefix": n_prefix,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_timeout": n_timeout,
        "avg_resets": avg_resets,
        "avg_puzzle_calls": avg_puzzle_calls,
        "avg_rounds_solved": avg_rounds_solved,
    }
