"""Evaluate retrieval (hit@1, hit@3 per method) and optionally the full pipeline's answers on eval/questions.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag import Index, ask, build_index, tokenize

QUESTIONS = json.loads((Path(__file__).parent / "eval" / "questions.json").read_text(encoding="utf-8"))
METHODS = ["dense", "bm25_plain", "bm25", "hybrid"]


def retrieval_table(index: Index, questions: list[dict] = QUESTIONS) -> str:
    answerable = [q for q in questions if q["store_id"]]  # out-of-data questions have no gold store
    rows = ["| method | hit@1 | hit@3 | not at rank 1 |", "|---|---|---|---|"]
    for method in METHODS:
        top3 = {q["id"]: [c.id for c, _ in index.search(q["question"], 3, method)] for q in answerable}
        hit1 = [q["id"] for q in answerable if top3[q["id"]][0] == q["store_id"]]
        hit3 = [q["id"] for q in answerable if q["store_id"] in top3[q["id"]]]
        misses = ", ".join(q["id"] for q in answerable if q["id"] not in hit1) or "–"
        rows.append(f"| {method} | {len(hit1) / len(answerable):.2f} | {len(hit3) / len(answerable):.2f} | {misses} |")
    return "\n".join(rows) + f"\n\n{len(answerable)} questions with a gold store; out-of-data questions excluded."


def is_correct(answer: str, expected: list[list[str]]) -> bool:
    """Each group must match; a group matches if any alternative occurs (ignoring case, diacritics, spaces)."""
    compact = lambda s: "".join(tokenize(s, stem=False))  # noqa: E731
    return all(any(compact(alt) in compact(answer) for alt in group) for group in expected)


def answer_table(index: Index, questions: list[dict] = QUESTIONS) -> str:
    rows, correct = ["| question | ok | answer |", "|---|---|---|"], 0
    for q in questions:
        answer, _ = ask(q["question"], index)
        ok = is_correct(answer, q["expected"])
        correct += ok
        rows.append(f"| {q['id']} | {'✅' if ok else '❌'} | {answer.replace(chr(10), ' ')} |")
    return f"Answer accuracy: **{correct}/{len(questions)}** (keyword check, review answers manually)\n\n" + "\n".join(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="also run the LLM on every question (slow on CPU)")
    args = parser.parse_args()
    index = build_index()
    print(retrieval_table(index))
    if args.llm:
        print("\n" + answer_table(index))
