from __future__ import annotations

import csv
from pathlib import Path
from random import Random

REVIEW_FIELDS = ("qid", "chunk_id", "question", "model_answer", "chunk_text", "oracle_role", "human_role")


def sample_review_rows(scenarios, cases, *, n: int = 30, seed: int = 0, roles=("culprit", "misleading")):
    """Sample labelled passages for a human to confirm.

    Defaults to the contested roles — culprit and misleading — since those are the labels the
    benchmark's claims rest on. Each row leaves ``human_role`` blank for the reviewer to fill.
    """
    chunk_text = {chunk.chunk_id: chunk.text for scenario in scenarios for chunk in scenario.chunks}
    scenarios_by_qid = {scenario.qid: scenario for scenario in scenarios}
    rows = []
    for case in cases:
        if case.qid not in scenarios_by_qid:
            continue
        for role in case.chunk_roles:
            if role.role in roles:
                rows.append(
                    {
                        "qid": case.qid,
                        "chunk_id": role.chunk_id,
                        "question": case.question,
                        "model_answer": case.original_answer,
                        "chunk_text": chunk_text.get(role.chunk_id, ""),
                        "oracle_role": role.role,
                        "human_role": "",
                    }
                )
    Random(seed).shuffle(rows)
    return rows[:n]


def write_review_sheet(path, rows) -> None:
    # utf-8-sig so Excel/Sheets render non-ASCII passage text correctly when reviewers open it.
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_review_sheet(path) -> list:
    # utf-8-sig reads both BOM'd (Excel-saved) and plain utf-8 files.
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def review_agreement(rows) -> dict:
    """Agreement between the human labels and the oracle, over the rows a human filled in."""
    filled = [row for row in rows if row.get("human_role", "").strip()]
    if not filled:
        return {"n": 0, "agreement": None}
    agree = sum(1 for row in filled if row["human_role"].strip() == row["oracle_role"].strip())
    return {"n": len(filled), "agreement": agree / len(filled)}


# --- Blind validation: a human picks the culprit passage from the whole case ----------------
# The reviewer sees the question, the model's wrong answer, and every passage (lettered), and
# names the one they think caused the error -- or "none" if no single passage is to blame. This
# is the judgment a human can actually make, and "none" doubles as a check on the no-single-
# culprit claim. The answer key (the oracle's culprit letter, or "none") lives in a separate
# file keyed by an opaque row_id, so the reviewer never sees the label under test. The same
# blank sheet goes to every reviewer, which also yields inter-rater agreement.

LETTERS = "ABCDEFGHIJ"
SCENARIO_FIELDS = ("row_id", "question", "model_answer", "passages", "your_pick")
KEY_FIELDS = ("row_id", "source", "qid", "answer", "n_chunks")


def _normalize_pick(value: str) -> str:
    value = (value or "").strip().lower()
    if value == "":
        return ""
    if value in {"none", "n", "-"}:
        return "none"
    if value in {"unsure", "skip", "?", "idk", "dunno", "dont know", "don't know"}:
        return "unsure"
    return value.upper()


def sample_scenario_rows(sources, *, n: int = 50, seed: int = 0, prefer=None):
    """Pool wrong cases from several runs into a blank culprit-picking sheet + a hidden key.

    ``sources`` is a list of ``(label, scenarios, cases)``; cases are the wrong cases. Each row
    shows all passages lettered in their stored (already randomized) order. The pool is deduped
    to one row per distinct question. If ``prefer`` is a list of keywords, cases whose text
    mentions one are floated to the front, so the sample leans toward topics the annotators know.
    Returns ``(blank_rows, key_rows)``.
    """
    pool = []
    for label, scenarios, cases in sources:
        by_qid = {scenario.qid: scenario for scenario in scenarios}
        for case in cases:
            scenario = by_qid.get(case.qid)
            if scenario is None or len(scenario.chunks) > len(LETTERS):
                continue
            culprits = [role.chunk_id for role in case.chunk_roles if role.role == "culprit"]
            pool.append(
                {
                    "label": label,
                    "qid": case.qid,
                    "question": case.question,
                    "answer": case.original_answer,
                    "chunks": list(scenario.chunks),
                    "culprit_id": culprits[0] if culprits else None,
                }
            )
    Random(seed).shuffle(pool)
    seen, deduped = set(), []
    for row in pool:  # one row per distinct question
        if row["qid"] not in seen:
            seen.add(row["qid"])
            deduped.append(row)
    pool = deduped
    if prefer:
        keywords = [word.lower() for word in prefer]
        def familiar(row):
            blob = (row["question"] + " " + " ".join(chunk.text for chunk in row["chunks"])).lower()
            return any(word in blob for word in keywords)
        pool.sort(key=lambda row: not familiar(row))  # stable: keeps the shuffled order within tiers
    pool = pool[:n]

    blank, key = [], []
    for index, row in enumerate(pool):
        row_id = f"r{index:03d}"
        letter_of = {chunk.chunk_id: LETTERS[i] for i, chunk in enumerate(row["chunks"])}
        passages = "\n\n".join(f"{LETTERS[i]}) {chunk.text}" for i, chunk in enumerate(row["chunks"]))
        blank.append(
            {
                "row_id": row_id,
                "question": row["question"],
                "model_answer": row["answer"],
                "passages": passages,
                "your_pick": "",
            }
        )
        answer = letter_of.get(row["culprit_id"], "none") if row["culprit_id"] else "none"
        key.append(
            {
                "row_id": row_id,
                "source": row["label"],
                "qid": row["qid"],
                "answer": answer,
                "n_chunks": len(row["chunks"]),
            }
        )
    return blank, key


def write_csv(path, rows, fields) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_scenario(key_rows, sheets: dict) -> dict:
    """Score filled culprit-picking sheets against the hidden key.

    ``sheets`` maps a reviewer name to their filled rows. Returns each reviewer's agreement with
    the oracle, the majority-vote agreement, the average pairwise agreement between humans, and
    the agreement restricted to the no-culprit ("none") cases.
    """
    oracle = {row["row_id"]: _normalize_pick(row["answer"]) for row in key_rows}
    per_reviewer, picks_by_row = {}, {}
    for name, rows in sheets.items():
        normalized = [(r["row_id"], _normalize_pick(r.get("your_pick", ""))) for r in rows]
        picks = [(row_id, pick) for row_id, pick in normalized if pick not in ("", "unsure")]
        unsure = sum(1 for _, pick in normalized if pick == "unsure")
        agree = sum(1 for row_id, pick in picks if oracle.get(row_id) == pick)
        per_reviewer[name] = {"n": len(picks), "agreement": agree / len(picks) if picks else None, "unsure": unsure}
        for row_id, pick in picks:
            picks_by_row.setdefault(row_id, []).append(pick)

    majority_agree = majority_n = 0
    none_agree = none_n = 0
    pair_match = pair_total = 0
    for row_id, picks in picks_by_row.items():
        if row_id in oracle:
            vote = max(set(picks), key=picks.count)
            majority_n += 1
            majority_agree += int(vote == oracle[row_id])
            if oracle[row_id] == "none":
                none_n += 1
                none_agree += int(vote == "none")
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                pair_total += 1
                pair_match += int(picks[i] == picks[j])
    return {
        "per_reviewer": per_reviewer,
        "majority_vs_oracle": majority_agree / majority_n if majority_n else None,
        "majority_n": majority_n,
        "none_recall": none_agree / none_n if none_n else None,
        "none_n": none_n,
        "inter_rater": pair_match / pair_total if pair_total else None,
    }
