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
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_review_sheet(path) -> list:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def review_agreement(rows) -> dict:
    """Agreement between the human labels and the oracle, over the rows a human filled in."""
    filled = [row for row in rows if row.get("human_role", "").strip()]
    if not filled:
        return {"n": 0, "agreement": None}
    agree = sum(1 for row in filled if row["human_role"].strip() == row["oracle_role"].strip())
    return {"n": len(filled), "agreement": agree / len(filled)}
