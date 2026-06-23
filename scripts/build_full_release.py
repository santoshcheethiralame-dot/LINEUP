"""Package the full 12-cell benchmark (3 models x 2 datasets x 2 conditions) as one release.

Each cell contributes its cases and per-passage roles; every record is tagged with its dataset,
condition, and model, and grouped by a composite case_id (roles are model-specific). Writes an
anonymized data card suitable for double-blind submission (no author name or repository URL).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from lineup.config import PROJECT_ROOT
from lineup.data.serialization import read_roles, read_scenarios
from lineup.release import ROLES, case_records, chunk_records, dataset_statistics

DATA = PROJECT_ROOT / "paper" / "results_data"
MODELS = {"qwen": "Qwen2.5-7B", "phi": "Phi-3.5-mini", "mistral": "Mistral-7B"}
DATASETS = ["hotpotqa", "2wiki"]
CONDS = ["baseline", "hardtraps"]
SOURCE = {"hotpotqa": "hotpotqa/hotpot_qa", "2wiki": "2WikiMultiHopQA"}


def anonymized_card(total, per_cell) -> str:
    rc = total["role_counts"]
    lines = [
        "---", "license: mit", "task_categories:", "- question-answering",
        "language:", "- en", "tags:", "- retrieval-augmented-generation", "- attribution", "- benchmark",
        "pretty_name: LINEUP organic chunk-role attribution",
        "configs:", "- config_name: chunks", "  data_files: chunks.jsonl",
        "- config_name: cases", "  data_files: cases.jsonl", "---", "",
        "# LINEUP — organic chunk-role attribution (full benchmark)", "",
        "A controlled benchmark for tracing which retrieved passage caused a confidently wrong answer in",
        "retrieval-augmented generation, in the organic (non-adversarial) error setting. Every retrieved",
        "passage is assigned a ground-truth causal role by an exact leave-one-out oracle that uses only the",
        "known gold answer and the model's own behaviour — never an attribution method under test.", "",
        "This release spans the full matrix: three model families x two multi-hop QA datasets x two",
        "conditions (a single planted near-miss, and hard-traps with redundant decoys). Roles are",
        "model-specific, so each case is identified by `case_id = dataset/condition/model/qid`.", "",
        "## Roles", "",
        "|            | salient    | not salient |",
        "| ---------- | ---------- | ----------- |",
        "| causal     | culprit    | silent      |",
        "| not causal | misleading | inert       |", "",
        "A passage is *causal* if removing it changes the answer, and *salient* if it carries the value the",
        "model produced. The hard case is the misleading passage: salient but not causal.", "",
        "## Cell breakdown", "",
        "| dataset | condition | model | cases | labeled |",
        "| ------- | --------- | ----- | ----: | ------: |",
    ]
    for (ds, cond, model_name), st in per_cell:
        lines.append(f"| {ds} | {cond} | {model_name} | {st['n_cases']} | {st['n_labeled_cases']} |")
    lines += [
        "", "## Fields", "",
        "Both configs carry `dataset`, `condition`, `model`, and `case_id` (group passages by `case_id`).",
        "`chunks` — one row per passage: `qid`, `question`, `gold_answer`, `model_answer`, `original_correct`,",
        "`position`, `chunk_id`, `title`, `text`, `provenance` (gold / distractor / misleading / decoy),",
        "`role` (culprit / misleading / silent / inert; empty on correct cases), `is_causal`, `is_salient`,",
        "`delta_logprob`. `cases` — one row per case with question-level metadata.", "",
        "## Statistics", "",
        f"- Cells: {len(per_cell)} (3 models x 2 datasets x 2 conditions)",
        f"- Cases: {total['n_cases']} ({total['n_labeled_cases']} with leave-one-out labels)",
        f"- Passages: {total['n_chunks']}",
        f"- Role counts: " + ", ".join(f"{r} {rc[r]}" for r in ROLES),
        "- Source: HotpotQA (validation) and 2WikiMultiHopQA; greedy decoding, seeded.", "",
        "## Using the benchmark", "",
        "Load `chunks`, group rows by `case_id`, present each case's passages (ordered by `position`) to your",
        "system, and for a wrongly-answered case have your method name the culprit passage; score against",
        "`role`. Because the labels come only from the gold answer and the model's own behaviour, they are",
        "independent of any method under test.", "",
        "## License", "",
        "Released under the MIT License. The underlying questions and passages derive from HotpotQA",
        "(CC BY-SA 4.0) and 2WikiMultiHopQA; refer to their original licenses for the source content.", "",
        "## Citation", "",
        "> Citation withheld for anonymous review.", "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "release_full")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    all_cases, all_chunks, per_cell = [], [], []
    total_roles = Counter()
    n_cases = n_labeled = n_chunks = 0

    for ds in DATASETS:
        for cond in CONDS:
            for mk, mname in MODELS.items():
                d = DATA / ds / cond / mk
                sp, rp = d / "scenarios.jsonl", d / "roles.jsonl"
                if not (sp.exists() and rp.exists()):
                    continue
                scen = read_scenarios(sp)
                cases = read_roles(rp)
                cid = f"{ds}/{cond}/{mk}"
                for rec in case_records(scen, cases):
                    all_cases.append({"dataset": ds, "condition": cond, "model": mname,
                                      "case_id": f"{cid}/{rec['qid']}", **rec})
                for rec in chunk_records(scen, cases):
                    all_chunks.append({"dataset": ds, "condition": cond, "model": mname,
                                       "case_id": f"{cid}/{rec['qid']}", **rec})
                st = dataset_statistics(scen, cases)
                per_cell.append(((ds, cond, mname), st))
                total_roles.update(st["role_counts"])
                n_cases += st["n_cases"]
                n_labeled += st["n_labeled_cases"]
                n_chunks += st["n_chunks"]

    with (out / "cases.jsonl").open("w", encoding="utf-8") as h:
        for r in all_cases:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as h:
        for r in all_chunks:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = {"n_cases": n_cases, "n_labeled_cases": n_labeled, "n_chunks": n_chunks,
             "role_counts": dict(total_roles)}
    (out / "README.md").write_text(anonymized_card(total, per_cell), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "name": "lineup", "cells": len(per_cell), "created": datetime.now(timezone.utc).isoformat(),
        "statistics": total, "anonymized": True,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"cells: {len(per_cell)}  cases: {n_cases} ({n_labeled} labeled)  passages: {n_chunks}")
    print("roles:", dict(total_roles))
    for name in ("cases.jsonl", "chunks.jsonl", "README.md", "manifest.json"):
        print("wrote", out / name)


if __name__ == "__main__":
    main()
