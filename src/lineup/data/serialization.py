from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable

from .schema import CaseRoles, Chunk, ChunkRole, GenerationResult, Recipe, Scenario


def scenario_to_dict(scenario: Scenario) -> dict:
    return dataclasses.asdict(scenario)


def scenario_from_dict(data: dict) -> Scenario:
    chunks = []
    for raw in data["chunks"]:
        chunk = Chunk(**raw)
        chunk.supporting_sentence_ids = tuple(chunk.supporting_sentence_ids)
        chunks.append(chunk)
    return Scenario(
        qid=data["qid"],
        question=data["question"],
        gold_answer=data["gold_answer"],
        chunks=chunks,
        recipe=Recipe(**data["recipe"]),
        meta=data.get("meta", {}),
    )


def case_roles_from_dict(data: dict) -> CaseRoles:
    return CaseRoles(
        qid=data["qid"],
        question=data["question"],
        gold_answer=data["gold_answer"],
        original_answer=data["original_answer"],
        original_correct=data["original_correct"],
        chunk_roles=[ChunkRole(**role) for role in data["chunk_roles"]],
    )


def _write_jsonl(path, records: Iterable, to_dict) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_dict(record), ensure_ascii=False) + "\n")


def _read_jsonl(path, from_dict) -> list:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(from_dict(json.loads(line)))
    return records


def write_scenarios(path, scenarios: Iterable[Scenario]) -> None:
    _write_jsonl(path, scenarios, scenario_to_dict)


def read_scenarios(path) -> list[Scenario]:
    return _read_jsonl(path, scenario_from_dict)


def write_generations(path, results: Iterable[GenerationResult]) -> None:
    _write_jsonl(path, results, dataclasses.asdict)


def read_generations(path) -> list[GenerationResult]:
    return _read_jsonl(path, lambda data: GenerationResult(**data))


def write_roles(path, cases: Iterable[CaseRoles]) -> None:
    _write_jsonl(path, cases, dataclasses.asdict)


def read_roles(path) -> list[CaseRoles]:
    return _read_jsonl(path, case_roles_from_dict)
