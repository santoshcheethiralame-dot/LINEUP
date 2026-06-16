from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable

from .schema import Chunk, GenerationResult, Recipe, Scenario


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


def write_scenarios(path, scenarios: Iterable[Scenario]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario_to_dict(scenario), ensure_ascii=False) + "\n")


def read_scenarios(path) -> list[Scenario]:
    scenarios = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                scenarios.append(scenario_from_dict(json.loads(line)))
    return scenarios


def write_generations(path, results: Iterable[GenerationResult]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(dataclasses.asdict(result), ensure_ascii=False) + "\n")


def read_generations(path) -> list[GenerationResult]:
    results = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                results.append(GenerationResult(**json.loads(line)))
    return results
