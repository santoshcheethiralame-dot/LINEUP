from __future__ import annotations

from typing import Sequence

from .data.schema import CaseRoles, Scenario

ROLES = ("culprit", "misleading", "silent", "inert")


def _roles_by_chunk(case: CaseRoles | None) -> dict:
    if case is None:
        return {}
    return {role.chunk_id: role for role in case.chunk_roles}


def role_counts(cases: Sequence[CaseRoles]) -> dict:
    counts = {role: 0 for role in ROLES}
    for case in cases:
        for role in case.chunk_roles:
            if role.role in counts:
                counts[role.role] += 1
    return counts


def case_records(scenarios: Sequence[Scenario], cases: Sequence[CaseRoles]) -> list[dict]:
    """One row per case: the question-level view of the benchmark."""
    by_qid = {case.qid: case for case in cases}
    records = []
    for scenario in scenarios:
        case = by_qid.get(scenario.qid)
        roles = _roles_by_chunk(case)
        records.append(
            {
                "qid": scenario.qid,
                "question": scenario.question,
                "gold_answer": scenario.gold_answer,
                "model_answer": case.original_answer if case else "",
                "original_correct": case.original_correct if case else False,
                "k": scenario.recipe.k,
                "num_chunks": len(scenario.chunks),
                "has_culprit": any(role.role == "culprit" for role in roles.values()),
                "intended_wrong_answer": scenario.recipe.intended_wrong_answer,
                "substitution_type": scenario.recipe.substitution_type,
                "chunk_order": [chunk.chunk_id for chunk in scenario.chunks],
            }
        )
    return records


def chunk_records(scenarios: Sequence[Scenario], cases: Sequence[CaseRoles]) -> list[dict]:
    """One row per passage: the unit a method is scored against."""
    by_qid = {case.qid: case for case in cases}
    records = []
    for scenario in scenarios:
        case = by_qid.get(scenario.qid)
        roles = _roles_by_chunk(case)
        for position, chunk in enumerate(scenario.chunks):
            role = roles.get(chunk.chunk_id)
            records.append(
                {
                    "qid": scenario.qid,
                    "question": scenario.question,
                    "gold_answer": scenario.gold_answer,
                    "model_answer": case.original_answer if case else "",
                    "original_correct": case.original_correct if case else False,
                    "position": position,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "provenance": chunk.provenance,
                    # Unlabelled passages (those in cases the model answered correctly) carry
                    # an empty role and default flags, so every column keeps a stable type and
                    # the dataset loads even when no case has labels. Filter on role != "".
                    "role": role.role if role else "",
                    "is_causal": role.causal if role else False,
                    "is_salient": role.salient if role else False,
                    "delta_logprob": role.delta_logprob if role else 0.0,
                }
            )
    return records


def dataset_statistics(scenarios: Sequence[Scenario], cases: Sequence[CaseRoles]) -> dict:
    labeled = [case for case in cases if case.chunk_roles]
    ks = [scenario.recipe.k for scenario in scenarios]
    return {
        "n_cases": len(scenarios),
        "n_labeled_cases": len(labeled),
        "n_chunks": sum(len(scenario.chunks) for scenario in scenarios),
        "role_counts": role_counts(cases),
        "k": ks[0] if ks else None,
    }


def build_manifest(
    *,
    lineup_version: str,
    model: str,
    seed: int,
    source_dataset: str,
    source_split: str,
    created: str,
    git_commit: str | None,
    packages: dict,
    statistics: dict,
) -> dict:
    """A record of exactly how a release was produced, for reproduction."""
    return {
        "name": "lineup",
        "lineup_version": lineup_version,
        "model": model,
        "seed": seed,
        "source": {"dataset": source_dataset, "split": source_split},
        "created": created,
        "git_commit": git_commit,
        "packages": packages,
        "statistics": statistics,
    }


def data_card(stats: dict) -> str:
    """Render the HuggingFace dataset card (README) for a release."""
    roles = stats.get("role_counts", {role: 0 for role in ROLES})
    frontmatter = "\n".join(
        [
            "---",
            "license: mit",
            "task_categories:",
            "- question-answering",
            "language:",
            "- en",
            "tags:",
            "- retrieval-augmented-generation",
            "- attribution",
            "- benchmark",
            "pretty_name: lineup organic chunk-role attribution",
            "configs:",
            "- config_name: chunks",
            "  data_files: chunks.jsonl",
            "- config_name: cases",
            "  data_files: cases.jsonl",
            "---",
        ]
    )
    body = f"""# lineup — organic chunk-role attribution

A controlled benchmark for tracing which retrieved passage caused a confidently wrong answer in retrieval-augmented generation, in the organic (non-adversarial) error setting.

Each case is a multi-hop question whose retrieved context holds the gold supporting passages, realistic distractors, and one constructed near-miss: a passage cloned from the gold paragraph with the answer replaced by a plausible, same-type wrong value. The model answers from this context, and where it answers wrongly an exact leave-one-out procedure assigns every passage its true role.

## Roles

|            | salient    | not salient |
| ---------- | ---------- | ----------- |
| causal     | culprit    | silent      |
| not causal | misleading | inert       |

A passage is *causal* if removing it changes the answer, and *salient* if it carries the value the model produced. The hard case is the misleading passage: salient but not causal. A method that blames it has confused correlation with cause.

## Configurations

`chunks` — one row per passage, the unit a method is scored against:

| field | description |
| ----- | ----------- |
| qid | case identifier; group rows by it to reconstruct the context |
| question | the question posed to the model |
| gold_answer | the reference answer |
| model_answer | the answer the model gave for this case |
| original_correct | whether the model answered the case correctly |
| position | the passage's position in the presented context (0-based) |
| chunk_id | passage identifier |
| title | passage title |
| text | passage text |
| provenance | how the passage was sourced: gold, distractor, or misleading |
| role | the ground-truth role (culprit, misleading, silent, inert); empty on correct cases |
| is_causal | did removing the passage change the answer |
| is_salient | does the passage carry the model's answer value |
| delta_logprob | drop in the answer's log-probability when the passage is removed |

`cases` — one row per case, with question-level metadata: `qid`, `question`, `gold_answer`, `model_answer`, `original_correct`, `k`, `num_chunks`, `has_culprit`, `intended_wrong_answer`, `substitution_type`, and `chunk_order`.

## Using the benchmark

Load the `chunks` configuration, group rows by `qid`, and present each case's passages (ordered by `position`) to your system. For a case the model answers wrongly, your method should name the culprit passage; score that prediction against `role`. The headline number is the *misleading-as-culprit rate* — how often a method blames a passage whose role is `misleading` rather than `culprit`. Because the labels come only from the known gold answer and the model's own behaviour, they are independent of any method under test.

## Statistics

- Cases: {stats.get("n_cases", 0)} ({stats.get("n_labeled_cases", 0)} with leave-one-out labels)
- Passages: {stats.get("n_chunks", 0)}
- Role counts: culprit {roles.get("culprit", 0)}, misleading {roles.get("misleading", 0)}, silent {roles.get("silent", 0)}, inert {roles.get("inert", 0)}
- Passages per case (k): {stats.get("k", "n/a")}
- Model: {stats.get("model", "n/a")}
- Seed: {stats.get("seed", "n/a")}
- Source: {stats.get("source_dataset", "n/a")} ({stats.get("source_split", "n/a")})

## Reproduction

Scenario construction is deterministic given the seed and the source corpus; generation is greedy (temperature 0). Rebuild the run with `python scripts/run_pipeline.py` and repackage it with `python scripts/build_release.py`. Run all model stages on one machine and one model revision: greedy decoding is deterministic per machine, but log-probabilities drift across GPUs and precisions. The accompanying `manifest.json` records the model, seed, source split, and package versions used for this release.

## Limitations

The labels describe one model's behaviour on one question family; a different model may be fooled by different passages. Organic wrong answers are a minority of cases, so the labeled set is smaller than the full case set. The near-miss is built by typed value substitution and covers years, numbers, and named entities. All questions are in English.

## License

Released under the MIT License. The underlying questions and passages derive from HotpotQA (CC BY-SA 4.0) and 2WikiMultiHopQA; refer to their original licenses for the source content.

## Citation

```bibtex
@misc{{lineup,
  title  = {{lineup: a controlled benchmark for organic chunk-role attribution in RAG}},
  author = {{Santosh Cheethirala}},
  year   = {{2026}},
  note   = {{https://github.com/santoshcheethiralame-dot/LINEUP}}
}}
```
"""
    return frontmatter + "\n\n" + body
