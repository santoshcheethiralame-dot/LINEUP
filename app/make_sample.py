"""Write a small, illustrative set of pipeline outputs so the explorer runs out of the box.

Replace the files in app/sample with a real run's outputs/ to explore actual results.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lineup.data.schema import (
    CaseRoles,
    Chunk,
    ChunkRole,
    ChunkScore,
    GenerationResult,
    MethodPrediction,
    Recipe,
    Scenario,
)
from lineup.data.serialization import (
    write_generations,
    write_predictions,
    write_roles,
    write_scenarios,
)

# Each chunk is (id, title, text, provenance, role); each case lists method -> per-chunk scores.
CASES = [
    {
        "qid": "c1", "question": "Who designed the Eiffel Tower?", "gold": "Gustave Eiffel",
        "wrong": "Alexandre Bartholdi", "answer": "Alexandre Bartholdi", "correct": False, "logprob": -0.20,
        "chunks": [
            ("c1::0", "Eiffel Tower", "The Eiffel Tower was designed by the engineer Gustave Eiffel.", "gold", "inert"),
            ("c1::1", "Eiffel Tower (cached)", "The Eiffel Tower was designed by Alexandre Bartholdi, completed in 1889.", "misleading", "misleading"),
            ("c1::2", "Statue of Liberty", "Its internal iron pylon was engineered in Paris and shipped abroad in 1885.", "distractor", "silent"),
            ("c1::3", "Wrought iron", "A malleable iron alloy used widely in 19th-century construction.", "distractor", "inert"),
        ],
        "scores": {"contextcite": [0.1, 0.7, 0.3, 0.0], "lexical_similarity": [0.5, 0.8, 0.2, 0.1], "llm_judge": [0.0, 1.0, 0.0, 0.0]},
    },
    {
        "qid": "c2", "question": "In what year did the Eiffel Tower open?", "gold": "1889",
        "wrong": "1887", "answer": "1887", "correct": False, "logprob": -0.55,
        "chunks": [
            ("c2::0", "Eiffel Tower", "The Eiffel Tower opened to the public in 1889.", "gold", "inert"),
            ("c2::1", "Eiffel Tower (draft)", "The Eiffel Tower opened to the public in 1887.", "misleading", "culprit"),
            ("c2::2", "Paris expositions", "The city hosted World's Fairs in 1878, 1889 and 1900.", "distractor", "inert"),
            ("c2::3", "Gustave Eiffel", "A French civil engineer known for the tower and the Garabit viaduct.", "distractor", "inert"),
        ],
        "scores": {"contextcite": [0.2, 0.9, 0.1, 0.0], "lexical_similarity": [0.6, 0.7, 0.3, 0.2], "llm_judge": [0.0, 1.0, 0.0, 0.0]},
    },
    {
        "qid": "c3", "question": "Who discovered penicillin?", "gold": "Alexander Fleming",
        "wrong": "Howard Florey", "answer": "Howard Florey", "correct": False, "logprob": -0.70,
        "chunks": [
            ("c3::0", "Penicillin", "Penicillin was discovered by Alexander Fleming in 1928.", "gold", "inert"),
            ("c3::1", "Penicillin development", "Penicillin was turned into a usable drug by Howard Florey's team.", "distractor", "culprit"),
            ("c3::2", "Penicillin (note)", "Penicillin was discovered by Howard Florey in 1928.", "misleading", "misleading"),
            ("c3::3", "Antibiotics", "A class of drugs that inhibit the growth of bacteria.", "distractor", "inert"),
        ],
        "scores": {"contextcite": [0.1, 0.7, 0.3, 0.0], "lexical_similarity": [0.3, 0.5, 0.6, 0.1], "llm_judge": [0.0, 0.0, 1.0, 0.0]},
    },
    {
        "qid": "c4", "question": "What is the capital of Australia?", "gold": "Canberra",
        "wrong": "Sydney", "answer": "Sydney", "correct": False, "logprob": -0.24,
        "chunks": [
            ("c4::0", "Canberra", "Canberra is the capital city of Australia.", "gold", "inert"),
            ("c4::1", "Australia (overview)", "Australia's capital is Sydney, its largest city.", "misleading", "misleading"),
            ("c4::2", "Sydney", "The most populous city in Australia, on the east coast.", "distractor", "silent"),
            ("c4::3", "Melbourne", "A major Australian city, the seat of government until 1927.", "distractor", "inert"),
        ],
        "scores": {"contextcite": [0.1, 0.6, 0.4, 0.0], "lexical_similarity": [0.4, 0.8, 0.6, 0.2], "llm_judge": [0.0, 1.0, 0.0, 0.0]},
    },
    {
        "qid": "c5", "question": "Who wrote Romeo and Juliet?", "gold": "William Shakespeare",
        "wrong": "Christopher Marlowe", "answer": "William Shakespeare", "correct": True, "logprob": -0.25,
        "chunks": [
            ("c5::0", "Romeo and Juliet", "The play Romeo and Juliet was written by William Shakespeare.", "gold", "culprit"),
            ("c5::1", "Romeo and Juliet (alt)", "The play Romeo and Juliet was written by Christopher Marlowe.", "misleading", "misleading"),
            ("c5::2", "Elizabethan theatre", "A period of English drama in the late 16th century.", "distractor", "inert"),
            ("c5::3", "The Globe", "The London playhouse associated with Shakespeare's company.", "distractor", "inert"),
        ],
        "scores": {"contextcite": [0.8, 0.2, 0.0, 0.1], "lexical_similarity": [0.7, 0.6, 0.2, 0.3], "llm_judge": [1.0, 0.0, 0.0, 0.0]},
    },
    {
        "qid": "c6", "question": "What is the chemical symbol for gold?", "gold": "Au",
        "wrong": "Ag", "answer": "Au", "correct": True, "logprob": -0.30,
        "chunks": [
            ("c6::0", "Gold", "Gold has the chemical symbol Au, from the Latin aurum.", "gold", "culprit"),
            ("c6::1", "Gold (typo)", "Gold has the chemical symbol Ag.", "misleading", "misleading"),
            ("c6::2", "Silver", "A precious metal with the symbol Ag and atomic number 47.", "distractor", "inert"),
            ("c6::3", "Periodic table", "Organises the elements by atomic number and properties.", "distractor", "inert"),
        ],
        "scores": {"contextcite": [0.7, 0.1, 0.2, 0.0], "lexical_similarity": [0.6, 0.5, 0.4, 0.2], "llm_judge": [1.0, 0.0, 0.0, 0.0]},
    },
]


def build(case):
    chunks = case["chunks"]
    chunk_objs = [Chunk(cid, title, text, [text], provenance=prov) for cid, title, text, prov, _ in chunks]
    gold_ids = [c[0] for c in chunks if c[3] == "gold"]
    misleading_id = next((c[0] for c in chunks if c[3] == "misleading"), "")
    recipe = Recipe(
        seed=0, k=len(chunks), original_value=case["gold"], intended_wrong_answer=case["wrong"],
        substitution_type="entity", source_gold_chunk_id=gold_ids[0] if gold_ids else "", source_sentence_id=0,
        gold_chunk_ids=gold_ids, misleading_chunk_id=misleading_id,
        distractor_chunk_ids=[c[0] for c in chunks if c[3] == "distractor"], order=[c[0] for c in chunks],
    )
    scenario = Scenario(case["qid"], case["question"], case["gold"], chunk_objs, recipe)

    generation = GenerationResult(
        case["qid"], case["question"], case["gold"], case["answer"], case["correct"], "judge",
        case["wrong"].lower() in case["answer"].lower(), case["logprob"],
    )

    chunk_roles = []
    for cid, _, _, prov, role in chunks:
        causal = role in ("culprit", "silent")
        chunk_roles.append(ChunkRole(
            chunk_id=cid, provenance=prov, role=role, causal=causal, salient=role in ("culprit", "misleading"),
            now_correct=causal and not case["correct"], delta_logprob=1.6 if causal else 0.0,
            answer_without=case["gold"] if causal and not case["correct"] else case["answer"],
        ))
    case_roles = CaseRoles(case["qid"], case["question"], case["gold"], case["answer"], case["correct"], chunk_roles)

    predictions = []
    for method, scores in case["scores"].items():
        chunk_scores = [ChunkScore(c[0], c[3], score) for c, score in zip(chunks, scores)]
        best = max(chunk_scores, key=lambda s: s.score).chunk_id
        predictions.append(MethodPrediction(case["qid"], method, best, chunk_scores))
    return scenario, generation, case_roles, predictions


def main():
    scenarios, generations, roles, predictions = [], [], [], []
    for case in CASES:
        scenario, generation, case_roles, case_predictions = build(case)
        scenarios.append(scenario)
        generations.append(generation)
        roles.append(case_roles)
        predictions.extend(case_predictions)

    out = Path(__file__).parent / "sample"
    out.mkdir(exist_ok=True)
    write_scenarios(out / "scenarios.jsonl", scenarios)
    write_generations(out / "generations.jsonl", generations)
    write_roles(out / "roles.jsonl", roles)
    write_predictions(out / "predictions.jsonl", predictions)
    print(f"wrote {len(scenarios)} cases to {out}")


if __name__ == "__main__":
    main()
