import json

from lineup.data.schema import GenerationResult
from lineup.data.serialization import read_generations, write_generations


def test_truncated_flag_roundtrips(tmp_path):
    result = GenerationResult("q", "question", "gold", "ans", True, "exact", False, -0.1, truncated=True)
    path = tmp_path / "generations.jsonl"
    write_generations(path, [result])
    assert read_generations(path)[0].truncated is True


def test_generations_without_a_truncated_field_default_to_false(tmp_path):
    path = tmp_path / "generations.jsonl"
    record = {
        "qid": "q", "question": "x", "gold_answer": "g", "model_answer": "a",
        "is_correct": False, "judged_by": "exact", "matched_intended_wrong": False, "answer_logprob": -0.1,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert read_generations(path)[0].truncated is False
