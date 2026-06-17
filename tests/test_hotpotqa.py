from lineup.data.hotpotqa import parse_example

RAW = {
    "id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "context": {
        "title": ["Scott Derrickson", "Ed Wood", "Tim Burton"],
        "sentences": [
            ["Scott Derrickson is an American director.", " He was born in 1966."],
            ["Edward Davis Wood Jr. was an American filmmaker."],
            ["Tim Burton is an American director."],
        ],
    },
    "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
}


def test_parse_splits_gold_and_distractors():
    example = parse_example(RAW)

    assert example.answer == "yes"
    assert {c.title for c in example.gold_chunks} == {"Scott Derrickson", "Ed Wood"}
    assert [c.title for c in example.distractor_pool] == ["Tim Burton"]

    derrickson = next(c for c in example.gold_chunks if c.title == "Scott Derrickson")
    assert derrickson.provenance == "gold"
    assert derrickson.supporting_sentence_ids == (0,)
    assert example.distractor_pool[0].provenance == "distractor"


def test_parse_handles_record_layout():
    raw = {
        "id": "x", "question": "q?", "answer": "yes", "type": "comparison", "level": "hard",
        "context": [
            {"title": "A", "sentences": ["A is American."]},
            {"title": "B", "sentences": ["B is British."]},
        ],
        "supporting_facts": [{"title": "A", "sent_id": 0}],
    }
    example = parse_example(raw)
    assert [c.title for c in example.gold_chunks] == ["A"]
    assert [c.title for c in example.distractor_pool] == ["B"]
    assert example.gold_chunks[0].supporting_sentence_ids == (0,)
