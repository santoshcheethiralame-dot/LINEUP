import types

import pytest

from lineup.data.multihop import parse_example
from lineup.data.sources import load_examples

WIKI2_RAW = {
    "id": "abc",
    "question": "Who directed the film?",
    "answer": "Tim Burton",
    "type": "compositional",
    "evidences": [["The film", "director", "Tim Burton"]],
    "context": {
        "title": ["The Film", "Unrelated"],
        "sentences": [["The film was directed by Tim Burton."], ["An unrelated fact."]],
    },
    "supporting_facts": {"title": ["The Film"], "sent_id": [0]},
}


def test_parse_example_records_the_source_and_splits_gold():
    example = parse_example(WIKI2_RAW, source="2wiki")
    assert example.meta["source"] == "2wiki"
    assert example.answer == "Tim Burton"
    assert [c.title for c in example.gold_chunks] == ["The Film"]
    assert example.gold_chunks[0].supporting_sentence_ids == (0,)
    assert [c.title for c in example.distractor_pool] == ["Unrelated"]


def test_dispatch_rejects_an_unknown_dataset():
    with pytest.raises(ValueError):
        load_examples("nope")


def test_dispatch_is_lazy_for_known_datasets():
    # the loaders are generators, so dispatch touches no network until iterated
    assert isinstance(load_examples("2wiki", "validation", limit=1), types.GeneratorType)
    assert isinstance(load_examples("hotpotqa", "validation", limit=1), types.GeneratorType)
