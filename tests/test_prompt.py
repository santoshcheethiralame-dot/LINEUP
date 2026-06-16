from lineup.data.schema import Chunk, Recipe, Scenario
from lineup.prompt import build_messages, render_context


def _scenario() -> Scenario:
    chunks = [
        Chunk("a", "Alpha", "Alpha text.", ["Alpha text."]),
        Chunk("b", "Beta", "Beta text.", ["Beta text."]),
    ]
    recipe = Recipe(
        seed=0, k=2, original_value="x", intended_wrong_answer="y", substitution_type="entity",
        source_gold_chunk_id="a", source_sentence_id=0, gold_chunk_ids=["a"],
        misleading_chunk_id="b", distractor_chunk_ids=[], order=["a", "b"],
    )
    return Scenario(qid="q", question="What is X?", gold_answer="x", chunks=chunks, recipe=recipe)


def test_render_context_numbers_chunks_in_order():
    rendered = render_context(_scenario().chunks)
    assert rendered.index("[1] Alpha") < rendered.index("[2] Beta")


def test_build_messages_has_system_and_user_with_question():
    messages = build_messages(_scenario())
    assert [m.role for m in messages] == ["system", "user"]
    assert "What is X?" in messages[1].content
    assert "Alpha" in messages[1].content and "Beta" in messages[1].content
