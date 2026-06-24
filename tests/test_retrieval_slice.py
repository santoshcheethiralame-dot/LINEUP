from lineup.data.retrieval import BM25DistractorRetriever, build_corpus
from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import Chunk, QAExample
from lineup.data.substitution import build_answer_pool


def _example(qid, question, answer, gold_text, distractor_texts):
    gold = Chunk(f"{qid}::g", f"{qid}-G", gold_text, [gold_text], provenance="gold", supporting_sentence_ids=(0,))
    distractors = [Chunk(f"{qid}::d{i}", f"{qid}-P{i}", t, [t]) for i, t in enumerate(distractor_texts)]
    return QAExample(qid, question, answer, [gold], distractors)


EXAMPLES = [
    _example("q1", "Who designed the Eiffel Tower?", "Gustave Eiffel",
             "The Eiffel Tower was designed by Gustave Eiffel.",
             ["The Eiffel Tower stands in Paris.", "Towers are tall structures."]),
    _example("q2", "Who painted the Mona Lisa?", "Leonardo da Vinci",
             "The Mona Lisa was painted by Leonardo da Vinci.",
             ["The Louvre museum is in Paris.", "Painting is a visual art."]),
]


def test_retriever_sources_real_distractors():
    pool = build_answer_pool(EXAMPLES)
    retriever = BM25DistractorRetriever(build_corpus(EXAMPLES))
    scenario = ScenarioBuilder(answer_pool=pool, k=4, seed=0, natural=True, retriever=retriever).build(EXAMPLES[0])

    assert scenario is not None
    assert scenario.recipe.substitution_type == "retrieval"          # tagged as the real-retrieval slice
    assert scenario.recipe.misleading_chunk_id == ""                 # no planting
    assert any(c.provenance == "gold" for c in scenario.chunks)      # gold still present
    nongold = [c for c in scenario.chunks if c.provenance != "gold"]
    assert nongold and all(c.retrieval_score is not None for c in nongold)  # distractors came from the retriever
