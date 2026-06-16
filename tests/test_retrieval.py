from lineup.data.retrieval import BM25DistractorRetriever
from lineup.data.schema import Chunk


def _chunk(title: str, text: str) -> Chunk:
    return Chunk(chunk_id=title, title=title, text=text)


def test_retrieve_excludes_gold_and_ranks_by_overlap():
    corpus = [
        _chunk("Paris", "Paris is the capital of France."),
        _chunk("France", "France is a country in Europe and its capital is Paris."),
        _chunk("Germany", "Germany is a country in central Europe."),
        _chunk("Spain", "Spain is a country in southern Europe."),
    ]
    retriever = BM25DistractorRetriever(corpus)

    hits = retriever.retrieve("capital of France", k=2, exclude_titles=["Paris"])
    titles = [hit.title for hit in hits]

    assert "Paris" not in titles
    assert titles[0] == "France"
    assert all(hit.provenance == "distractor" for hit in hits)
