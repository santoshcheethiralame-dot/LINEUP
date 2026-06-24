from lineup.data.musique import parse_musique


RAW = {
    "id": "2hop__123",
    "question": "Who designed the tower in the city where the Louvre is?",
    "answer": "Gustave Eiffel",
    "question_decomposition": [{"question": "city of the Louvre"}, {"question": "tower designer"}],
    "paragraphs": [
        {"idx": 0, "title": "Louvre", "paragraph_text": "The Louvre is a museum in Paris. It is large.", "is_supporting": True},
        {"idx": 1, "title": "Eiffel Tower", "paragraph_text": "The Eiffel Tower in Paris was designed by Gustave Eiffel.", "is_supporting": True},
        {"idx": 2, "title": "Iron", "paragraph_text": "Iron is a metal.", "is_supporting": False},
    ],
}


def test_parse_musique_splits_gold_and_distractors():
    ex = parse_musique(RAW)
    assert ex.qid == "2hop__123"
    assert ex.answer == "Gustave Eiffel"
    assert len(ex.gold_chunks) == 2 and len(ex.distractor_pool) == 1
    assert all(c.provenance == "gold" for c in ex.gold_chunks)
    assert ex.distractor_pool[0].provenance == "distractor"
    assert ex.meta["type"] == "2hop"                      # hop count recorded for the structure analysis
    # the supporting paragraph is sentence-split, with all sentences marked supporting
    louvre = ex.gold_chunks[0]
    assert len(louvre.sentences) == 2 and louvre.supporting_sentence_ids == (0, 1)
