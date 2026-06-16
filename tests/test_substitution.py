from random import Random

from lineup.data.schema import QAExample
from lineup.data.substitution import build_answer_pool, classify_answer, perturb_value


def test_classify_answer():
    assert classify_answer("1889") == "year"
    assert classify_answer("42") == "number"
    assert classify_answer("3,000") == "number"
    assert classify_answer("Gustave Eiffel") == "entity"


def test_perturb_year_is_a_different_plausible_year():
    out = perturb_value("1889", "year", {}, Random(0))
    assert out != "1889"
    assert out.isdigit() and len(out) == 4


def test_perturb_number_scales_to_a_different_value():
    out = perturb_value("100", "number", {}, Random(0))
    assert out != "100"
    assert out.isdigit()


def test_perturb_entity_samples_a_different_pool_member():
    pool = {"year": [], "number": [], "entity": ["Alexandre Gustave", "Henri Banks"]}
    out = perturb_value("Gustave Eiffel", "entity", pool, Random(1))
    assert out in pool["entity"]
    assert out != "Gustave Eiffel"


def test_perturb_is_deterministic():
    pool = {"year": [], "number": [], "entity": ["Alexandre Gustave", "Henri Banks"]}
    assert perturb_value("Gustave Eiffel", "entity", pool, Random(7)) == perturb_value(
        "Gustave Eiffel", "entity", pool, Random(7)
    )


def test_perturb_year_avoids_a_value_already_in_context():
    context = "It was built between 1887 and 1889."
    outs = {perturb_value("1889", "year", {}, Random(seed), context=context) for seed in range(20)}
    assert "1887" not in outs   # already a true value in the paragraph
    assert "1889" not in outs   # the answer itself


def test_perturb_entity_avoids_a_value_already_in_context():
    pool = {"year": [], "number": [], "entity": ["Alexandre Gustave", "Henri Banks"]}
    out = perturb_value("Gustave Eiffel", "entity", pool, Random(0), context="Henri Banks lived nearby")
    assert out == "Alexandre Gustave"   # "Henri Banks" is in the context, so it is excluded


def test_build_answer_pool_dedupes_and_types():
    examples = [
        QAExample("a", "q", "1889", [], []),
        QAExample("b", "q", "Gustave Eiffel", [], []),
        QAExample("c", "q", "gustave eiffel", [], []),  # case-insensitive duplicate
    ]
    pool = build_answer_pool(examples)
    assert pool["year"] == ["1889"]
    assert pool["entity"] == ["Gustave Eiffel"]
