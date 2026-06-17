from lineup.backends.base import Generation, LanguageModel, Scoring
from lineup.data.schema import Chunk, GenerationResult, Recipe, Scenario
from lineup.data.serialization import case_roles_from_dict
from lineup.generation import generate_and_judge
from lineup.oracle import answer_key, assign_role, is_salient, leave_one_out

RIGHT = "Gustave Eiffel"
WRONG = "Alexandre Bartholdi"


def _scenario() -> Scenario:
    gold = Chunk("g", "Eiffel Tower", f"The tower was designed by {RIGHT}.", [f"The tower was designed by {RIGHT}."], provenance="gold")
    misleading = Chunk("m", "Eiffel Tower", f"The tower was designed by {WRONG}.", [f"The tower was designed by {WRONG}."], provenance="misleading")
    distractor = Chunk("d", "Paris", "Paris is in France.", ["Paris is in France."], provenance="distractor")
    recipe = Recipe(
        seed=0, k=3, original_value=RIGHT, intended_wrong_answer=WRONG, substitution_type="entity",
        source_gold_chunk_id="g", source_sentence_id=0, gold_chunk_ids=["g"],
        misleading_chunk_id="m", distractor_chunk_ids=["d"], order=["g", "m", "d"],
    )
    return Scenario(qid="q", question="Who designed the Eiffel Tower?", gold_answer=RIGHT, chunks=[gold, misleading, distractor], recipe=recipe)


class _Triggered(LanguageModel):
    """Answers WRONG when the misleading chunk's value is in context, else RIGHT."""

    def generate(self, messages, max_new_tokens=None):
        wrong_present = WRONG in messages[-1].content
        return Generation(WRONG if wrong_present else RIGHT, [1], [-0.1])

    def score(self, messages, response):
        wrong_present = WRONG in messages[-1].content
        return Scoring([], [], [-0.5] if wrong_present else [-2.0])


class _AlwaysWrong(LanguageModel):
    """Answers WRONG no matter the context."""

    def generate(self, messages, max_new_tokens=None):
        return Generation(WRONG, [1], [-0.1])

    def score(self, messages, response):
        return Scoring([], [], [-1.0])


def _roles_by_provenance(case):
    return {role.provenance: role for role in case.chunk_roles}


def test_assign_role_covers_the_four_quadrants():
    assert assign_role(True, True) == "culprit"
    assert assign_role(False, True) == "misleading"
    assert assign_role(True, False) == "silent"
    assert assign_role(False, False) == "inert"


def test_answer_key_collapses_phrasing():
    assert answer_key("Gustave Eiffel", RIGHT, WRONG) == "gold"
    assert answer_key("the engineer Gustave Eiffel himself", RIGHT, WRONG) == "gold"


def test_answer_key_distinguishes_values():
    assert answer_key(WRONG, RIGHT, WRONG) == "wrong"
    assert answer_key(WRONG, RIGHT, WRONG) != answer_key(RIGHT, RIGHT, WRONG)


def test_is_salient_uses_the_canonical_value_for_a_verbose_answer():
    gold_chunk = Chunk("g", "T", f"The tower was designed by {RIGHT}.", ["..."])
    distractor = Chunk("d", "D", "Paris is in France.", ["..."])
    assert is_salient(gold_chunk, "the answer is Gustave Eiffel", RIGHT, WRONG)
    assert not is_salient(distractor, "the answer is Gustave Eiffel", RIGHT, WRONG)


def test_causal_and_salient_misleading_chunk_is_a_culprit():
    scenario = _scenario()
    model = _Triggered()
    original = generate_and_judge(model, scenario)
    assert original.model_answer == WRONG and not original.is_correct

    case = leave_one_out(model, scenario, original)
    roles = _roles_by_provenance(case)
    assert roles["misleading"].role == "culprit"      # removing it fixes the answer, and it holds the answer
    assert roles["misleading"].now_correct is True
    assert roles["misleading"].delta_logprob > 0
    assert roles["gold"].role == "inert"
    assert roles["distractor"].role == "inert"


def test_salient_but_non_causal_misleading_chunk_is_misleading():
    scenario = _scenario()
    model = _AlwaysWrong()
    original = generate_and_judge(model, scenario)
    assert original.model_answer == WRONG

    case = leave_one_out(model, scenario, original)
    roles = _roles_by_provenance(case)
    assert roles["misleading"].role == "misleading"   # looks responsible but removing it changes nothing
    assert roles["misleading"].causal is False
    assert roles["misleading"].salient is True


def test_case_roles_round_trip_through_dict():
    import dataclasses

    scenario = _scenario()
    case = leave_one_out(_Triggered(), scenario, generate_and_judge(_Triggered(), scenario))
    restored = case_roles_from_dict(dataclasses.asdict(case))
    assert restored.qid == case.qid
    assert [r.role for r in restored.chunk_roles] == [r.role for r in case.chunk_roles]
