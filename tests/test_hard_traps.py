from lineup.backends.base import Generation, LanguageModel, Scoring
from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import Chunk, QAExample
from lineup.data.substitution import build_answer_pool
from lineup.generation import generate_and_judge
from lineup.oracle import leave_one_out
from lineup.textnorm import contains_phrase


def _example(qid, question, answer, gold_text):
    gold = Chunk(f"{qid}::g", "Gold", gold_text, [gold_text], provenance="gold", supporting_sentence_ids=(0,))
    distractors = [
        Chunk(f"{qid}::d{i}", f"P{i}", text, [text])
        for i, text in enumerate(["Paris is in France.", "Iron is a metal.", "A tower is tall.", "The Seine flows north."])
    ]
    return QAExample(qid, question, answer, [gold], distractors)


POOL = build_answer_pool([
    _example("q1", "Who designed the Eiffel Tower?", "Gustave Eiffel", "The Eiffel Tower was designed by Gustave Eiffel."),
    _example("q2", "Who painted the Mona Lisa?", "Leonardo da Vinci", "The Mona Lisa was painted by Leonardo da Vinci."),
])
EXAMPLE = _example("q1", "Who designed the Eiffel Tower?", "Gustave Eiffel", "The Eiffel Tower was designed by Gustave Eiffel.")


def test_hard_trap_adds_a_redundant_decoy():
    scenario = ScenarioBuilder(answer_pool=POOL, k=6, seed=0, hard_traps=True).build(EXAMPLE)
    assert scenario is not None
    decoy = next(c for c in scenario.chunks if c.provenance == "decoy")
    near_miss = next(c for c in scenario.chunks if c.provenance == "misleading")
    wrong = scenario.recipe.intended_wrong_answer
    # the near-miss and the decoy each independently assert the wrong value -> redundant evidence
    assert contains_phrase(near_miss.text, wrong)
    assert contains_phrase(decoy.text, wrong)
    assert scenario.recipe.decoy_chunk_id == decoy.chunk_id


def test_normal_mode_adds_no_decoy():
    scenario = ScenarioBuilder(answer_pool=POOL, k=6, seed=0).build(EXAMPLE)
    assert scenario is not None
    assert scenario.recipe.decoy_chunk_id == ""
    assert all(c.provenance != "decoy" for c in scenario.chunks)


class _RedundancyFooled(LanguageModel):
    """Adopts the wrong value whenever any present passage asserts it; else gives the gold answer."""

    def __init__(self, gold, wrong):
        self._gold, self._wrong = gold, wrong

    def generate(self, messages, max_new_tokens=None):
        present = self._wrong in messages[-1].content
        return Generation(self._wrong if present else self._gold, [1], [-0.1])

    def score(self, messages, response):
        return Scoring([], [], [-0.1])


def test_redundant_decoy_dissolves_the_single_culprit():
    # The mechanism check: a model fooled by either redundant copy should, under hard traps,
    # have NO single culprit (each wrong-value chunk is non-causal) — both become misleading.
    normal = ScenarioBuilder(answer_pool=POOL, k=6, seed=0).build(EXAMPLE)
    wrong = normal.recipe.intended_wrong_answer
    model = _RedundancyFooled(EXAMPLE.answer, wrong)

    generation = generate_and_judge(model, normal, llm_judge=None)
    assert not generation.is_correct
    normal_roles = leave_one_out(model, normal, generation)
    assert any(role.role == "culprit" for role in normal_roles.chunk_roles)   # lone near-miss is the culprit

    hard = ScenarioBuilder(answer_pool=POOL, k=6, seed=0, hard_traps=True).build(EXAMPLE)
    hard_generation = generate_and_judge(model, hard, llm_judge=None)
    assert not hard_generation.is_correct
    hard_roles = leave_one_out(model, hard, hard_generation)
    assert not any(role.role == "culprit" for role in hard_roles.chunk_roles)   # redundancy dissolves it
    assert sum(role.role == "misleading" for role in hard_roles.chunk_roles) >= 2
