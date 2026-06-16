from lineup.backends.base import Generation, LanguageModel, Scoring
from lineup.correctness import LLMJudge, matches_intended_wrong, normalize, normalized_exact_match


class _Echo(LanguageModel):
    def __init__(self, reply: str):
        self._reply = reply

    def generate(self, messages, max_new_tokens=None):
        return Generation(text=self._reply, token_ids=[], token_logprobs=[])

    def score(self, messages, response):
        return Scoring(tokens=[], token_ids=[], logprobs=[])


def test_normalize_strips_articles_punctuation_and_case():
    assert normalize("The Eiffel Tower!") == "eiffel tower"


def test_normalized_exact_match():
    assert normalized_exact_match("Gustave Eiffel", "  gustave eiffel ")
    assert not normalized_exact_match("Bartholdi", "Eiffel")


def test_matches_intended_wrong_allows_containment():
    assert matches_intended_wrong("It was Alexandre Bartholdi.", "Alexandre Bartholdi")
    assert not matches_intended_wrong("Gustave Eiffel", "Alexandre Bartholdi")


def test_llm_judge_parses_yes_and_no():
    assert LLMJudge(_Echo("yes")).is_correct("q", "gold", "pred") is True
    assert LLMJudge(_Echo("No, incorrect")).is_correct("q", "gold", "pred") is False
