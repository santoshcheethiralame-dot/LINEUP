from lineup.backends.base import Generation, LanguageModel, Scoring
from lineup.correctness import LLMJudge
from lineup.oracle import answer_key
from lineup.textnorm import contains_phrase, normalize


def test_decimal_is_not_fused_into_an_integer():
    assert normalize("3.14") == "3 14"
    assert not contains_phrase("the value was 314 units", "3.14")
    assert contains_phrase("pi is about 3.14 today", "3.14")


def test_thousands_separator_still_equals_the_plain_integer():
    assert normalize("1,000") == normalize("1000") == "1000"


def test_short_value_still_not_found_inside_a_longer_one():
    assert not contains_phrase("the year was 1885", "5")   # the property the docstring promises


def test_answer_key_prefers_the_planted_wrong_value_when_it_embeds_gold():
    assert answer_key("Washington DC", "D.C.", "Washington DC") == "wrong"
    assert answer_key("Franklin Roosevelt", "Roosevelt", "Franklin Roosevelt") == "wrong"


def test_answer_key_keys_gold_and_wrong_and_other_normally():
    assert answer_key("Gustave Eiffel", "Gustave Eiffel", "Alexandre Bartholdi") == "gold"
    assert answer_key("Alexandre Bartholdi", "Gustave Eiffel", "Alexandre Bartholdi") == "wrong"
    assert answer_key("someone else entirely", "Gustave Eiffel", "Alexandre Bartholdi") == "someone else entirely"


class _Echo(LanguageModel):
    def __init__(self, text):
        self._text = text

    def generate(self, messages, max_new_tokens=None):
        return Generation(self._text, [1], [-0.1])

    def score(self, messages, response):
        return Scoring([], [], [-0.1])


def test_judge_reads_a_cue_word_not_only_the_first_token():
    assert LLMJudge(_Echo("The answer is correct")).is_correct("q", "g", "p") is True
    assert LLMJudge(_Echo("No, that is wrong")).is_correct("q", "g", "p") is False
    assert LLMJudge(_Echo("yes")).is_correct("q", "g", "p") is True
    assert LLMJudge(_Echo("banana pancakes")).is_correct("q", "g", "p") is False
