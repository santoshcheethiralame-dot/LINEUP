from __future__ import annotations

from .backends.base import LanguageModel
from .correctness import LLMJudge, matches_intended_wrong, normalized_exact_match
from .data.schema import GenerationResult, Scenario
from .prompt import build_messages


def generate_and_judge(
    model: LanguageModel, scenario: Scenario, *, llm_judge: LLMJudge | None = None
) -> GenerationResult:
    """Run the model on one scenario and label the answer correct or wrong.

    Correctness is decided by normalized exact match first; phrasing variants that miss
    are deferred to the LLM judge when one is supplied, otherwise counted as wrong.
    """
    generation = model.generate(build_messages(scenario))
    answer = generation.text.strip()
    gold = scenario.gold_answer

    if normalized_exact_match(answer, gold):
        is_correct, judged_by = True, "exact"
    elif llm_judge is not None:
        is_correct, judged_by = llm_judge.is_correct(scenario.question, gold, answer), "judge"
    else:
        is_correct, judged_by = False, "exact"

    return GenerationResult(
        qid=scenario.qid,
        question=scenario.question,
        gold_answer=gold,
        model_answer=answer,
        is_correct=is_correct,
        judged_by=judged_by,
        matched_intended_wrong=matches_intended_wrong(answer, scenario.recipe.intended_wrong_answer),
        answer_logprob=generation.mean_logprob,
    )
