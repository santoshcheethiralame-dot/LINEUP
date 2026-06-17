from __future__ import annotations

from .backends.base import LanguageModel
from .correctness import judge_correct, matches_intended_wrong
from .data.schema import GenerationResult, Scenario
from .prompt import build_messages


def generate_and_judge(model: LanguageModel, scenario: Scenario, *, llm_judge=None) -> GenerationResult:
    """Run the model on one scenario and label the answer correct or wrong."""
    generation = model.generate(build_messages(scenario))
    answer = generation.text.strip()
    is_correct, judged_by = judge_correct(scenario.question, scenario.gold_answer, answer, llm_judge)

    return GenerationResult(
        qid=scenario.qid,
        question=scenario.question,
        gold_answer=scenario.gold_answer,
        model_answer=answer,
        is_correct=is_correct,
        judged_by=judged_by,
        matched_intended_wrong=matches_intended_wrong(answer, scenario.recipe.intended_wrong_answer),
        answer_logprob=generation.mean_logprob,
        truncated=generation.truncated,
    )
