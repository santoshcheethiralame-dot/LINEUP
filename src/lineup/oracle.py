from __future__ import annotations

from .backends.base import LanguageModel
from .correctness import judge_correct, normalized_exact_match
from .data.schema import CaseRoles, ChunkRole, GenerationResult, Scenario
from .prompt import build_messages_for
from .textnorm import contains_phrase


def assign_role(causal: bool, salient: bool) -> str:
    if causal and salient:
        return "culprit"
    if salient:
        return "misleading"
    if causal:
        return "silent"
    return "inert"


def leave_one_out(
    model: LanguageModel, scenario: Scenario, original: GenerationResult, *, llm_judge=None
) -> CaseRoles:
    """Assign every chunk its 2x2 role by exact leave-one-out.

    The causal axis is whether removing the chunk changes the model's answer; the salience
    axis is whether the chunk actually contains the model's answer. Both are derived from
    the known gold answer and the model's own behavior, never from a method under test, so
    the labels are non-circular.
    """
    chunks = scenario.chunks
    answer = original.model_answer
    full_messages = build_messages_for(scenario.question, chunks)
    full_logprob = model.score(full_messages, answer).total_logprob if answer else 0.0

    chunk_roles = []
    for index, chunk in enumerate(chunks):
        remaining = [c for j, c in enumerate(chunks) if j != index]
        ablated_messages = build_messages_for(scenario.question, remaining)
        answer_without = model.generate(ablated_messages).text.strip()

        causal = not normalized_exact_match(answer_without, answer)
        salient = contains_phrase(chunk.text, answer)
        now_correct, _ = judge_correct(scenario.question, scenario.gold_answer, answer_without, llm_judge)
        ablated_logprob = model.score(ablated_messages, answer).total_logprob if answer else 0.0

        chunk_roles.append(
            ChunkRole(
                chunk_id=chunk.chunk_id,
                provenance=chunk.provenance,
                role=assign_role(causal, salient),
                causal=causal,
                salient=salient,
                now_correct=now_correct,
                delta_logprob=full_logprob - ablated_logprob,
                answer_without=answer_without,
            )
        )

    return CaseRoles(
        qid=scenario.qid,
        question=scenario.question,
        gold_answer=scenario.gold_answer,
        original_answer=answer,
        original_correct=original.is_correct,
        chunk_roles=chunk_roles,
    )
