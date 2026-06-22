from __future__ import annotations

from .backends.base import LanguageModel
from .correctness import judge_correct
from .data.schema import CaseRoles, Chunk, ChunkRole, GenerationResult, Scenario
from .prompt import build_messages_for
from .textnorm import contains_phrase, normalize


def answer_key(answer: str, gold: str, intended_wrong: str) -> str:
    """A canonical key for the value an answer carries, so that phrasing variants of the
    same value compare equal — a verbose "the designer was X" and a bare "X" are one
    value, and removing a chunk that only triggers a rephrase is not mistaken for causal."""
    has_gold = contains_phrase(answer, gold)
    has_wrong = bool(intended_wrong) and contains_phrase(answer, intended_wrong)
    if has_gold and has_wrong:
        # The answer carries both values because the planted wrong value embeds the gold
        # tokens as a sub-run (e.g. gold "D.C." inside wrong "Washington DC"). The more
        # specific value — the one with more tokens — is the one the answer actually asserts.
        return "wrong" if len(normalize(intended_wrong).split()) >= len(normalize(gold).split()) else "gold"
    if has_gold:
        return "gold"
    if has_wrong:
        return "wrong"
    return normalize(answer)


def is_salient(chunk: Chunk, answer: str, gold: str, intended_wrong: str) -> bool:
    """Does the chunk look responsible for the answer? It contains the answer outright, or
    the canonical value (gold or planted-wrong) that a verbose answer carries."""
    if contains_phrase(chunk.text, answer):
        return True
    if contains_phrase(answer, gold) and contains_phrase(chunk.text, gold):
        return True
    if intended_wrong and contains_phrase(answer, intended_wrong) and contains_phrase(chunk.text, intended_wrong):
        return True
    return False


def assign_role(causal: bool, salient: bool) -> str:
    if causal and salient:
        return "culprit"
    if salient:
        return "misleading"
    if causal:
        return "silent"
    return "inert"


def leave_one_out(
    model: LanguageModel, scenario: Scenario, original: GenerationResult, *, llm_judge=None, score_logprobs: bool = True
) -> CaseRoles:
    """Assign every chunk its 2x2 role by exact leave-one-out.

    The causal axis is whether removing the chunk changes the value of the model's answer;
    the salience axis is whether the chunk holds that answer's value. Both are derived from
    the known gold answer and the model's own behavior, never from a method under test, so
    the labels are non-circular.

    ``score_logprobs`` controls only the informational ``delta_logprob`` field; set it False for
    generation-only backends (the API models), which cannot teacher-force a fixed answer. The role
    assignment is unaffected — it depends on generation, not scoring.
    """
    chunks = scenario.chunks
    answer = original.model_answer
    gold = scenario.gold_answer
    intended_wrong = scenario.recipe.intended_wrong_answer
    full_messages = build_messages_for(scenario.question, chunks)
    full_logprob = model.score(full_messages, answer).total_logprob if (answer and score_logprobs) else 0.0
    original_key = answer_key(answer, gold, intended_wrong)

    chunk_roles = []
    for index, chunk in enumerate(chunks):
        remaining = [c for j, c in enumerate(chunks) if j != index]
        ablated_messages = build_messages_for(scenario.question, remaining)
        answer_without = model.generate(ablated_messages).text.strip()

        causal = answer_key(answer_without, gold, intended_wrong) != original_key
        salient = is_salient(chunk, answer, gold, intended_wrong)
        now_correct, _ = judge_correct(scenario.question, gold, answer_without, llm_judge)
        ablated_logprob = model.score(ablated_messages, answer).total_logprob if (answer and score_logprobs) else 0.0

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
        gold_answer=gold,
        original_answer=answer,
        original_correct=original.is_correct,
        chunk_roles=chunk_roles,
    )
