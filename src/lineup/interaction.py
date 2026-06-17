from __future__ import annotations

from dataclasses import dataclass

from .backends.base import LanguageModel
from .data.schema import GenerationResult, Scenario
from .oracle import answer_key
from .prompt import build_messages_for


@dataclass
class InteractionPair:
    chunk_id_a: str
    chunk_id_b: str
    a_causal: bool         # removing a alone changes the answer
    b_causal: bool         # removing b alone changes the answer
    joint_causal: bool     # removing both changes the answer
    synergy: bool          # the pair changes the answer though neither chunk alone does


@dataclass
class CaseInteractions:
    qid: str
    pairs: list
    has_synergy: bool


def _key_without(model, scenario, removed, gold, intended_wrong) -> str:
    remaining = [chunk for index, chunk in enumerate(scenario.chunks) if index not in removed]
    answer = model.generate(build_messages_for(scenario.question, remaining)).text.strip()
    return answer_key(answer, gold, intended_wrong)


def pairwise_interactions(
    model: LanguageModel, scenario: Scenario, original: GenerationResult
) -> CaseInteractions:
    """Probe for coalition effects single-chunk leave-one-out cannot see.

    Leave-one-out misses a pair of passages that only matters jointly — each redundant with
    the other, so removing either alone leaves the answer unchanged, while removing both
    flips it. This checks every pair for that synergy, using the same value-based causal test
    as the oracle.
    """
    gold = scenario.gold_answer
    intended_wrong = scenario.recipe.intended_wrong_answer
    original_key = answer_key(original.model_answer, gold, intended_wrong)
    k = len(scenario.chunks)

    single_causal = [
        _key_without(model, scenario, {i}, gold, intended_wrong) != original_key for i in range(k)
    ]
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            joint_causal = _key_without(model, scenario, {i, j}, gold, intended_wrong) != original_key
            synergy = joint_causal and not single_causal[i] and not single_causal[j]
            pairs.append(
                InteractionPair(
                    chunk_id_a=scenario.chunks[i].chunk_id,
                    chunk_id_b=scenario.chunks[j].chunk_id,
                    a_causal=single_causal[i],
                    b_causal=single_causal[j],
                    joint_causal=joint_causal,
                    synergy=synergy,
                )
            )
    return CaseInteractions(scenario.qid, pairs, any(pair.synergy for pair in pairs))
