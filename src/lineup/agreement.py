from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .data.schema import CaseRoles

_ROLES = ("culprit", "misleading", "silent", "inert")


@dataclass
class AgreementReport:
    """How two models, run over the same cases, compare on where the error lands."""

    n_common: int                       # cases both runs cover
    n_both_wrong: int                   # cases both models answered wrongly
    wrong_rate_a: float
    wrong_rate_b: float
    same_culprit_rate: float | None     # of both-wrong cases, share with an identical culprit set
    culprit_jaccard: float | None       # mean overlap of the two culprit sets on both-wrong cases
    role_agreement: float | None        # of shared passages on both-wrong cases, share with the same role
    role_kappa: float | None            # Cohen's kappa on those role labels


def _culprit_ids(case: CaseRoles) -> set:
    return {role.chunk_id for role in case.chunk_roles if role.role == "culprit"}


def _role_map(case: CaseRoles) -> dict:
    return {role.chunk_id: role.role for role in case.chunk_roles}


def _cohen_kappa(pairs: list) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    marginal_a = {role: sum(1 for a, _ in pairs if a == role) / n for role in _ROLES}
    marginal_b = {role: sum(1 for _, b in pairs if b == role) / n for role in _ROLES}
    expected = sum(marginal_a[role] * marginal_b[role] for role in _ROLES)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def compare_models(cases_a: Iterable[CaseRoles], cases_b: Iterable[CaseRoles]) -> AgreementReport:
    by_a = {case.qid: case for case in cases_a}
    by_b = {case.qid: case for case in cases_b}
    common = sorted(set(by_a) & set(by_b))

    wrong_a = [qid for qid in common if not by_a[qid].original_correct]
    wrong_b = [qid for qid in common if not by_b[qid].original_correct]
    both_wrong = [qid for qid in common if not by_a[qid].original_correct and not by_b[qid].original_correct]

    same_culprit = []
    jaccards = []
    role_pairs = []
    for qid in both_wrong:
        set_a, set_b = _culprit_ids(by_a[qid]), _culprit_ids(by_b[qid])
        same_culprit.append(1.0 if set_a == set_b else 0.0)
        union = set_a | set_b
        jaccards.append(len(set_a & set_b) / len(union) if union else 1.0)
        roles_a, roles_b = _role_map(by_a[qid]), _role_map(by_b[qid])
        for chunk_id in set(roles_a) & set(roles_b):
            role_pairs.append((roles_a[chunk_id], roles_b[chunk_id]))

    n_common = len(common)
    return AgreementReport(
        n_common=n_common,
        n_both_wrong=len(both_wrong),
        wrong_rate_a=len(wrong_a) / n_common if n_common else 0.0,
        wrong_rate_b=len(wrong_b) / n_common if n_common else 0.0,
        same_culprit_rate=sum(same_culprit) / len(same_culprit) if same_culprit else None,
        culprit_jaccard=sum(jaccards) / len(jaccards) if jaccards else None,
        role_agreement=sum(1 for a, b in role_pairs if a == b) / len(role_pairs) if role_pairs else None,
        role_kappa=_cohen_kappa(role_pairs),
    )
