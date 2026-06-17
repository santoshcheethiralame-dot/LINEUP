from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

from .data.schema import CaseRoles, MethodPrediction, Scenario
from .scoring import score_predictions


def breakdown(
    scenarios: Iterable[Scenario],
    cases: Iterable[CaseRoles],
    predictions: Iterable[MethodPrediction],
    key: Callable,
) -> dict:
    """Group cases by ``key(scenario, case)`` and score every group separately.

    A key of None drops the case (for example a case with no culprit when grouping by the
    culprit's position). Returns {group_label: [MethodReport, ...]}.
    """
    scenarios_by_qid = {scenario.qid: scenario for scenario in scenarios}
    groups: dict = defaultdict(list)
    for case in cases:
        scenario = scenarios_by_qid.get(case.qid)
        if scenario is None:
            continue
        label = key(scenario, case)
        if label is not None:
            groups[label].append(case)
    predictions = list(predictions)
    return {label: score_predictions(groups[label], predictions) for label in sorted(groups)}


def by_substitution_type(scenarios, cases, predictions) -> dict:
    """How each method does on year vs number vs entity near-misses."""
    return breakdown(scenarios, cases, predictions, lambda scenario, case: scenario.recipe.substitution_type)


def _culprit_position(scenario: Scenario, case: CaseRoles):
    culprits = {role.chunk_id for role in case.chunk_roles if role.role == "culprit"}
    for position, chunk in enumerate(scenario.chunks):
        if chunk.chunk_id in culprits:
            return position
    return None


def by_culprit_position(scenarios, cases, predictions) -> dict:
    """How each method does as a function of where the culprit sits in the context."""
    return breakdown(scenarios, cases, predictions, _culprit_position)
