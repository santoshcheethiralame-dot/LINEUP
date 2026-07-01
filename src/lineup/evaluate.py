"""Public evaluation API: score any attribution method against the LINEUP oracle roles in one call.

Typical use (bring your own method):

    from lineup.data.serialization import read_roles
    from lineup.evaluate import evaluate, predictions_from_scores, leaderboard_markdown

    cases = read_roles("roles.jsonl")                       # the benchmark's ground-truth roles
    scores_by_qid = {c.qid: my_method(c.question, ...) for c in cases}   # {qid: {chunk_id: score}}
    preds = predictions_from_scores("my_method", scores_by_qid)
    print(leaderboard_markdown(evaluate(cases, preds)))

The metrics returned are the headline benchmark numbers: top-1 culprit accuracy (does it find the
culprit when one exists?), recall@1 vs recall@k (the single-pick ceiling vs the set), and the two
confidence AUROCs (does the method know when it is right / when no single culprit exists?).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .data.schema import CaseRoles, ChunkScore, MethodPrediction
from .scoring import score_predictions
from .setvalued import attribution_recovery

LEADERBOARD_COLUMNS = (
    ("top1_culprit_accuracy", "culprit acc (well-posed)"),
    ("recall_at_1", "recall@1"),
    ("recall_at_k", "recall@k"),
    ("reliability_auroc", "reliability AUROC"),
)


@dataclass
class MethodScore:
    method: str
    n_wrong: int
    n_with_culprit: int
    top1_culprit_accuracy: float | None   # over cases with a culprit: does it find it?
    recall_at_1: float | None             # single pick's coverage of the responsible set
    recall_at_k: float | None             # top-|responsible| set's coverage (the set ceiling)
    reliability_auroc: float | None       # does the score margin flag the method's own errors?
    single_culprit_auroc: float | None    # does the score margin flag cases with no single culprit?
    misleading_as_culprit_rate: float     # how often it blames the salient red-herring


def no_culprit_rate(cases: Iterable[CaseRoles]) -> float:
    """Share of (wrong) cases with no single culprit -- a property of the benchmark, not a method."""
    cases = list(cases)
    if not cases:
        return float("nan")
    return sum(1 for c in cases if not any(r.role == "culprit" for r in c.chunk_roles)) / len(cases)


def predictions_from_scores(method: str, scores_by_qid: Mapping[str, Mapping[str, float]]) -> list[MethodPrediction]:
    """Build predictions from the natural output of an attribution method: {qid: {chunk_id: score}}.
    The top-scored chunk per case becomes the predicted culprit."""
    preds = []
    for qid, scores in scores_by_qid.items():
        if not scores:
            continue
        top = max(scores, key=lambda cid: scores[cid])
        preds.append(
            MethodPrediction(
                qid=qid,
                method=method,
                predicted_culprit_id=top,
                chunk_scores=[ChunkScore(chunk_id=cid, provenance="", score=float(s)) for cid, s in scores.items()],
            )
        )
    return preds


def evaluate(cases: Iterable[CaseRoles], predictions: Iterable[MethodPrediction]) -> list[MethodScore]:
    """Score one or more attribution methods against the oracle roles, joined by question id.

    Pass the wrong cases (from read_roles) and your method's predictions; returns one MethodScore
    per method, sorted by top-1 culprit accuracy. Reuses the same scoring used for the paper, so
    your numbers are directly comparable to the published leaderboard.
    """
    cases = list(cases)
    predictions = list(predictions)
    base = {r.method: r for r in score_predictions(cases, predictions)}
    recov = {r.method: r for r in attribution_recovery(cases, predictions)}
    scores = []
    for method, b in base.items():
        rc = recov.get(method)
        scores.append(
            MethodScore(
                method=method,
                n_wrong=b.n_cases,
                n_with_culprit=b.n_with_culprit,
                top1_culprit_accuracy=b.top1_culprit_accuracy,
                recall_at_1=rc.recall_at_1 if rc else None,
                recall_at_k=rc.recall_at_k if rc else None,
                reliability_auroc=rc.reliability_auroc if rc else None,
                single_culprit_auroc=rc.single_culprit_auroc if rc else None,
                misleading_as_culprit_rate=b.misleading_as_culprit_rate,
            )
        )
    return sorted(scores, key=lambda s: -(s.top1_culprit_accuracy or 0.0))


def leaderboard_markdown(scores: Iterable[MethodScore], no_culprit: float | None = None) -> str:
    """Render scores as a markdown leaderboard table."""
    scores = list(scores)
    header = "| method | " + " | ".join(label for _, label in LEADERBOARD_COLUMNS) + " | n well-posed | n wrong |"
    rule = "|---|" + "---:|" * (len(LEADERBOARD_COLUMNS) + 2)

    def fmt(x):
        return "--" if x is None else f"{x:.2f}"

    lines = [header, rule]
    for s in scores:
        cells = " | ".join(fmt(getattr(s, attr)) for attr, _ in LEADERBOARD_COLUMNS)
        lines.append(f"| {s.method} | {cells} | {s.n_with_culprit} | {s.n_wrong} |")
    out = "\n".join(lines)
    if no_culprit is not None:
        out += (
            f"\n\nNo-culprit rate (a property of the benchmark, not any method): "
            f"**{100 * no_culprit:.0f}%** of wrong cases have no single culprit."
        )
    return out
