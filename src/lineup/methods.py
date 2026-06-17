from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from random import Random

from .backends.base import LanguageModel, Message
from .data.schema import ChunkScore, MethodPrediction, Scenario
from .prompt import build_messages_for, render_context

_WORD = re.compile(r"\w+")


def _tokens(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def _f1(predicted: set, reference: set) -> float:
    overlap = len(predicted & reference)
    if overlap == 0 or not predicted or not reference:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def _first_int(text: str) -> int | None:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def _rng_for(seed: int, qid: str) -> Random:
    digest = hashlib.sha256(f"{seed}:{qid}".encode()).digest()
    return Random(int.from_bytes(digest[:8], "big"))


def _log1mexp(x: float) -> float:
    # Numerically stable log(1 - exp(x)) for x < 0 (Maechler's split at -log 2).
    if x > -0.6931471805599453:
        return math.log(-math.expm1(x))
    return math.log1p(-math.exp(x))


def _logit_from_logprob(logprob: float) -> float:
    # The logit of the response probability, which is the target ContextCite regresses
    # against. Clamp away from zero so a fully confident response (p = 1) stays finite.
    clamped = min(logprob, -1e-6)
    return clamped - _log1mexp(clamped)


def _target(logprob: float, transform: str) -> float:
    return _logit_from_logprob(logprob) if transform == "logit" else logprob


class AttributionMethod(ABC):
    """A method under test: given the scenario and the model's answer, score every chunk
    so that a higher score means more responsible. The predicted culprit is the argmax."""

    name: str

    @abstractmethod
    def score_chunks(self, model: LanguageModel, scenario: Scenario, answer: str) -> list[float]:
        ...


class LexicalSimilarity(AttributionMethod):
    """A pure-salience baseline: token overlap between each chunk and the question and
    answer. It rates the near-miss highly because it is a near-duplicate of the gold
    evidence, so it is expected to blame the misleading chunk."""

    name = "lexical_similarity"

    def score_chunks(self, model, scenario, answer):
        query = _tokens(f"{scenario.question} {answer}")
        return [_f1(_tokens(chunk.text), query) for chunk in scenario.chunks]


class ContextCite(AttributionMethod):
    """ContextCite: sample random ablations of the context, teacher-force-score the fixed
    answer under each, and fit a sparse linear surrogate of the answer's (logit-transformed)
    probability against the inclusion mask. The Lasso weights are the per-chunk attribution."""

    name = "contextcite"

    def __init__(self, n_ablations: int = 32, alpha: float = 0.01, seed: int = 0, transform: str = "logit"):
        self.n_ablations = n_ablations
        self.alpha = alpha
        self.seed = seed
        self.transform = transform

    def score_chunks(self, model, scenario, answer):
        import numpy as np
        from sklearn.linear_model import Lasso

        chunks = scenario.chunks
        k = len(chunks)
        if not answer or k == 0:
            return [0.0] * k

        rng = _rng_for(self.seed, scenario.qid)
        masks = []
        targets = []
        for _ in range(self.n_ablations):
            mask = [rng.random() < 0.5 for _ in range(k)]
            included = [chunks[i] for i in range(k) if mask[i]]
            logprob = model.score(build_messages_for(scenario.question, included), answer).total_logprob
            masks.append([1.0 if bit else 0.0 for bit in mask])
            targets.append(_target(logprob, self.transform))

        surrogate = Lasso(alpha=self.alpha).fit(np.array(masks), np.array(targets))
        return [float(weight) for weight in surrogate.coef_]


class LLMJudgeCulprit(AttributionMethod):
    """Ask the model itself which numbered passage is most responsible for the answer."""

    name = "llm_judge"

    def score_chunks(self, model, scenario, answer):
        chunks = scenario.chunks
        prompt = (
            f"{render_context(chunks)}\n\nQuestion: {scenario.question}\n"
            f"A system answered: {answer}\n"
            "Which single numbered passage is most responsible for that answer? "
            "Reply with just the number."
        )
        messages = [
            Message("system", "You identify which passage a system relied on. Reply with one number."),
            Message("user", prompt),
        ]
        picked = _first_int(model.generate(messages, max_new_tokens=8).text)
        return [1.0 if picked == index + 1 else 0.0 for index in range(len(chunks))]


class SingleChunkSupport(AttributionMethod):
    """A cheap model-based salience baseline: score each chunk by the answer's log-probability
    when that chunk is the only passage in context. A chunk that on its own makes the answer
    likely looks responsible, whether or not it truly caused it, so like the lexical baseline
    it is expected to favour the near-miss."""

    name = "single_chunk"

    def score_chunks(self, model, scenario, answer):
        if not answer:
            return [0.0] * len(scenario.chunks)
        return [
            model.score(build_messages_for(scenario.question, [chunk]), answer).total_logprob
            for chunk in scenario.chunks
        ]


def run_method(
    method: AttributionMethod, model: LanguageModel, scenario: Scenario, answer: str
) -> MethodPrediction:
    scores = method.score_chunks(model, scenario, answer)
    chunk_scores = [
        ChunkScore(chunk.chunk_id, chunk.provenance, float(score))
        for chunk, score in zip(scenario.chunks, scores)
    ]
    if chunk_scores:
        top = max(score.score for score in chunk_scores)
        leaders = [index for index, score in enumerate(chunk_scores) if score.score == top]
        # Break ties without position bias: a deterministic per-(method, case) choice among
        # the top-scoring chunks, so a method that degenerates to all-equal scores does not
        # systematically blame whichever chunk happens to be presented first.
        predicted = chunk_scores[_rng_for(0, f"{method.name}:{scenario.qid}").choice(leaders)].chunk_id
    else:
        predicted = ""
    return MethodPrediction(
        qid=scenario.qid,
        method=method.name,
        predicted_culprit_id=predicted,
        chunk_scores=chunk_scores,
    )
