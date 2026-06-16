from __future__ import annotations

from typing import Sequence

from .backends.base import Message
from .data.schema import Chunk, Scenario

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Reply with the shortest possible answer — a name, date, or phrase — and nothing else."
)


def render_context(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {chunk.title}\n{chunk.text}" for i, chunk in enumerate(chunks, 1))


def build_messages(scenario: Scenario) -> list[Message]:
    user = f"Context:\n{render_context(scenario.chunks)}\n\nQuestion: {scenario.question}"
    return [Message("system", SYSTEM_PROMPT), Message("user", user)]
