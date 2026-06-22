from __future__ import annotations

import os
import time
from typing import Sequence

from .base import Generation, LanguageModel, Message, Scoring

# Known OpenAI-compatible endpoints with free tiers. Any base_url that speaks the chat-completions
# API works; pass --base-url to use another.
PROVIDERS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "cerebras": "https://api.cerebras.ai/v1",
    "openai": "https://api.openai.com/v1",
}


class APIModel(LanguageModel):
    """Generation-only backend for OpenAI-compatible chat APIs (Groq, Gemini, OpenAI).

    These endpoints do not return teacher-forced logprobs for a supplied answer, so ``score`` is
    unavailable and the logprob attribution methods (ContextCite, SingleChunkSupport) cannot run.
    The counterfactual oracle (call it with ``score_logprobs=False``), LexicalSimilarity, and
    LLMJudgeCulprit all use only ``generate`` and work fine. Used for the free frontier-scale slice.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client=None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        max_retries: int = 8,
        min_interval: float = 0.0,
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_interval = min_interval   # seconds between calls, to stay under a per-minute cap
        self._last_call = 0.0
        if client is not None:
            self.client = client
            return
        key = api_key or os.environ.get("LINEUP_API_KEY")
        if not key:
            raise ValueError("set the LINEUP_API_KEY environment variable (or pass api_key=)")
        from openai import OpenAI  # optional dependency, imported only when actually calling an API

        self.client = OpenAI(base_url=base_url, api_key=key)

    def _chat(self, messages: Sequence[Message], max_new_tokens: int | None) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        delay = 2.0
        for attempt in range(self.max_retries):
            if self.min_interval:        # proactively stay under a per-minute request cap
                wait = self.min_interval - (time.monotonic() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    temperature=self.temperature,
                    max_tokens=max_new_tokens or self.max_new_tokens,
                )
                self._last_call = time.monotonic()
                message = response.choices[0].message
                text = (message.content or "").strip()
                if "</think>" in text:        # some reasoning models inline their thinking
                    text = text.split("</think>")[-1].strip()
                if not text:                  # others leave content empty and put it in a side field
                    text = (getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or "").strip()
                return text
            except Exception:
                self._last_call = time.monotonic()
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)        # back off on rate limits / transient queue errors
                delay = min(delay * 2, 60)
        return ""

    def generate(self, messages: Sequence[Message], max_new_tokens: int | None = None) -> Generation:
        text = self._chat(messages, max_new_tokens)
        return Generation(text=text.strip(), token_ids=[], token_logprobs=[], truncated=False)

    def score(self, messages: Sequence[Message], response: str) -> Scoring:
        raise NotImplementedError(
            "API models do not expose teacher-forced logprobs; run the oracle with "
            "score_logprobs=False and use the LexicalSimilarity / LLMJudgeCulprit methods only."
        )
