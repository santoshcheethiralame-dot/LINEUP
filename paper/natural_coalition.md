# The wrong primitive holds on un-planted (natural) coalitions

Natural slice (no planting). A *natural coalition* is a wrong case with >=2 causal non-gold
passages — distributed causation the model produces on its own. ContextCite, on those:

| cell | wrong | natural coalitions | recall@1 | recall@k |
|---|---:|---:|---:|---:|
| hotpotqa/qwen | 99 | 28 | 0.06 | 0.50 |
| 2wiki/qwen | 158 | 54 | 0.13 | 0.45 |
| hotpotqa/mistral | 68 | 38 | 0.09 | 0.52 |
| **pooled** | 325 | 120 | **0.10** | **0.48** |

Across the natural slice, **37%** of wrong cases are natural
coalitions, and on them a single pick recovers only **recall@1 0.10** of the responsible
set while a set of the right size recovers **recall@k 0.48** — the same structural gap as
the planted dose-response, with nothing constructed. The wrong-primitive finding is not an
artifact of our decoys; redundant causation occurs organically and a single pick can't cover it.