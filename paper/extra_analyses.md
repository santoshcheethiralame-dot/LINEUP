# Appendix: position and graded-correctness checks

## Position effect (ContextCite top-1 by culprit position)
Culprit position is randomized by construction, so this tests attribution's positional bias.

| position | top-1 culprit acc | n |
|---|---:|---:|
| early | 0.81 | 221 |
| middle | 0.77 | 286 |
| late | 0.81 | 294 |

Spread across thirds: **0.04**. Small — no strong 'lost in the middle' effect for attribution.

## Graded correctness (no-culprit rate by error type)
A *near-miss* error shares a content token with the gold answer; a *total-miss* does not.

| error type | no-culprit % | n |
|---|---:|---:|
| near-miss | 46 | 333 |
| total-miss | 35 | 1433 |

Difference: **11 points**. No-culprit rate differs by error type.