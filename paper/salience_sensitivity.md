# No-culprit rate vs. the salience definition

Pooled wrong cases (12 cells): **1766**. No single culprit = no causal passage that is
also salient, under each salience rule:

| salience rule | no-culprit % |
|---|---:|
| STRICT (answer appears verbatim) | 42.7 |
| shipped (normalized phrase match) | 36.7 |
| LOOSE (any 4+ char answer token)  | 19.3 |

**Honest reading:** the rate is matcher-dependent, **[19%, 43%]**.
Stricter salience marks fewer passages salient (more no-culprit); the LOOSE token-overlap rule is
over-generous (one shared word like "protocol" marks a passage salient) and is the floor. The
shipped phrase-level matcher (37%) is the principled choice — it matches whole values, not stray
tokens. The finding is **directionally robust** (always substantial; >=19% even under an
over-generous rule) but the precise value depends on the salience definition, which we disclose
as the one soft axis. We report the phrase-level number and this band.