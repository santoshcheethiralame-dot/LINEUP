# Retrieval-depth sweep (natural slice, HotpotQA/qwen)

| k | cases | wrong | error rate % | no-culprit % |
|---:|---:|---:|---:|---:|
| 4 | 370 | 90 | 24 | 23 |
| 6 | 370 | 99 | 27 | 28 |
| 8 | 370 | 109 | 29 | 23 |
| 10 | 370 | 124 | 34 | 23 |

As retrieval depth grows from 4 to 10, the **error rate climbs** (24% -> 34%)
-- more passages give the model more ways to go wrong -- but the **no-culprit fraction is flat**
([23%, 28%]). Ill-posedness is therefore not an artifact of how much you retrieve;
a steady ~quarter of errors have no single culprit regardless of context size. (Fig 14, appendix.)