# Natural-retrieval slice: ill-posedness is not a construction artifact

Organic errors from each dataset's own distractors, with **no planted near-miss** and no decoys.
The no-culprit rate tracks the constructed baseline in every cell -- and reproduces the
cross-family ordering (Mistral more ill-posed than Qwen).

| cell | natural no-culprit % | constructed no-culprit % | natural n |
|---|---:|---:|---:|
| hotpotqa/qwen | 28 | 29 | 99 |
| 2wiki/qwen | 26 | 28 | 158 |
| hotpotqa/mistral | 54 | 49 | 68 |

Zero planting, same ill-posedness, across 2 datasets and 2 model families. (Fig 12.)