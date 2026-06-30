# Oracle-supervised learned baseline (upper bound on single-passage attribution)

A gradient-boosted ranker over every method's per-chunk score + structural features (position,
retrieval score), trained on the oracle culprit labels with grouped 5-fold CV (groups = qid).
It sees the answer key in training, so it bounds what any deployable single-passage method could do.

**Methods stacked:** contextcite, lexical_similarity, llm_judge, single_chunk.

## Culprit accuracy (top-1 = culprit), well-posed cases (n=1117)
| attributor | top-1 culprit accuracy |
|---|--:|
| contextcite | 0.825 |
| lexical_similarity | 0.366 |
| llm_judge | 0.700 |
| single_chunk | 0.786 |
| learned (stack) | 0.848 |

The oracle-trained stack tops out at **0.848**, a **+0.022** lift over the best single method (contextcite, 0.825). A learned method helps *where a culprit exists* — but only modestly.

## Responsible-set recall, coalition cases |R|>=2 (n=1033, 1/m ceiling = 0.500)
| attributor | recall@1 | recall@k |
|---|--:|--:|
| contextcite | 0.305 | 0.646 |
| lexical_similarity | 0.100 | 0.262 |
| llm_judge | 0.268 | 0.422 |
| single_chunk | 0.316 | 0.690 |
| learned (stack) | 0.299 | 0.651 |

On coalitions the oracle-trained stack reaches recall@1 = **0.299** — no better than the best heuristic (single_chunk, 0.316) and still under the structural bound recall@1 ≤ 1/m = 0.500 that no single pick can exceed. Training on the labels neither beats the methods nor approaches the bound: which of m redundant chunks to name is not a property of any one chunk, so no per-chunk feature — learned or hand-built — carries it. The limit is the single-passage primitive, shown here against a method that trains on the answer key itself.
