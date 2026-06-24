# LINEUP leaderboard

Headline metrics for each attribution method, pooled across all 12 cells (3 model families x 2 datasets x 2 conditions), wrong cases only. recall@1 is the single-pick ceiling; recall@k is the set. Regenerate with `python scripts/build_leaderboard.py`; add your own method via `lineup.evaluate.evaluate`.

| method | top-1 culprit acc | recall@1 | recall@k | reliability AUROC | n |
|---|---:|---:|---:|---:|---:|
| contextcite | 0.83 | 0.30 | 0.50 | 0.72 | 1766 |
| single_chunk | 0.79 | 0.32 | 0.54 | 0.75 | 1766 |
| llm_judge | 0.70 | 0.25 | 0.34 | 0.50 | 1766 |
| lexical_similarity | 0.37 | 0.10 | 0.19 | 0.57 | 1766 |

No-culprit rate (a property of the benchmark, not any method): **37%** of wrong cases have no single culprit.
