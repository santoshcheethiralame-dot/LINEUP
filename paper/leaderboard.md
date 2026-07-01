# LINEUP leaderboard

Headline metrics for each attribution method, pooled across all 12 cells (3 model families x 2 datasets x 2 conditions), wrong cases only. Denominators differ by column: culprit acc is localization *given* a single culprit exists (the well-posed subset, not all errors); recall@1 vs recall@k is coverage of the planted responsible set -- the single-pick ceiling vs the set. Regenerate with `python scripts/build_leaderboard.py`; add your own method via `lineup.evaluate.evaluate`.

| method | culprit acc (well-posed) | recall@1 | recall@k | reliability AUROC | n well-posed | n wrong |
|---|---:|---:|---:|---:|---:|---:|
| contextcite | 0.83 | 0.30 | 0.50 | 0.72 | 1117 | 1766 |
| single_chunk | 0.79 | 0.32 | 0.54 | 0.75 | 1117 | 1766 |
| llm_judge | 0.70 | 0.25 | 0.34 | 0.50 | 1117 | 1766 |
| lexical_similarity | 0.37 | 0.10 | 0.19 | 0.57 | 1117 | 1766 |

No-culprit rate (a property of the benchmark, not any method): **37%** of wrong cases have no single culprit.
