# LINEUP results

Matrix coverage: 4/12 cells. Regenerate with `python scripts/build_results.py`.

## Table 1 — Attribution accuracy (top-1 culprit, wrong cases)

| dataset | condition | model | n_wrong | contextcite | single_chunk | llm_judge | lexical_similarity |
|---|---|---|--:|--:|--:|--:|--:|
| hotpotqa | baseline | Qwen2.5-7B | 90 | 0.92 | 0.95 | 0.72 | 0.52 |
| hotpotqa | baseline | Phi-3.5-mini | 89 | 0.82 | 0.80 | 0.65 | 0.53 |
| hotpotqa | baseline | Qwen2.5-14B | pending | — | — | — | — |
| hotpotqa | hardtraps | Qwen2.5-7B | 157 | 0.89 | 0.88 | 0.76 | 0.41 |
| hotpotqa | hardtraps | Phi-3.5-mini | 172 | 0.84 | 0.78 | 0.70 | 0.39 |
| hotpotqa | hardtraps | Qwen2.5-14B | pending | — | — | — | — |
| 2wiki | baseline | Qwen2.5-7B | pending | — | — | — | — |
| 2wiki | baseline | Phi-3.5-mini | pending | — | — | — | — |
| 2wiki | baseline | Qwen2.5-14B | pending | — | — | — | — |
| 2wiki | hardtraps | Qwen2.5-7B | pending | — | — | — | — |
| 2wiki | hardtraps | Phi-3.5-mini | pending | — | — | — | — |
| 2wiki | hardtraps | Qwen2.5-14B | pending | — | — | — | — |

## Table 2 — Ill-posedness and set recovery (ContextCite)

no-culprit% = errors with no single causal culprit. recall@1 vs recall@k = single pick vs top-|R| set.

| dataset | condition | model | no-culprit% | recall@1 | recall@k | reliability AUROC | single-culprit AUROC |
|---|---|---|--:|--:|--:|--:|--:|
| hotpotqa | baseline | Qwen2.5-7B | 29% | 0.41 | 0.41 | 0.83 | 0.75 |
| hotpotqa | baseline | Phi-3.5-mini | 38% | 0.30 | 0.30 | 0.80 | 0.66 |
| hotpotqa | baseline | Qwen2.5-14B | pending | — | — | — | — |
| hotpotqa | hardtraps | Qwen2.5-7B | 32% | 0.33 | 0.68 | 0.63 | 0.66 |
| hotpotqa | hardtraps | Phi-3.5-mini | 27% | 0.35 | 0.74 | 0.67 | 0.66 |
| hotpotqa | hardtraps | Qwen2.5-14B | pending | — | — | — | — |
| 2wiki | baseline | Qwen2.5-7B | pending | — | — | — | — |
| 2wiki | baseline | Phi-3.5-mini | pending | — | — | — | — |
| 2wiki | baseline | Qwen2.5-14B | pending | — | — | — | — |
| 2wiki | hardtraps | Qwen2.5-7B | pending | — | — | — | — |
| 2wiki | hardtraps | Phi-3.5-mini | pending | — | — | — | — |
| 2wiki | hardtraps | Qwen2.5-14B | pending | — | — | — | — |

## Table 3 — Cross-model agreement (per-passage role kappa, both-wrong cases)

| dataset | condition | qwen-vs-phi | qwen-vs-qwen14 | phi-vs-qwen14 |
|---|---|--:|--:|--:|
| hotpotqa | baseline | 0.35 | pending | pending |
| hotpotqa | hardtraps | 0.41 | pending | pending |
| 2wiki | baseline | pending | pending | pending |
| 2wiki | hardtraps | pending | pending | pending |
