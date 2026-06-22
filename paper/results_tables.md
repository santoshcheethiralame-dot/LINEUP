# LINEUP results

Matrix coverage: 12/12 cells. Regenerate with `python scripts/build_results.py`.

## Table 1 — Attribution accuracy (top-1 culprit, wrong cases)

| dataset | condition | model | n_wrong | contextcite | single_chunk | llm_judge | lexical_similarity |
|---|---|---|--:|--:|--:|--:|--:|
| hotpotqa | baseline | Qwen2.5-7B | 90 | 0.92 | 0.95 | 0.72 | 0.52 |
| hotpotqa | baseline | Phi-3.5-mini | 89 | 0.82 | 0.80 | 0.65 | 0.53 |
| hotpotqa | baseline | Mistral-7B | 73 | 0.86 | 0.76 | 0.76 | 0.57 |
| hotpotqa | hardtraps | Qwen2.5-7B | 157 | 0.89 | 0.88 | 0.76 | 0.41 |
| hotpotqa | hardtraps | Phi-3.5-mini | 172 | 0.84 | 0.78 | 0.70 | 0.39 |
| hotpotqa | hardtraps | Mistral-7B | 104 | 0.80 | 0.64 | 0.68 | 0.39 |
| 2wiki | baseline | Qwen2.5-7B | 158 | 0.89 | 0.85 | 0.75 | 0.35 |
| 2wiki | baseline | Phi-3.5-mini | 179 | 0.87 | 0.83 | 0.63 | 0.42 |
| 2wiki | baseline | Mistral-7B | 144 | 0.75 | 0.67 | 0.76 | 0.34 |
| 2wiki | hardtraps | Qwen2.5-7B | 208 | 0.84 | 0.84 | 0.75 | 0.30 |
| 2wiki | hardtraps | Phi-3.5-mini | 220 | 0.76 | 0.75 | 0.58 | 0.30 |
| 2wiki | hardtraps | Mistral-7B | 172 | 0.70 | 0.62 | 0.69 | 0.22 |

## Table 2 — Ill-posedness and set recovery (ContextCite)

no-culprit% = errors with no single causal culprit. recall@1 vs recall@k = single pick vs top-|R| set.

| dataset | condition | model | no-culprit% | recall@1 | recall@k | reliability AUROC | single-culprit AUROC |
|---|---|---|--:|--:|--:|--:|--:|
| hotpotqa | baseline | Qwen2.5-7B | 29% | 0.41 | 0.41 | 0.83 | 0.75 |
| hotpotqa | baseline | Phi-3.5-mini | 38% | 0.30 | 0.30 | 0.80 | 0.66 |
| hotpotqa | baseline | Mistral-7B | 49% | 0.49 | 0.49 | 0.92 | 0.68 |
| hotpotqa | hardtraps | Qwen2.5-7B | 32% | 0.33 | 0.68 | 0.63 | 0.66 |
| hotpotqa | hardtraps | Phi-3.5-mini | 27% | 0.35 | 0.74 | 0.67 | 0.66 |
| hotpotqa | hardtraps | Mistral-7B | 46% | 0.37 | 0.75 | 0.60 | 0.53 |
| 2wiki | baseline | Qwen2.5-7B | 28% | 0.23 | 0.23 | 0.83 | 0.69 |
| 2wiki | baseline | Phi-3.5-mini | 48% | 0.23 | 0.23 | 0.72 | 0.56 |
| 2wiki | baseline | Mistral-7B | 53% | 0.22 | 0.22 | 0.79 | 0.55 |
| 2wiki | hardtraps | Qwen2.5-7B | 27% | 0.29 | 0.64 | 0.77 | 0.72 |
| 2wiki | hardtraps | Phi-3.5-mini | 34% | 0.27 | 0.57 | 0.68 | 0.51 |
| 2wiki | hardtraps | Mistral-7B | 41% | 0.26 | 0.57 | 0.58 | 0.48 |

## Table 3 — Cross-model agreement (per-passage role kappa, both-wrong cases)

| dataset | condition | qwen-vs-phi | qwen-vs-mistral | phi-vs-mistral |
|---|---|--:|--:|--:|
| hotpotqa | baseline | 0.35 | 0.47 | 0.36 |
| hotpotqa | hardtraps | 0.41 | 0.42 | 0.42 |
| 2wiki | baseline | 0.14 | 0.26 | 0.13 |
| 2wiki | hardtraps | 0.32 | 0.31 | 0.31 |
