# No-culprit rate with 95% confidence intervals

| dataset | condition | model | n_wrong | no-culprit % [95% CI] |
|---|---|---|--:|--:|
| hotpotqa | baseline | Qwen-7B | 90 | 29% [20, 38] |
| hotpotqa | baseline | Phi-3.5 | 89 | 38% [28, 48] |
| hotpotqa | baseline | Mistral-7B | 73 | 49% [38, 61] |
| hotpotqa | hardtraps | Qwen-7B | 157 | 32% [25, 40] |
| hotpotqa | hardtraps | Phi-3.5 | 172 | 27% [20, 33] |
| hotpotqa | hardtraps | Mistral-7B | 104 | 46% [37, 56] |
| 2wiki | baseline | Qwen-7B | 158 | 28% [21, 35] |
| 2wiki | baseline | Phi-3.5 | 179 | 48% [41, 55] |
| 2wiki | baseline | Mistral-7B | 144 | 53% [45, 62] |
| 2wiki | hardtraps | Qwen-7B | 208 | 27% [21, 33] |
| 2wiki | hardtraps | Phi-3.5 | 220 | 34% [27, 40] |
| 2wiki | hardtraps | Mistral-7B | 172 | 41% [34, 49] |
