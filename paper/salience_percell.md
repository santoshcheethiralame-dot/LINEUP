# Per-cell no-culprit rate by salience matcher (Wilson 95% CIs)

No single culprit = no causal passage that is also salient. STRICT = answer verbatim; shipped =
phrase match (reported headline); LOOSE = any shared 4+ char token (over-generous floor).

| cell | n | shipped % [95% CI] | loose % (floor) [95% CI] |
|---|--:|--:|--:|
| hotpotqa/baseline/qwen | 90 | 29 [21–39] | 29 [21–39] |
| hotpotqa/baseline/phi | 89 | 38 [29–49] | 22 [15–32] |
| hotpotqa/baseline/mistral | 73 | 49 [38–61] | 16 [10–27] |
| hotpotqa/hardtraps/qwen | 157 | 32 [26–40] | 36 [29–43] |
| hotpotqa/hardtraps/phi | 172 | 27 [21–34] | 18 [13–24] |
| hotpotqa/hardtraps/mistral | 104 | 46 [37–56] | 28 [20–37] |
| 2wiki/baseline/qwen | 158 | 28 [21–35] | 20 [14–27] |
| 2wiki/baseline/phi | 179 | 48 [41–55] | 9 [6–14] |
| 2wiki/baseline/mistral | 144 | 53 [45–61] | 14 [9–20] |
| 2wiki/hardtraps/qwen | 208 | 27 [21–33] | 25 [19–31] |
| 2wiki/hardtraps/phi | 220 | 34 [28–40] | 12 [8–17] |
| 2wiki/hardtraps/mistral | 172 | 41 [34–49] | 13 [9–19] |
| **pooled** | 1766 | **37 [35–39]** | **19 [17–21]** |

Pooled band: strict 43%, shipped 37%, loose 19%.
**Survival:** every cell's LOOSE-matcher lower CI is the strongest test — the floor under the most
over-generous salience rule. If those stay clear of 0, the no-single-culprit finding survives the
harshest reading of the soft axis.
