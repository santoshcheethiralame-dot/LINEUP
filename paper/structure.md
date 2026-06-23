# Does ill-posedness track reasoning structure?

### All wrong cases (pooled over models + conditions)

| dataset | question type | no-culprit % | 95% CI | n |
|---|---|---:|---:|---:|
| hotpotqa | bridge | 33.5 | [30, 37] | 638 |
| hotpotqa | comparison | 57.4 | [43, 70] | 47 |
| 2wiki | compositional | 36.3 | [33, 40] | 615 |
| 2wiki | inference | 27.8 | [22, 34] | 198 |
| 2wiki | bridge_comparison | 54.7 | [47, 62] | 172 |
| 2wiki | comparison | 37.5 | [28, 47] | 96 |

### Baseline only (no hard-traps planting)

| dataset | question type | no-culprit % | 95% CI | n |
|---|---|---:|---:|---:|
| hotpotqa | bridge | 36.1 | [30, 42] | 230 |
| hotpotqa | comparison | 59.1 | [39, 77] | 22 |
| 2wiki | compositional | 43.3 | [37, 50] | 247 |
| 2wiki | inference | 30.9 | [22, 41] | 94 |
| 2wiki | bridge_comparison | 56.2 | [46, 66] | 89 |
| 2wiki | comparison | 41.2 | [29, 55] | 51 |

Reading: comparison-style questions (hotpotqa *comparison*, 2wiki *bridge_comparison*) show a
markedly higher no-culprit rate than single-chain ones (2wiki *inference*, hotpotqa *bridge*).
Comparison questions need facts about two entities -> two necessary causes -> no single culprit:
ill-posedness is partly a fingerprint of multi-entity reasoning, not just our construction. The
baseline-only panel rules out a hard-traps artifact. Types with n<20 omitted. (Fig 11.)