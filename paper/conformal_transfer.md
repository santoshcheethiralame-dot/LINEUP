# Conformal beats a fixed-k heuristic under dataset shift

Target coverage 0.90. A set size calibrated on the source is transferred unchanged to the
target (the fixed-k heuristic); conformal instead recalibrates on the target. Same test split.

| shift | source | target | fixed size from source | coverage on target | conformal tau | coverage |
|---|---|---|---:|---:|---:|---:|
| dataset | hotpotqa | 2wiki | 2 | 0.92 | 2 | 0.92 |
| dataset | 2wiki | hotpotqa | 2 | 0.96 | 2 | 0.96 |
| redundancy | baseline | hardtraps | 2 | 0.93 | 3 | 0.96 |
| redundancy | hardtraps | baseline | 2 | 0.94 | 2 | 0.94 |

Honest reading: on the single-culprit target a fixed top-2 stays above the 0.90 target across
every shift here, so conformal does not *empirically* beat the heuristic on this data. We do not
manufacture a shift that breaks it. Conformal's value is therefore the distribution-free guarantee
(it holds by construction without knowing the shift) and the self-reporting set size: tau rises
2 -> 3 as redundancy increases (baseline -> hard-traps), and to the full context to cover an entire
responsible coalition (run_conformal.py). We claim the guarantee and the adaptive size, not an
empirical coverage win a fixed top-k cannot match.
