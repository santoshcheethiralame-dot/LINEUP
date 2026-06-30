# Construction-recovery validation (no humans, no LLM judge, non-circular)

The oracle never sees `provenance`; it derives causal/salient purely from leave-one-out
ablation. `provenance` is the construction ground truth. Agreement between the two is a
validity check whose ground truth comes from the data generator -- not the oracle's own
signal, and not an LLM judge (which this paper argues is unreliable). All rows are wrong
cases; the oracle only labels errors. Rates carry 95% Wilson intervals.

## 1. The oracle recovers the designed roles
Pooled provenance x oracle-role counts (row % in brackets):

| provenance \ oracle role | culprit | misleading | silent | inert | row total |
|---|---:|---:|---:|---:|---:|
| gold | 512 (13%) | 407 (11%) | 1214 (31%) | 1743 (45%) | 3876 |
| misleading | 424 (24%) | 504 (29%) | 359 (20%) | 479 (27%) | 1766 |
| decoy | 441 (43%) | 223 (22%) | 156 (15%) | 213 (21%) | 1033 |
| distractor | 105 (3%) | 104 (3%) | 1152 (29%) | 2560 (65%) | 3921 |

The planted **misleading** and **decoy** passages carry the salient roles (culprit + misleading):
**53%** and **64%** of them, against just **5%** of distractors.
Distractors are overwhelmingly non-salient and only 3% culprit; gold, ignored on these error
cases, is mostly inert/silent. The oracle reconstructs which passages carry the wrong value
without ever seeing provenance.

## 2. Specificity: the oracle does not blame filler
- Distractor passages labelled **inert**: 65% [64-67] (n=3921)
- Distractor passages the oracle calls **causal** (false cause): 32% [31-34] (n=3921)
- Distractor passages the oracle calls **salient**: 5% [5-6] (n=3921)
- Of every passage the oracle calls **culprit**, share that are unplanted distractors: 7% [6-9] (n=1482)

Note the honest split: a distractor is **often causal** (removing any passage perturbs a model
that is already answering wrongly), yet **almost never salient**. Because the culprit label
needs *both*, blame on filler collapses to the 7% above. This is direct evidence for the 2x2
itself -- the salience axis filters the noise in the raw causal axis: causal-alone would blame a
third of the filler; the gate brings it to single digits.

## 3. Manipulation check: the oracle detects *designed* redundancy
Hard-traps swaps one distractor for a decoy asserting the **same** wrong value as the
misleading chunk -- a redundant cause, planted on purpose (context size is fixed at 6 either
way, so this is a swap, not an extra passage). Prediction: with two sufficient causes, removing
the misleading chunk alone flips the answer **less** often, and **no single** passage is causal
**more** often. Conditioned on the bait being taken (the misleading chunk is salient):

| | baseline (1 planted cause) | hard-traps (2 planted causes) |
|---|---|---|
| misleading chunk individually causal | 58% [53-64] (n=281) | 40% [36-44] (n=647) |
| no single passage is causal | 14% [10-18] (n=281) | 17% [14-20] (n=647) |

Adding the designed redundant cause drops the misleading chunk's individual-causal rate by
**18 points** (58% -> 40%, intervals do not overlap) -- the textbook
signature of redundancy: a second sufficient cause strips the first of its individual necessity,
and the oracle registers it from ablation alone, against a structure change it was never told
about. (The no-single-cause rate rises only 3 points, within noise: the planted redundancy
mostly shows up as **over-determination** -- causality shifts onto the decoy, itself culprit
43% of the time vs the misleading chunk's 24% -- rather than as a leave-one-out-blind
coalition. We report the significant signal, individual necessity, not the no-single count.)
