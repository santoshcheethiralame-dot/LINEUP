# Oracle robustness & no-culprit taxonomy

Pooled wrong cases (all 12 cells): **1766**. As-shipped no-culprit rate: **36.7%**.

## 1. The causal axis is threshold-free, and an independent signal confirms it
The oracle marks a chunk causal by a *discrete counterfactual* (removing it flips the answer's
value); `delta_logprob` (the leave-one-out drop in the answer's logprob) is recorded but never
used for the label. The two agree strongly:

- **AUROC(delta -> causal) = 0.902** over all 10596 chunks.
- **AUROC = 0.916** restricted to salient chunks (the culprit-relevant decision).
- median delta: **3.09 nats** for causal vs **0.000** for non-causal.

Two unrelated definitions of "this chunk mattered" coincide, so the causal label is not an
artifact of the counterfactual rule. (Fig 9.)

## 2. Operating point of a continuous alternative
Substituting a tunable rule (causal := delta > T) is threshold-sensitive, as expected -- which
is exactly why we use the threshold-free counterfactual. At the natural band (T = 0.5-1 nat) the
continuous rule reproduces the discrete rate:

| T (nats) | no-culprit % |
|---:|---:|
| -1.0 | 21.0 |
| -0.5 | 21.3 |
| 0.0 | 24.3 |
| 0.5 | 35.1 |  <- natural band
| 1.0 | 38.9 |  <- natural band
| 2.0 | 44.5 |
| 3.0 | 48.9 |
| 5.0 | 57.0 |
| 8.0 | 64.8 |
| 12.0 | 73.7 |
| 18.0 | 83.3 |
| 25.0 | 90.0 |

At T=0.5-1, the rate is 35-39%, on the discrete value (37%).

## 3. Why is there no single culprit? (causal-structure taxonomy)
Built on the reliable causal axis. In a no-culprit case every causal chunk is non-salient, so the
question is how the causal mass is distributed:

| structure | meaning | count | % |
|---|---|---:|---:|
| coalition | coalition (>=2 chunks each flip the answer -- redundant blame) | 351 | 54.1 |
| one_silent_driver | one silent driver (a single chunk flips it but never states the value) | 106 | 16.3 |
| no_single_cause | no single cause (no chunk individually flips it -- robust/redundant or parametric) | 192 | 29.6 |
