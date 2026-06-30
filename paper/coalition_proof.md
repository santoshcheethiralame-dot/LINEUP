# Leave-two-out proof: the no-single-cause cases are redundant coalitions

On wrong cases where single-chunk leave-one-out finds no causal passage, we run the pairwise
(leave-two-out) test. A synergy pair is one whose joint removal flips the answer though neither
passage alone does -- exactly the redundancy single leave-one-out cannot see.

| cell | no-single-cause cases | with a synergy pair |
|---|---:|---:|
| 2wiki/baseline/mistral | 3 | 3 (100%) |
| 2wiki/baseline/phi | 3 | 2 (67%) |
| 2wiki/baseline/qwen | 17 | 15 (88%) |
| 2wiki/hardtraps/mistral | 9 | 9 (100%) |
| 2wiki/hardtraps/phi | 18 | 15 (83%) |
| 2wiki/hardtraps/qwen | 30 | 19 (63%) |
| hotpotqa/baseline/mistral | 7 | 6 (86%) |
| hotpotqa/baseline/phi | 12 | 10 (83%) |
| hotpotqa/baseline/qwen | 16 | 13 (81%) |
| hotpotqa/hardtraps/mistral | 15 | 15 (100%) |
| hotpotqa/hardtraps/phi | 27 | 24 (89%) |
| hotpotqa/hardtraps/qwen | 35 | 24 (69%) |
| **pooled** | 192 | **155 (81%)** |

Across the pooled no-single-cause slice, **81%** have a pair whose joint removal flips the
answer though neither passage alone does. The redundancy is shown by direct ablation, not inferred
from the absence of a single cause: these cases are genuine coalitions the single-removal oracle is
blind to, which is why a single pick cannot name a culprit.

**Planted vs organic (honest).** Of the synergy pairs, **22% are between un-planted passages (organic redundancy)**, 48% one planted + one real, and 30% both planted (misleading+decoy).
Every scenario plants exactly one misleading chunk; hard-traps adds a decoy on top, so a
both-planted pair can only form under hard-traps. The honest organic claim is the **organic**
tally above (a synergy pair of two un-planted passages) -- not the baseline cells, which still
carry the planted misleading chunk and so contribute mixed pairs too.

Worked example:

- question gold answer: `Washington`
- the model answered: `June 13, 2008` (wrong; no single passage is causal)
- removing the pair `2b81511c0bde11eba7f7acde48001122::2` + `2b81511c0bde11eba7f7acde48001122::4` changes the answer to: `Tim Russert died in the United States.`
