# Credit dilution (why single-chunk attribution is the wrong primitive)

For each wrong case with a responsible set R = {near-miss, decoys}, we measure the share of
ContextCite's responsible-score mass captured by its single strongest responsible chunk. Perfect
concentration = 1.0 (one chunk holds all the blame); perfect splitting = 1/|R|.

| |R| | n decoys | top responsible chunk's share | 1/|R| floor | n |
|---:|---:|---:|---:|---:|
| 2 | 1 | 0.70 | 0.50 | 138 |
| 3 | 2 | 0.53 | 0.33 | 183 |
| 4 | 3 | 0.48 | 0.25 | 187 |

Pooled hard-traps cells (|R|=2, all 6 cells, n=968): top share **0.66** (floor 0.50) -- corroborates the |R|=2 dose point.

The strongest responsible chunk's share falls monotonically toward the 1/|R| floor as the
coalition grows: the method genuinely spreads credit across the responsible set rather than
concentrating it. A single pick must therefore shed the rest of the blame -- the empirical
mechanism behind the recall@1 decay (Fig 7) and the 'wrong primitive' claim. The linear
surrogate already shows the credit-splitting that Shapley would predict, so no Shapley run is
needed.