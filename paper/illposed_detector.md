# Can a method self-detect 'no single culprit'?

Pooled wrong cases with contextcite scores: **1766**. No-culprit base rate: **36.7%**.
Each signal is computed from the method's chunk scores alone (no oracle at inference).

| signal | AUROC vs no-culprit |
|---|---:|
| n_effective | 0.583 |
| entropy | 0.659 |
| neg_margin | 0.603 |
| neg_top1_share | 0.638 |

Best: **entropy**, AUROC **0.659** [0.633, 0.686] (1000-sample bootstrap).

Reading: a method can flag ill-posed (no-single-culprit) errors from the *spread* of its own
effect scores, well above chance, without any ground truth. This is the per-case signal that
justifies switching to a set, and explains why the calibrated conformal set grows under
redundancy (C3): the same diffuse-score structure that flags ill-posedness forces a larger set.