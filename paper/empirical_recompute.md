# Empirical recompute (reviewer items #2, #3, #4, #13, #14) — all from local per-case data

All numbers below are recomputed from `paper/results_data/**` (per-case `roles.jsonl`,
`predictions.jsonl`, `scenarios.jsonl`) and the human-round CSVs. Scripts:
`compute_salience_percell.py`, `compute_conformal_recall.py`, `compute_kappa_bootstrap.py`.

## #2 — Metric reconciliation (VERIFIED against the reported tables)
Three distinct metrics, two sharing the name "recall@1":

| metric (code name) | pooled | reproduces |
|---|---:|---|
| `top1_culprit_accuracy` — P(top-1 = culprit \| case has a culprit), over **well-posed** cases | **0.83** (0.70–0.92/cell) | Table 1 (hotpotqa/base/qwen 0.92, n=90 ✓) |
| planted-set `recall_at_1` — \|top-1 ∩ R\|/\|R\|, R = `{misleading, decoy}` | **0.30** | A4 leaderboard ✓ |
| planted-set `recall_at_k` — top-\|R\| coverage | **0.50** | A4 ✓ (hard-traps 0.57–0.75 = §5) |

- **The collaborator's reconciliation guessed R wrong** — code uses the *planted* set, not "every causal passage."
- **Table 1 n bug:** the n-column (90, 89, …, 172) sums to **1,766 = all-wrong**, but `top1_culprit_accuracy`
  is over **well-posed** cases (pooled **1,117**; per cell 64, 55, 37, …). Fix n to the well-posed count.
- The two metrics are non-comparable (different numerator, denominator, and target). Relabel Table 1 →
  "Culprit accuracy"; A4/A8 → "Planted-set recall@1/@k (\|R\| = misleading+decoy)".

## #3 — De-oracled recall (fixed conformal size, never the per-case oracle \|R\|)
Recall over the planted set R at a fixed top-τ (vs oracle recall@k which needs per-case \|R\|):

| condition | mean \|R\| | recall@1 | recall@2 | recall@3 | recall@k (oracle) |
|---|---:|---:|---:|---:|---:|
| baseline | 1.00 | 0.29 | 0.48 | 0.66 | 0.29 |
| hard-traps | 2.00 | 0.30 | **0.65** | 0.77 | **0.65** |

A fixed top-2 recovers the planted set as well as the oracle-sized set (0.65 = 0.65) — recall@k is **not**
an artifact of knowing \|R\|.

## #4 + #13(Wilson) — No-culprit per cell by salience matcher (Wilson 95% CIs)
Pooled (n=1,766): strict **43%**, shipped **37% [35–39]**, loose **19% [17–21]**.
Per cell, the LOOSE (over-generous) matcher floor ranges **9%–36%**; **every cell's lower CI is > 0**
(min 6%), so the finding survives the harshest reading — though it shrinks to 9–13% in several 2Wiki
cells under the loose rule. Full per-cell table: `salience_percell.md`.

## #13(Holm) — No-culprit by question type (Wilson CIs + Holm–Bonferroni, m=4)
| contrast | rates [95% CI] | p | CIs | Holm |
|---|---|---:|---|---|
| 2Wiki bridge_comp vs inference (pooled) | 55% [47–62] vs 28% [22–34] | <0.0001 | disjoint | ✓ reject |
| 2Wiki bridge_comp vs inference (baseline) | 56% [46–66] vs 31% [22–41] | 0.0005 | disjoint | ✓ reject |
| HotpotQA comp vs bridge (pooled) | 57% [43–70] vs 34% [30–37] | 0.0009 | disjoint | ✓ reject |
| HotpotQA comp vs bridge (baseline) | 59% [39–77] vs 36% [30–42] | 0.0338 | **overlap** (n=22) | ✓ reject |

- The paper's headline "57% vs 34% / 55% vs 28%, disjoint CIs" = the **pooled** numbers (correct).
- **Caveat:** "holds in the baseline slice alone" is true by a 2-proportion test (survives Holm), but the
  HotpotQA *baseline* comparison CIs **overlap** (n=22 too small for disjoint). Soften "disjoint CIs" →
  "significant by a 2-proportion test (Holm-corrected)" for the baseline claim, or note the small n.

## #14 — κ bootstrap CIs (case-resampling, B=2000)
**Confirmatory round (n=30) — clean, all CIs clear of 0:**
| | raw | κ | 95% CI |
|---|---:|---:|---|
| Reviewer 1 vs oracle | 70% | 0.63 | [0.42, 0.83] |
| Reviewer 2 vs oracle | 80% | 0.75 | [0.55, 0.92] |
| inter-rater | 67% | 0.59 | [0.39, 0.79] |

**Data gap (honest):** the pilot's full-50 raw picks were **not preserved** — only blank templates, the
25 contested cases (`adjudication.xlsx`: Santosh κ 0.25 [0.08, 0.45], Nivas 0.00 [−0.13, 0.14], the
expected-low *disagreement* subset), and the aggregate (66%/48%/60%) survive. Rushi's raw picks are
gone, so the 3-rater Fleiss / exclusion-sensitivity isn't computable. The confirmatory round is the
primary result, so the headline κ is well-bounded; the pilot stands on its disclosed aggregate.

## #15 — Stronger (oracle-supervised) learned baseline — DONE locally (`learned_baseline.md`)
An oracle-trained gradient-boosted stack of all four method scores + structure (position, retrieval),
grouped 5-fold CV. Well-posed culprit accuracy **0.848** vs ContextCite 0.825 (**+2.2pt**, near-saturated);
coalition (|R|≥2, n=1033) recall@1 **0.299** — no better than single_chunk 0.316, under the 1/m = 0.50
bound. Even training on the answer key cannot beat the heuristics on coalitions: the limit is the
single-passage primitive, not method strength. No GPU needed (stronger as a local upper bound).

## #16 — 4-bit vs fp16 sanity — DONE (Qwen, 2×T4): NOT a quantization artifact (`fp16_sanity.md`)
**Pooled (n_both=200): no-single-culprit 4-bit 32% vs fp16 31%, per-case agreement 88%.** Hardtraps
cells agree tightest (46/46, 28/23). Re-ran the full generate→leave-one-out pipeline in fp16 on a
matched Qwen subsample (60 wrong@4bit/cell × 4 cells). The finding survives full precision.
Reproduce for Phi-3.5 / Mistral-7B on Kaggle 2×T4:
```
python scripts/run_fp16_sanity.py \
  --data-dir paper/results_data/hotpotqa/baseline/qwen paper/results_data/hotpotqa/hardtraps/qwen \
             paper/results_data/2wiki/baseline/qwen   paper/results_data/2wiki/hardtraps/qwen \
  --model Qwen/Qwen2.5-7B-Instruct --limit 60
```
Swap `--model` (and the cell paths) to the exact Phi-3.5 / Mistral-7B ids for the other two models.
