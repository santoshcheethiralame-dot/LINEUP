# lineup — paper outline

Working title: **Which chunk lied? Organic chunk-role attribution in retrieval-augmented generation.**

Target venues: an ACL/EMNLP Findings short paper, a RAG or trustworthy-NLP workshop, or the NeurIPS Datasets and Benchmarks track. The dataset-and-benchmark framing fits the last best; the negative result fits a workshop.

The paper rests on three pillars, and every section serves one of them:

1. **The framing** — chunk roles as a 2×2 of counterfactual effect against apparent salience, with the *misleading* passage as the case existing tools conflate.
2. **The negative result** — salience-based attribution blames the misleading passage, and a controlled benchmark makes that measurable rather than anecdotal.
3. **The downstream cost** — failing to localise the culprit has a price, shown through selective answering.

## Abstract (draft)

Retrieval-augmented generation fails most dangerously when it is confidently wrong, and such failures usually trace to a single retrieved passage. Existing tools report *that* an answer is unfaithful or *how much* each passage contributed, but they are easiest to fool when the offending passage is not adversarial — merely stale, near-duplicate, or topically adjacent. We introduce a controlled benchmark whose chunk roles are known by construction: each case pairs gold supporting passages and realistic distractors with one constructed organic near-miss, and an exact leave-one-out oracle labels every passage as culprit, misleading, silent, or inert. On this benchmark, contributive and lexical attribution methods identify the true culprit far less often than their salience suggests, blaming the misleading passage at a high rate; and a selective-answering study shows that this localisation failure forfeits an abstention signal an oracle would capture. We release the benchmark, the construction harness, and the evaluation code.

## 1. Introduction

- The confident-wrong-answer problem in RAG; the cost of not knowing *which* passage caused it.
- The gap: faithfulness detection answers *whether*, contributive attribution answers *how much* (and conflates salience with cause), poison-traceback assumes an injected signature that organic errors lack.
- The organic setting: stale / near-duplicate / topically-adjacent passages, not adversarial injection.
- Contributions: (i) the 2×2 role framing; (ii) a benchmark with non-circular ground truth; (iii) the measured failure of salience-based attribution; (iv) the downstream selective-answering result; (v) the released dataset and harness.

## 2. Related work

- **Faithfulness / hallucination detection** (RAGAS, self-check) — flags unfaithfulness, not the responsible passage.
- **Contributive attribution** (ContextCite, SelfCite, TokenShapley) — estimates contribution but conflates a salient near-duplicate with the cause.
- **Poison traceback** (RAGOrigin, RAGCharacter) — recovers injected text by its signature; organic errors leave none.
- **Noise in RAG** (The Power of Noise, Lost in the Middle, Distracted by Irrelevant Context) — establishes that semantically related but wrong passages are the harmful ones; motivates the near-miss construction.
- The gap each leaves: none provides ground-truth roles for organic errors to evaluate against.

## 3. The benchmark

- **Roles.** The 2×2 of counterfactual effect (causal) against apparent salience, giving culprit, misleading, silent, inert.
- **Construction.** Multi-hop questions with annotated gold support (HotpotQA); a retriever for realistic distractors; the organic near-miss built by typed value substitution — clone the gold paragraph, replace the answer with a plausible same-type wrong value (year, number, entity); randomise order; record a reproducible recipe.
- **The oracle.** Exact leave-one-out: remove each passage, regenerate, read the correctness flip and the answer's value shift for the causal axis, and a verbosity-robust salience measure for the other. Cross them to place each passage in the 2×2.
- **Non-circularity.** Labels use only the known gold answer and the model's own behaviour, never a method under test — the property the benchmark's validity rests on.

## 4. Methods under test

- **ContextCite** — random ablations, logit-transformed response probability, Lasso surrogate; the contributive baseline.
- **Lexical similarity** — pure salience; the baseline that should blame the near-miss.
- **Single-chunk support** — a model-based salience baseline: the answer's log-probability under each passage alone.
- **LLM-judge culprit** — the model names the passage it thinks caused the error.
- Deliberately excluded: exact leave-one-out as a *method* (it is the oracle — circular).

## 5. Results — the negative result (Pillar 2)

- Headline **misleading-as-culprit rate** per method.
- **Culprit-over-misleading win-rate** — the within-case pairwise comparison on the crux; near 0.5 means a coin flip on the hard case.
- **Top-1 culprit accuracy** over cases that have a culprit, and the full 2×2 confusion of predicted versus true role.
- Read against the salience baseline: contribution methods track salience, not cause.

## 6. Results — does it matter (Pillar 3)

- Selective answering / abstention: can a signal tell correct answers from wrong ones?
- Signals compared: model self-confidence, each method's attribution decisiveness (top1−top2 margin), and the oracle's fixable-error flag as an upper bound.
- Metric: AUROC, plus risk–coverage. The gap between the oracle and the deployable signals is the cost of the localisation failure.

## 7. Limitations

- Labels reflect one model's behaviour at one precision; a different model may be fooled differently.
- Organic wrong answers are a minority, so the labelled set is smaller than the case set.
- Substitution covers years, numbers, and named entities; English only; one question family.

## 8. Conclusion

- A controlled benchmark turns "attribution conflates salience with cause" from intuition into measurement, and shows the failure carries a downstream cost.

## Claim → evidence map

| Claim | Evidence | Produced by |
| ----- | -------- | ----------- |
| Roles are well defined and assignable | the 2×2, role distribution per construction | `oracle.py`, `docs/stage4.md` |
| Labels are non-circular | construction + oracle use only gold answer and model behaviour | `docs/stage4.md` |
| Salience methods blame the near-miss | misleading-as-culprit rate, culprit-over-misleading win-rate | `scripts/run_scoring.py` |
| The failure is on the crux, not noise | 2×2 confusion, top-1 over cases-with-a-culprit | `scoring.py` |
| The failure has a downstream cost | AUROC / risk–coverage, oracle vs deployable signals | `scripts/run_abstention.py` |
| The rates are not sampling noise | 95% bootstrap confidence intervals | `scoring.bootstrap_intervals` |
| The result is not one model's quirk | cross-model culprit agreement, role kappa | `scripts/run_agreement.py` |
| The failure concentrates by type and position | per-slice scores | `scripts/run_breakdowns.py` |
| Coalition effects bound single-chunk attribution | synergy-pair rate | `scripts/run_interactions.py` |
| The oracle labels match human judgement | human/oracle agreement | `scripts/make_review_sheet.py` |
| The benchmark reproduces | deterministic construction, manifest | `scripts/build_release.py`, `tests/test_release.py` |

## Before submission

- Scale the run well beyond the demo sizes, so the per-method rates are tight.
- Add a second model (for example a Llama-3.1-8B-Instruct) so the labels are not tied to one model's quirks; report agreement with `scripts/run_agreement.py`.
- Add a cross-dataset check on a second multi-hop source (2WikiMultiHopQA) — the loader is wired; run with `--dataset 2wiki`.
- Finalise the figures: the 2×2 confusion and the risk–coverage curve.
