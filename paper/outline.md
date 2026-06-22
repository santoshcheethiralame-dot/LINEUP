# LINEUP — paper skeleton

> Write the prose yourself from these bullets. Every results bullet points at a number in
> `claim_evidence.md` / a figure in `figures/`. Do not state anything not backed there.

Working titles (pick one):
- **No Single Culprit: The Limits of Chunk Attribution for Organic RAG Errors**
- When Attribution Has No Answer: Ill-Posed Chunk Attribution in RAG
- Who's to Blame? Auditing Which Passage Caused a Wrong RAG Answer

Target venues: a **RAG / trustworthy-NLP workshop** (primary; the *Insights from Negative
Results* workshop fits the reframe), with an **ACL/EMNLP Findings** or **NeurIPS D&B** stretch.

## The arc (one sentence)
Effect-based attribution finds the culprit well *when there is one* — but a third of organic
errors have **no single culprit**, attribution can't tell when it's in that regime, and the fix is
to **abstain / go set-valued**. Holds across 3 model families × 2 datasets.

## Contributions (the real ones)
1. **LINEUP** — a controlled benchmark assigning every retrieved passage a ground-truth causal
   role (culprit / misleading / silent / inert) via a non-circular leave-one-out oracle, with
   organic near-misses and optional redundant decoys. (resource)
2. **Attribution works when well-posed** — effect-based methods localize the culprit at 0.70–0.92,
   far above lexical overlap. (calibrates expectations)
3. **~A third of organic errors have no single culprit** (27–53%) — single-chunk attribution is
   structurally ill-posed there; robust across families/datasets/conditions. (the spine)
4. **Silent failure + a remedy** — method confidence doesn't flag wrong attributions; a calibrated
   signal + abstention + set-valued fallback recovers trustworthiness. (the constructive turn)
5. **The culprit is model-specific** — which passage is to blame barely transfers across families
   (κ 0.13–0.47). (evaluate per model)

---

## Abstract (convey, in ~6 sentences — do not copy these as prose)
- RAG fails most dangerously when confidently wrong, and the failure usually traces to a passage.
- Prior tools say *whether* unfaithful or *how much* each passage contributed — not *which* is to blame, for *organic* (stale/near-duplicate) errors.
- We build LINEUP: known-by-construction roles + a non-circular leave-one-out oracle.
- Finding 1: effect-based attribution finds the culprit well *when one exists*.
- Finding 2 (headline): ~a third of errors have **no single culprit**, and methods can't tell — they answer confidently anyway.
- Remedy + scope: a calibrated abstention / set-valued method recovers trust; holds across 3 families × 2 datasets; dataset + code released.

## 1. Introduction (bullets to expand)
- The confident-wrong-answer problem; the cost of not knowing *which* passage caused it.
- Gap: faithfulness = *whether*; contributive attribution = *how much* (and is read as *which*); poison-traceback assumes an injected signature organic errors lack.
- The organic setting (stale / near-duplicate / topically-adjacent), not adversarial injection.
- The twist: the problem isn't that methods are bad — it's that the question is often **ill-posed**, and nobody measures that.
- Contributions list (above).

## 2. Related work (4 buckets + the gap)
- Faithfulness / hallucination detection (RAGAS, self-check) — *whether*, not *which*.
- Contributive attribution (ContextCite, SelfCite, TokenShapley) — *how much*; read as cause.
- Poison traceback (RAGOrigin, RAGCharacter) — injected-signature; organic errors leave none.
- Noise in RAG (Power of Noise, Lost in the Middle) — related-but-wrong passages are harmful.
- Gap: none gives ground-truth roles for organic errors, and none asks whether a single culprit *exists*.

## 3. The benchmark
- **Roles**: 2×2 of counterfactual effect × apparent salience → culprit / misleading / silent / inert.
- **Construction**: multi-hop QA (HotpotQA, 2Wiki) + gold support + retrieved distractors + an
  organic near-miss by typed value substitution; optional **redundant decoy** (hard-traps);
  randomized order; reproducible recipe.
- **Oracle**: exact leave-one-out — causal = answer value changes on removal; salient = passage
  holds the value; cross → role. **Non-circular** (uses only gold answer + model behavior).

## 4. Methods under test
- ContextCite (ablations → Lasso surrogate); SingleChunkSupport (answer logprob per chunk);
  LexicalSimilarity (pure salience); LLM-judge. Exact LOO excluded (it is the oracle).

## 5. Attribution finds the culprit — when one exists  → Table 1, Fig 3
- ContextCite/SingleChunk top-1 **0.70–0.92**, lexical **0.22–0.57**, judge middling. All 12 cells.
- Takeaway: methods are not broken; this makes the failures below credible.

## 6. A third of errors have no single culprit (the spine)  → Fig 1, results_ci.md
- no-culprit **27–53%** (Mistral most ill-posed); present in **baseline** too → not a decoy artifact.
- Define ill-posedness; redundancy is one controllable source (hard-traps), hallucination another.

## 7. Attribution can't tell when it's wrong  → Table 2
- Reliability AUROC **0.58–0.92**; single-culprit AUROC **0.48–0.75** (near chance in places).
- "Never high enough to gate on" — the silent-failure framing.

## 8. A remedy: selective + set-valued attribution  → Fig 5, Fig 2
- Naive top-1 correct on only **0.35** of all errors.
- Calibrated signal (margin) AUROC **0.77 [0.73,0.80]**; abstain→50% cov ⇒ **0.54 [0.49,0.58]** (+0.19); 30% ⇒ 0.64.
- On abstained cases, effect-set recall **0.31 → 0.53**; recall@1→@k **0.26–0.37 → 0.57–0.75** (Fig 2).
- Recommendation: set-valued attribution + redundancy-aware abstention.

## 9. The culprit is model-specific  → Fig 4, Table 3
- Per-passage role κ **0.13–0.47** across family pairs; 2Wiki baseline near chance. Evaluate per model.

## 10. Human validation  → (from score_validation.py)
- 3 reviewers, 50 blind cases, pick the culprit. Report human↔oracle, majority, no-culprit, inter-rater. [NUMBERS PENDING]

## 11. Limitations
- 7B-class open models (frontier = future work); 2 Wikipedia-based datasets; ~70–220 wrong/cell;
  ContextCite ablation-count sensitivity; human sample weighted to familiar topics (disclosed).

## 12. Conclusion
- A controlled benchmark turns "which passage?" into a measurable question — and shows the honest
  answer is often "no single one," which attribution must learn to say.

## Qualitative examples → examples.md (pick one clean culprit / coalition / trap)

---

## Before submission (checklist)
- [ ] Fold in human-validation numbers (§10).
- [ ] Release the dataset to HF (`build_release.py`) and cite it.
- [ ] Final figure captions; ensure every number in text matches `claim_evidence.md`.
- [ ] Mentor red-team review of the full draft by end of week 1.
- [ ] arXiv + workshop submission.
