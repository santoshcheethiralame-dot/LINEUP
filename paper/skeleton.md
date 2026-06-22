# Paper writing skeleton

Section-by-section structure. Each block says what to assert and which artifact backs it; pull the
numbers from `claim_evidence.md`, the figures from `figures/`, and the formal bits from
`theory.md`. Write the prose yourself — this is the map, not the text. Target ~8 pages.

Working title: **No Single Culprit: The Limits of Chunk Attribution for Organic RAG Errors**

## Abstract (~150 words)
One sentence each: the problem (which passage caused a wrong RAG answer) → the benchmark (LINEUP,
ground-truth chunk roles via a leave-one-out oracle) → the three results (attribution works on
single-cause errors; ~1/3 of organic errors have no single culprit and methods fail silently;
errors are model-specific) → the remedy (calibrated set-valued attribution + conformal coverage)
→ scope (3 model families, 2 datasets, human-validated).

## 1. Introduction (~1 page)
- Hook: RAG answers fail, and the fix usually traces to one retrieved passage — but *which*?
- Gap: faithfulness tells you *that*; attribution tells you *how much each contributed*; nobody
  measures *which passage caused it*, with ground truth, for organic (non-adversarial) errors.
- Contributions list (4): the benchmark; the ill-posedness + silent-failure finding; the
  set-valued/abstention remedy; the cross-model/cross-dataset/human-validated evidence.
- One punchy number up front: "across 12 settings, 27–53% of errors have no single culprit."

## 2. Related Work (~0.5 page)
Three buckets, each with the one-line differentiator (from claim_evidence C1 / lit-review):
faithfulness (RAGAS) — *that* not *which*; contributive attribution (ContextCite) — *how much*, and
assumes a single cause; poison traceback (RAGOrigin) — needs an adversarial signature. Position the
2×2 roles as the missing measurement.

## 3. The LINEUP Benchmark (~1.5 pages) — the resource contribution
- The 2×2: counterfactual effect × salience → culprit / misleading / silent / inert (Fig: the 2×2).
- Construction: multi-hop QA + gold + distractors + a typed-substitution near-miss (organic, not
  adversarial); optional redundant decoys for the m-fold-redundancy probe.
- The leave-one-out **oracle**: non-circular ground truth from the gold answer + model behavior,
  never a method under test. Define `answer_key` (value canonicalization) here.
- Released as a HuggingFace dataset (cite the artifact).

## 4. Experimental Setup (~0.5 page)
Models (Qwen2.5-7B, Phi-3.5-mini, Mistral-7B; + the gpt-oss-120B frontier check), datasets
(HotpotQA, 2Wiki), conditions (baseline, hard-traps), 4 attribution methods, metrics (top-1
culprit, no-culprit rate, recall@1/@k, reliability AUROC, role κ). 12 cells, ~1,700 wrong cases.

## 5. Results (~1.5 pages) — the diagnosis
- **5.1 Attribution works** — Table 1 / Fig 3 (claim_evidence Pillar A).
- **5.2 A third of errors have no single culprit** — Fig 1 + CIs; baseline ≈ hard-traps so it's
  inherent (Pillar B). *This is the spine — give it the most space.*
- **5.3 It fails silently** — reliability AUROC never high enough to gate on (Pillar D).
- **5.4 Errors are model-specific** — Fig 4, κ 0.13–0.47 (Pillar E).
- Qualitative: one clean culprit + one coalition + one trap from `examples.md`.

## 6. A Remedy: Set-Valued Attribution with Abstention (~1.5 pages) — the constructive turn
- **6.1 Theory** — `theory.md`: single-chunk attribution is *undefined* (not just inaccurate) under
  redundancy; top-1 recall ≤ 1/m; Shapley can't express it either.
- **6.2 Set recovery** — Fig 2; recall@1→@k ~2× (Pillar C). Dose-response Fig 7 traces the 1/m decay.
- **6.3 Selective attribution** — Fig 5; calibrated abstention lifts accuracy 0.35→0.54@50%/0.64@30% (C2).
- **6.4 Conformal sets** — Fig 6; provable 94% coverage, size grows 2→3 under redundancy (C3).

## 7. Validation (~0.5 page)
Blind human study: N annotators pick the culprit; human↔oracle agreement, majority agreement,
no-culprit ("none") agreement, inter-rater κ. Frontier check: gpt-oss-120B also shows ~X% no-culprit.

## 8. Limitations (~0.3 page)
From claim_evidence: open models + one frontier model; two Wikipedia-based datasets; ContextCite's
ablation count; convenience-weighted annotator sample; constructed near-miss (mitigated by the
natural-retrieval slice).

## 9. Conclusion (~0.3 page)
Single-chunk attribution is the wrong primitive for a third of real RAG errors; the field should
report sufficient-cause *sets* with calibrated abstention. Benchmark + code + dataset released.

## Figure/table inventory (all in figures/ and results_tables.md)
Fig1 no-culprit · Fig2 set-recovery · Fig3 methods · Fig4 κ · Fig5 selective · Fig6 conformal ·
Fig7 dose-response (pending runs) · Table1 accuracy · Table2 ill-posedness · Table3 agreement ·
results_ci.md · examples.md
