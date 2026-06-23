# Writing kit — per-section scaffold

How to use: each section below lists its **job**, the **beats** (one bullet ≈ one paragraph's task),
the **exact numbers** to state, the **exhibit** to cite, and **don'ts**. Write your own sentences from
the beats — do not copy these bullets as prose. Every number is frozen in `claim_evidence.md`; if a
number isn't here or there, don't state it. Target ≈9 pages main + appendix.

Reviewer lens to keep in mind throughout: *validity* (is the ground truth real?), *scope* (does it
generalize?), *so-what* (why should a practitioner care?). Each section should pre-empt one of these.

---

## Abstract (≈200 words, 1 paragraph)
- **Job:** problem → gap → resource → headline → remedy → release, in that order.
- Beats: (1) when RAG errs, "which passage is to blame?" matters; (2) methods exist but are evaluated
  on synthetic injections / proxies because organic errors have no ground truth; (3) we build LINEUP —
  non-circular leave-one-out oracle, 2×2 roles; (4) effect-based localizes well (top-1 **0.70–0.92**),
  lexical doesn't (**0.22–0.57**); (5) **27–53% of organic errors have no single culprit**, robust to
  construction / scale / dataset; (6) so single-passage attribution is the *wrong primitive*
  (recall@1 ≤ 1/m); (7) remedy: selective abstention (**0.35→0.64**) + conformal sets (**90%** guarantee,
  **94%** empirical); (8) human study reproduces the bias; we release everything.
- **Don't:** hedge ("we attempt to"), or state any number not in claim_evidence.

## 1. Introduction (≈1 page, 5–6 paragraphs)
- **Beat 1 — the question.** RAG grounds answers in passages [Lewis, Gao]; on failure, practitioners
  want the culprit passage (purge a doc / fix retrieval / distrust a citation).
- **Beat 2 — the circular gap.** Methods exist (ContextCite [Cohen-Wang], attention/gradient) but to
  grade them you need the right passage, which for organic errors doesn't exist; the field uses
  synthetic injection or proxy metrics. State the real question: for a *real* error with no answer key,
  is "which passage" even well-posed?
- **Beat 3 — our apparatus.** LINEUP: controlled-but-organic cases; non-circular oracle (causal =
  counterfactual flip, salient = states the value; both from gold answer + model behaviour, never a
  method under test). Name the four roles. Forward-ref Fig 1.
- **Beat 4 — the three findings** (one sentence each): effect-based localizes well; ~⅓ have no single
  culprit; the ill-posedness has structure (redundancy, reasoning type, dose).
- **Beat 5 — the methodological turn.** No single passage → a method forced to name one answers the
  wrong question; recall@1 ≤ 1/m; fix = return a set. Name the two remedies.
- **Beat 6 — contributions list** (5 bullets, verbatim from claim_evidence C1–C3 + release).
- **Don't:** oversell as "fooled by salience" — that hypothesis *died*; lead with ill-posedness.

## 2. The LINEUP Benchmark (≈¾ page) — pre-empts *validity*
- **Cases.** Multi-hop QA (HotpotQA [Yang], 2Wiki [Ho]) for gold supporting facts; six passages = gold +
  distractors + optional near-miss (typed value substitution). Three regimes: natural / baseline / hard-traps.
- **The non-circular oracle.** Leave-one-out: causal = answer's *value* flips (discrete counterfactual,
  no threshold); salient = passage states the value. Stress: no method under test is consulted → labels
  cannot be circular. → 2×2 roles (Fig 1b). Define "no single culprit" = no causal∧salient passage.
- **Why trustworthy.** The unused logprob-drop signal agrees with the discrete label at **AUROC 0.90**
  (0.92 salient; median **3.09** vs **0.00** nats). A tunable threshold reproduces the rate at the natural
  band (**35–39%** vs **37%**) — so the counterfactual needs no threshold. Cite Fig 9 (appendix).
- **Pipeline & release.** File-based (scenarios→generations→roles→predictions→metrics), 4-bit open models,
  125 tests, public release.
- **Exhibit:** Fig 1. **Don't:** bury the non-circularity — it's the whole validity argument.

## 3. Effect-Based Attribution Localizes the Culprit (≈½ page) — pre-empts "is attribution even meaningful?"
- **Beat 1.** Four methods: ContextCite, SingleChunk, LexicalSimilarity, LLM-judge. On cases with a
  culprit: ContextCite **0.70–0.92**, SingleChunk **0.62–0.95**, LLM-judge **0.58–0.76**, lexical
  **0.22–0.57**; effect-based beats lexical in **all 12** cells. → effect, not surface overlap. **Table 1.**
- **Beat 2 — the salience nuance.** On well-posed cases a method gets *wrong*, it lands on the planted
  salient trap above chance: single_chunk **2.8×**, ContextCite **2.5×**, llm_judge **2.4×**, lexical
  **1.1×**. So "fooled by salience" is real but *localized* to the causal methods' misses. **Table 2.**
- **Don't:** claim methods are broadly fooled — they aren't; this is the minority-of-errors refinement.

## 4. A Third of Errors Have No Single Culprit (≈1 page, the SPINE) — pre-empts *scope*
- **Headline.** **27–53%** no-culprit across 12 cells; Mistral most ill-posed (**41–53%**); present in
  baseline (**28–53%**) → intrinsic, not a hard-traps artifact; CIs above zero. **Fig 2.**
- **Not synthetic.** Natural slice (no planting) tracks baseline cell-for-cell: **28 vs 29**, **26 vs 28**,
  **54 vs 49**; reproduces cross-family ordering. (Fig 12, appendix.) — the key rebuttal.
- **Not a small-model artifact.** gpt-oss-120B: **44%** [31–58]; flat 3.8B→120B. (Fig 8, appendix.)
- **Tracks reasoning structure.** comparison **57%** [43,70] vs bridge **34%** [30,37]; bridge_comparison
  **55%** [47,62] vs inference **28%** [22,34]; holds baseline-only (59 vs 36). **Fig 3.**
- **Why (taxonomy).** coalition **54%**, no-single-cause **30%**, one-silent-driver **16%** → **~84%**
  redundant/distributed. (Causal-axis taxonomy.)
- **Exhibits:** Fig 2, Fig 3. **Don't:** state mistral as 46–53% (it's 41–53%, the one sweep fix).

## 5. Single-Passage Attribution Is the Wrong Primitive (≈¾ page, FLAGSHIP) — the "so-what"
- **Ceiling.** Define responsible set R; one pick recovers ≤ 1/|R| of it (recall@1 ≤ 1/m). It's a
  *format* problem, not accuracy. Empirically all 6 hard-trap cells: recall@1 **0.26–0.37** → recall@k
  **0.57–0.75** (~2×). Forward-ref Appendix theory.
- **Dose–response.** decoys 0→3: recall@1 **0.36→0.27→0.24→0.18** (hugs 1/m), recall@k
  **0.36→0.59→0.74→0.73**, no-culprit **29→32→44→47%**. Fig 4a.
- **Mechanism (credit dilution).** top responsible chunk's score share **0.70→0.53→0.48** toward the
  1/|R| floor; pooled hard-traps |R|=2 = **0.66**. Same splitting Shapley predicts (→ appendix). Fig 4b.
- **Exhibit:** Fig 4 (a,b). **Don't:** present recall@k as "the method is good" — it's "the *set* is right."

## 6. A Calibrated, Set-Valued Remedy (≈¾ page) — the constructive payoff
- **Selective abstention.** margin signal predicts self-correctness at **AUROC 0.77** [0.73,0.80]; answer
  all → **0.35**; abstain to 50% → **0.54** [0.49,0.58]; 30% → **0.64**; abstained-half set recall
  **0.31→0.53**. And it abstains for the *right reason*: abstained set **44.5%** no-culprit vs answered
  **29%**. Fig 5a.
- **Conformal sets.** split-conformal [Vovk, Angelopoulos]; α=0.1 → τ=2 → **0.94** coverage vs single
  pick **0.82**; set size grows **τ=2 (baseline) → τ=3 (hard-traps)** → size self-reports redundancy. Fig 5b.
- **Exhibit:** Fig 5 (a,b). **Don't:** call conformal "heuristic" — it's a distribution-free guarantee.

## 7. Failure Is Silent and Model-Specific (≈½ page) — pre-empts "just gate on confidence"
- **Silent.** reliability AUROC **0.58–0.92** (worse where redundant); self-detecting no-culprit from raw
  scores is weak — entropy **0.66** [0.63,0.69]. → need the guarantee, not raw confidence. **Table 3** (recovery).
- **Model-specific.** role κ **0.13–0.47**; 2Wiki baseline **0.13–0.26**. → evaluate per model. **Fig 6.**
- **Exhibit:** Fig 6, Table 3. **Don't:** overclaim transfer — it barely transfers.

## 8. Human Validation (≈½ page) — closes the validity loop
- **Setup.** 50 blind cases, 2 reliable annotators (3rd excluded, 20%, disclosed); chance ≈14%.
- **At/above ceiling.** annotator↔oracle **66% / 48%**; inter-annotator **60%** → oracle as close as
  humans are to each other.
- **Gap = the thesis.** culprit cases **80% / 63%** vs no-culprit **33% / 13%** (humans force a pick);
  precision-when-"none" **100% / 67%**. → humans share the single-culprit bias.
- **Adjudication.** team verdicts as truth: **37/50** correct, **4 (8%)** wrong, 9 ambiguous → **90%** on
  41 resolvable; 3 of 4 errors are no-culprit-vs-salient boundary calls. **Table 4.**
- **PENDING:** the neutral spot-check of 9 contested cases is out; if it confirms, keep "90% / 8% errors";
  if it shifts a case, update the 37/4/9 split and this paragraph only. Numbers above are otherwise final.
- **Don't:** report a bare agreement % — the *split* and the adjudication are the story.

## 9. Related Work (≈½ page)
- Six tagged mini-paragraphs (one cite-cluster each): context attribution [Cohen-Wang, Sundararajan];
  feature attribution/Shapley [Lundberg, Shapley]; RAG robustness/noise [Lewis, Gao, Cuconasu]; position
  [Liu]; conformal + selective [Angelopoulos, Vovk, Geifman]; RAG eval / attributed QA [Es, Bohnet].
- For each: one line on what they do, one line on how we differ (per-error causal role vs aggregate/proxy).
- **Don't:** list-dump; every cite needs a "we differ by…".

## 10. Limitations (≈⅓ page) — state, don't hide
- 7B-class + one 120B check; two Wikipedia multi-hop datasets, ~70–220 wrong/cell; ContextCite ablation-
  count sensitivity is future work; convenience human sample, non-expert annotators (verdicts weighted by
  notes); the no-culprit-vs-salient boundary is the one fuzzy axis (the 4 human-flagged errors live there).

## 11. Conclusion (≈⅓ page)
- "which single passage" is the wrong question for ~⅓ of RAG errors; LINEUP measures it rigorously;
  blame is genuinely shared → wrong primitive → calibrated set-valued remedy with a guarantee that
  self-reports uncertainty; release so methods are graded against cause, not proxy.

## Appendix (unbounded)
- A: theory (Prop 1 recall ≤ 1/m, Prop 2 Shapley dilution, Prop 3 conformal coverage) — see `theory.md`.
- B: oracle robustness (Fig 9) · C: natural slice (Fig 12) · D: scale (Fig 8) · E: cross-model κ table
- F: full 12-cell tables (`results_tables.md`) · G: qualitative examples (`examples.md`).
