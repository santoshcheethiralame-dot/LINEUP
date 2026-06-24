# Claim → evidence map

Every claim the paper makes, mapped to the exact figure/table/number that backs it. Write the
prose from this; do not state anything that is not on this page. Numbers are the full-matrix
ranges (12 cells = 3 model families × 2 datasets × {baseline, hard-traps}).

Artifacts: `results_tables.md` (Tables 1–3), `results_ci.md` (no-culprit CIs),
`figures/fig1..fig5.png`, `examples.md` (qualitative candidate cases — pick one per archetype).

## C1 — Resource: the LINEUP benchmark
- **Claim:** a controlled organic-error benchmark that assigns every retrieved passage a causal
  role via a non-circular leave-one-out oracle.
- **Evidence:** Method section; released HF dataset; the human-validation study (below).

## C1b — Oracle robustness (the causal axis has no tunable threshold)
- **Claim:** the causal label is a discrete counterfactual (removing a chunk flips the answer's
  value), so there is no threshold to game; an independent signal confirms it.
- **Evidence:** **Fig 9**; `run_oracle_robustness.py` (local, all 1,766 pooled wrong cases).
- **Numbers:** the never-used leave-one-out logprob drop agrees with the discrete causal label at
  **AUROC 0.90** (0.92 on salient chunks); median Δ **3.09 nats** causal vs **0.00** non-causal. A
  tunable rule (causal := Δ > T) is threshold-sensitive, but at the natural band T=0.5–1 it
  reproduces the discrete no-culprit rate (**35–39%** vs the shipped **37%**).
- **Taxonomy of no-culprit cases (n=649):** built on the reliable causal axis — **coalition (≥2
  chunks each flip the answer) 54%**, no single cause (none flips alone) 30%, one silent driver
  16%. So **~84% are redundant/distributed causation** — there is no single passage to name because
  there are several (or none singly). Directly explains the headline.

## Pillar A — Effect-based attribution localizes the culprit; lexical overlap does not
- **Claim:** ContextCite and SingleChunk find the culprit well; LLM-judge is middling; lexical lags.
- **Evidence:** Table 1; **Fig 3**.
- **Numbers:** ContextCite top-1 **0.70–0.92**, SingleChunk 0.62–0.95, LLM-judge 0.58–0.76,
  Lexical **0.22–0.57**. Effect-based beats lexical in **all 12** settings.
- **Errors are salience-shaped — for the causal methods (`run_error_direction.py`):** on the well-posed
  cases a method gets wrong, the misfire lands on the planted salient red-herring far above chance —
  single_chunk **2.8×**, ContextCite **2.5×**, llm_judge **2.4×** — but lexical only **1.1×** (its
  errors scatter to inert). The "fooled by salience" effect is real but localized to the minority of
  cases the effect-based methods already miss, and it is a property of *causal* scoring, not surface
  overlap. Refines (does not contradict) the aggregate result that methods are not salience-fooled.

## Pillar B — SPINE: a third of organic errors have no single culprit
- **Claim:** 27–53% of wrong cases are ill-posed — no single causal+salient passage.
- **Evidence:** **Fig 1**; `results_ci.md`.
- **Numbers:** no-culprit **27–53%** (Mistral most ill-posed, 41–53%). Present in **baseline**
  too (28–53%), so it is inherent to organic error, **not** a hard-traps artifact. CIs sit well
  above zero.
- **Frontier check (gpt-oss-120B via Cerebras, 136 cases):** 38% error rate; **44% no single
  culprit** (23/52), 95% CI [31–58%] — in the open-model band, so the ill-posedness holds at
  120B scale, not just 7B. (Generation-only oracle; no logprob methods on the closed API.)
- **Natural-retrieval slice (no planting; 2 datasets × 2 families; Fig 12):** organic errors from the
  dataset's own distractors — zero construction, no planted near-miss — show no-culprit rates that
  track the constructed baseline in **every cell**: HotpotQA/qwen **28%** (vs 29%), 2Wiki/qwen **26%**
  (vs 28%), HotpotQA/mistral **54%** (vs 49%). It even reproduces the cross-family ordering (Mistral
  most ill-posed). So ill-posedness is NOT an artifact of our construction; it holds with no planting
  at all, across datasets and models. (ContextCite top-1 0.77–0.96 on these.) Kills "it's synthetic."
- **Ill-posedness tracks reasoning structure (Fig 11; `run_structure.py`):** comparison-type questions
  are far more ill-posed than single-chain ones — HotpotQA **comparison 57%** [43,70] vs **bridge 34%**
  [30,37] (disjoint CIs); 2Wiki **bridge_comparison 55%** [47,62] vs **inference 28%** [22,34].
  Comparison questions need facts about two entities → two necessary causes → no single culprit. Holds
  in baseline alone (59% vs 36%), so not a planting artifact. Ill-posedness is partly a *fingerprint of
  multi-entity reasoning*, a semantic explanation of the headline.

## Pillar C — FLAGSHIP / constructive: single-chunk attribution is the wrong primitive
- **Claim:** under redundant evidence a single pick structurally cannot cover the culprit set;
  the effect-set recovers ~2× more. → use set-valued attribution.
- **Evidence:** **Fig 2**; Table 2 (recall@1 vs recall@k).
- **Numbers:** all **6** hard-traps cells: recall@1 **0.26–0.37** → recall@k **0.57–0.75** (~2×).
- **Dose-response (Fig 7; 0/1/2/3 redundant decoys, HotpotQA/qwen):** as redundancy grows, recall@1
  (single pick) decays monotonically **0.36 → 0.27 → 0.24 → 0.18** (tracking the top-1 ≤ 1/m bound)
  while recall@k (set) holds/grows **0.36 → 0.59 → 0.74 → 0.73**, and the no-culprit rate climbs
  **29 → 32 → 44 → 47%**. A clean controlled 4-point curve only a constructed benchmark can produce.
- **Credit dilution — the mechanism (Fig 10; `run_credit_dilution.py`, local):** the strongest
  responsible chunk's share of ContextCite's responsible score mass falls monotonically toward the
  1/|R| perfect-split floor as the coalition grows — **0.70 (|R|=2) → 0.53 (|R|=3) → 0.48 (|R|=4)**;
  pooled hard-traps cells (|R|=2, n=968) give **0.66**, corroborating the dose ladder. The method
  genuinely *splits* credit across the responsible set, so a single pick must shed the rest — the
  empirical mechanism behind the recall@1 decay. The linear surrogate already exhibits the
  credit-splitting Shapley predicts, so no Shapley run is needed.

## C2 — REMEDY: calibrated selective attribution (the method we build)
- **Claim:** a calibrated confidence signal lets attribution **abstain** when no single culprit is
  evident — raising the accuracy of the answers it does give — and falls back to an effect-set on
  the cases it abstains on. Diagnosis → working fix.
- **Evidence:** **Fig 5** (risk-coverage); `run_selective.py` (pooled, held-out test split).
- **Numbers:** 1,766 wrong cases (test 906). Naive top-1 is correct on only **0.35** of all errors.
  Best signal (top1−top2 margin) reaches **AUROC 0.77 [0.73, 0.80]** for "is this pick right".
  Abstaining to 50% coverage → accuracy **0.54 [0.49, 0.58]** (+0.19); to 30% → **0.64**. On the
  abstained half, effect-set recall **0.31 → 0.53**.
- **Abstention targets the ill-posed cases (`run_abstention_targeting.py`):** ranking by the ContextCite
  margin and abstaining the low-confidence tail, the abstained set is ~1.5× more no-culprit than the
  answered set at every coverage (50%: answered **29%** vs abstained **44.5%** no-culprit). The remedy
  stays silent precisely where the problem is ill-posed — it abstains for the *right reason*, not by
  dropping random hard cases. Direct validation of C2.

## C3 — RIGOR: conformal attribution sets (a coverage guarantee)
- **Claim:** a split-conformal set provably contains the culprit with probability ≥ 1−α, and the
  set size required grows under redundancy — ill-posedness made principled.
- **Evidence:** **Fig 6**; `run_conformal.py` (held-out calibration/test split).
- **Numbers:** 801 well-posed cases (test 390). At α=0.1, calibrated size **τ=2 → test coverage
  0.94** vs a single pick's **0.82**. Per condition: baseline **τ=2** (0.94), hard-traps **τ=3**
  (0.96).
- **Baselines — honest (`run_conformal_baselines.py`):** naive set rules reach *comparable* coverage
  on this data — fixed top-2 **0.94**, score-threshold **0.94** (avg size 1.5) — so conformal is NOT
  more efficient. Its contribution is the **distribution-free guarantee at a chosen α**, which the
  naive rules lack (their coverage rides on a hand-tuned k/threshold, with no recourse or
  cross-distribution promise). **Lead C3 with calibration, not performance** — this pre-empts the
  obvious "why not just top-k?" question. (Caveat: fixed top-2 still holds at 0.93 under redundancy,
  so the τ=2→3 growth is a *safer margin*, not a rescue from catastrophic failure; an adaptive APS
  variant bloats on diffuse effect scores and is dropped as a negative result.)

## Qualitative (Fig/Table in the appendix or main)
- **Evidence:** `examples.md` — a clean culprit (ContextCite succeeds), a coalition (no single
  culprit), and a salience trap (method blames a salient non-causal passage). Pick one of each.

## Pillar D — Attribution fails silently (no usable abstention)
- **Claim:** method confidence does not reliably flag wrong attributions, is never high enough to
  gate on, and degrades exactly when evidence is redundant.
- **Evidence:** Table 2 (reliability AUROC, single-culprit AUROC).
- **Numbers:** reliability AUROC **0.58–0.92** (baseline higher, hard-traps lower); single-culprit
  AUROC **0.48–0.75** (near chance in several cells). Frame as "never high enough to gate on."
- **But (links to C2):** a *calibrated, pooled* confidence signal reaches AUROC **0.77** for
  self-correctness — so the remedy is calibration + abstention, not raw method confidence.
- **Self-detecting ill-posedness is weak (`run_illposed_detector.py`):** the best score-only signal
  (effect-mass **entropy**) predicts the no-culprit label at only **AUROC 0.66 [0.63, 0.69]** —
  above chance but unreliable. Methods carry a faint self-signal that a case has no single culprit
  but cannot be trusted to flag it, which is exactly why the distribution-free conformal guarantee
  (C3) — not the method's own confidence — is the right remedy.

## Pillar E — Errors are model-specific
- **Claim:** which passage is to blame barely transfers across model families.
- **Evidence:** **Fig 4**; Table 3.
- **Numbers:** per-passage role κ **0.13–0.47** across all family pairs; 2Wiki baseline near
  chance (0.13–0.26). → attribution must be evaluated per model.

## Validation — the oracle's labels match human judgement (Table 3)
- **Claim:** independent humans pick the oracle's culprit at/above the rate they agree with each
  other, and the residual gap is the single-culprit bias itself.
- **Evidence:** 50 blind cases, 2 reliable annotators (Rushi excluded — 20%, GPT-assisted noise);
  `review_key.csv`; the adjudication study.
- **Numbers:** Santosh↔oracle **66%**, Nivas 48%; inter-rater **60%** — so the oracle sits *at/above*
  the human ceiling, vs ~14% chance on a 7-way choice. Agreement splits sharply: culprit cases
  **80% / 63%** vs no-culprit cases **33% / 13%** — humans force a pick when there is none, yet are
  **100% / 67% precise** when they do say "none."
- **Adjudication (team's discussed verdicts as ground truth):** taking the team-adjudicated answer as
  truth, the oracle is correct on **37/50**, **wrong on 4 (8%)**, ambiguous/unsure on 9 → accuracy
  **90%** on the 41 resolvable cases (74% floor if every unsure counts against it). The 4 errors:
  r019 (E→A), r022/r030/r042 (none→a salient passage) — **3 of 4 are NONE-vs-salient boundary calls**,
  the one genuinely fuzzy axis. Disagreements remain concentrated on no-culprit cases (humans force a
  pick), so the study validates the oracle and reproduces the single-culprit bias.
- **Caveats:** convenience sample (annotator-familiar topics); the team are not expert annotators
  (verdicts weighted by the discussion notes). Numbers from the team adjudication sheet + `review_key.csv`.

## Limitations (state these to preempt reviewers)
- 7B-class open models only (frontier GPT-4o/Claude is future work / optional slice).
- Two datasets, both Wikipedia-based multi-hop QA; ~70–220 wrong cases per cell.
- ContextCite depends on the ablation count (sensitivity check is future work).
- Human sample is weighted toward annotator-familiar topics (convenience sample, disclosed).
