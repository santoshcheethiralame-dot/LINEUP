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

## Pillar A — Effect-based attribution localizes the culprit; lexical overlap does not
- **Claim:** ContextCite and SingleChunk find the culprit well; LLM-judge is middling; lexical lags.
- **Evidence:** Table 1; **Fig 3**.
- **Numbers:** ContextCite top-1 **0.70–0.92**, SingleChunk 0.62–0.95, LLM-judge 0.58–0.76,
  Lexical **0.22–0.57**. Effect-based beats lexical in **all 12** settings.

## Pillar B — SPINE: a third of organic errors have no single culprit
- **Claim:** 27–53% of wrong cases are ill-posed — no single causal+salient passage.
- **Evidence:** **Fig 1**; `results_ci.md`.
- **Numbers:** no-culprit **27–53%** (Mistral most ill-posed, 46–53%). Present in **baseline**
  too (28–53%), so it is inherent to organic error, **not** a hard-traps artifact. CIs sit well
  above zero.
- **Frontier check (gpt-oss-120B via Cerebras, 136 cases):** 38% error rate; **44% no single
  culprit** (23/52), 95% CI [31–58%] — in the open-model band, so the ill-posedness holds at
  120B scale, not just 7B. (Generation-only oracle; no logprob methods on the closed API.)
- **Natural-retrieval slice (no planting, HotpotQA/qwen):** organic errors from the dataset's own
  distractors — zero construction — show **28% no-culprit**, matching the 29% constructed baseline.
  So ill-posedness is NOT an artifact of our planted near-miss; it holds with no planting at all.
  (ContextCite top-1 0.96 on these.) Kills the "it's synthetic" objection.

## Pillar C — FLAGSHIP / constructive: single-chunk attribution is the wrong primitive
- **Claim:** under redundant evidence a single pick structurally cannot cover the culprit set;
  the effect-set recovers ~2× more. → use set-valued attribution.
- **Evidence:** **Fig 2**; Table 2 (recall@1 vs recall@k).
- **Numbers:** all **6** hard-traps cells: recall@1 **0.26–0.37** → recall@k **0.57–0.75** (~2×).
- **Dose-response (Fig 7; 0/1/2/3 redundant decoys, HotpotQA/qwen):** as redundancy grows, recall@1
  (single pick) decays monotonically **0.36 → 0.27 → 0.24 → 0.18** (tracking the top-1 ≤ 1/m bound)
  while recall@k (set) holds/grows **0.36 → 0.59 → 0.74 → 0.73**, and the no-culprit rate climbs
  **29 → 32 → 44 → 47%**. A clean controlled 4-point curve only a constructed benchmark can produce.

## C2 — REMEDY: calibrated selective attribution (the method we build)
- **Claim:** a calibrated confidence signal lets attribution **abstain** when no single culprit is
  evident — raising the accuracy of the answers it does give — and falls back to an effect-set on
  the cases it abstains on. Diagnosis → working fix.
- **Evidence:** **Fig 5** (risk-coverage); `run_selective.py` (pooled, held-out test split).
- **Numbers:** 1,766 wrong cases (test 906). Naive top-1 is correct on only **0.35** of all errors.
  Best signal (top1−top2 margin) reaches **AUROC 0.77 [0.73, 0.80]** for "is this pick right".
  Abstaining to 50% coverage → accuracy **0.54 [0.49, 0.58]** (+0.19); to 30% → **0.64**. On the
  abstained half, effect-set recall **0.31 → 0.53**.

## C3 — RIGOR: conformal attribution sets (a coverage guarantee)
- **Claim:** a split-conformal set provably contains the culprit with probability ≥ 1−α, and the
  set size required grows under redundancy — ill-posedness made principled.
- **Evidence:** **Fig 6**; `run_conformal.py` (held-out calibration/test split).
- **Numbers:** 801 well-posed cases (test 390). At α=0.1, calibrated size **τ=2 → test coverage
  0.94** vs a single pick's **0.82**. Per condition: baseline **τ=2** (0.94), hard-traps **τ=3**
  (0.96) — redundancy demands a larger set for the same guarantee.

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

## Pillar E — Errors are model-specific
- **Claim:** which passage is to blame barely transfers across model families.
- **Evidence:** **Fig 4**; Table 3.
- **Numbers:** per-passage role κ **0.13–0.47** across all family pairs; 2Wiki baseline near
  chance (0.13–0.26). → attribution must be evaluated per model.

## Validation — the oracle's labels match human judgement
- **Claim:** independent humans pick the same culprit the oracle does.
- **Evidence:** human-validation table (50 blind cases, 3 reviewers).
- **Numbers:** PENDING — human↔oracle agreement, majority-vote agreement, no-culprit ("none")
  agreement, inter-rater agreement (from `score_validation.py`).

## Limitations (state these to preempt reviewers)
- 7B-class open models only (frontier GPT-4o/Claude is future work / optional slice).
- Two datasets, both Wikipedia-based multi-hop QA; ~70–220 wrong cases per cell.
- ContextCite depends on the ablation count (sensitivity check is future work).
- Human sample is weighted toward annotator-familiar topics (convenience sample, disclosed).
