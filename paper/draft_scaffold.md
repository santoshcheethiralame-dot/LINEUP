# Paper draft scaffold

A fill-in-the-blanks skeleton. Each bullet in [brackets] tells you what that paragraph must
do — write it in your own words and delete the bracket. Nothing here is paper prose; it is a
map so you never face a blank page.

Markers to leave in until resolved:
- `[CITE: x]` — drop in a citation (collect the BibTeX as you go).
- `[FIG: x]` — a figure you will insert (most come from `runs/dashboard.png`).
- `[NUM: x]` — a number from your study run.
- `[?]` — a decision still open.

How to use it: write the Method and Benchmark sections first (you know them cold), then
Results once the run lands, then Related Work, then Introduction, then the Abstract last.
Write closed-book — read a source, close it, write the idea from memory. Keep `paper/outline.md`
open beside this as your content map; pull facts from `docs/stage*.md`, but write fresh
sentences, not copied ones.

---

## Title
- [Decide a final title. It should name the task (organic chunk-role attribution in RAG) and
  hint at the finding. Avoid a vague title.]

## Abstract  —  ~150–200 words, WRITE THIS LAST
One sentence each, in this order:
- [Problem: when does RAG fail most dangerously, and where does the failure trace to?]
- [Gap: what do existing tools tell you, and where are they weakest?]
- [What you built: the benchmark with roles known by construction.]
- [Method in a phrase: the 2×2, the constructed near-miss, the leave-one-out oracle.]
- [Headline result: what the salience-based methods do — give the direction, no numbers needed here.]
- [Downstream: the selective-answering cost.]
- [Release: dataset + harness.]

## 1. Introduction  —  ~0.75–1 page
- [Hook, 2–3 sentences: the confident-wrong-answer failure mode and that it usually traces to one passage.]
- [The gap, 1 paragraph: faithfulness detection answers *whether*; contributive attribution answers *how much* and conflates salience with cause; poison traceback assumes an injected signature organic errors lack. [CITE: RAGAS] [CITE: ContextCite] [CITE: RAGOrigin]]
- [The organic setting, 1–2 sentences: stale / near-duplicate / topically-adjacent, not adversarial injection. [CITE: Power of Noise]]
- [Your move, 2–3 sentences: a controlled benchmark whose chunk roles are known by construction.]
- [The 2×2 in one sentence, and name the misleading chunk as the hard case.]
- **Contributions** (bulleted list of 4–5):
  - [the 2×2 role framing]
  - [the benchmark + its non-circular ground truth]
  - [the measured failure of salience-based attribution]
  - [the downstream selective-answering result]
  - [the released dataset and harness]
- [Optional one-sentence roadmap of the paper.]
- *Strong if:* a reviewer can state your contribution after reading only this page.

## 2. Related Work  —  ~0.5–0.75 page
Three to four mini-paragraphs; end each with one clause on what it leaves open for you.
- [Faithfulness / hallucination detection — what it flags, why "which chunk" is out of scope. [CITE: RAGAS]]
- [Contributive attribution — ContextCite, SelfCite, TokenShapley: estimate contribution, conflate a salient near-duplicate with the cause. [CITE: ContextCite] [CITE: TokenShapley]]
- [Poison traceback — RAGOrigin / RAGCharacter: recover injected text by its signature; organic errors leave none. [CITE: RAGOrigin]]
- [Noise in RAG — establishes that semantically-related-but-wrong passages are the harmful ones; motivates your near-miss. [CITE: Power of Noise] [CITE: Lost in the Middle]]
- *Strong if:* every paragraph ends in "…but none provides ground-truth roles for organic errors."

## 3. The Benchmark  —  your core, ~1–1.5 pages
- **Roles.** [Define the two axes (counterfactual effect, apparent salience) and the four quadrants. A small 2×2 table here. [FIG: the 2×2 table]]
- **Construction.** [Source of gold + support [CITE: HotpotQA]; how distractors are sourced; the organic near-miss via typed value substitution — clone the gold paragraph, swap the answer for a plausible same-type wrong value (year/number/entity); randomized order; reproducible recipe.]
- **The oracle.** [Exact leave-one-out: remove each passage, regenerate, read the value shift (causal axis) and the salience measure; cross them into the 2×2. [CITE: ContextCite] for the ablation idea, [CITE: Influence functions] for the lineage.]
- **Non-circularity.** [One careful paragraph: the labels use only the known gold answer and the model's own behaviour, never a method under test. This is the property the benchmark rests on — say why it holds.]
- *Strong if:* a skeptical reader is convinced the labels are trustworthy by the end of the non-circularity paragraph.

## 4. Methods Under Test  —  ~0.3 page
- [One sentence each: ContextCite [CITE], lexical similarity, single-chunk support, LLM-judge.]
- [State why exact leave-one-out is the oracle, not a method (using it would be circular).]

## 5. Results — the negative result  —  ~1 page
- [Setup sentence: model(s), number of cases, that scoring is on the wrong-answer cases.]
- [Headline: the misleading-as-culprit rate per method, and the culprit-over-misleading win-rate near [NUM: ~0.5]. [FIG: dashboard top-left — rate with CIs]]
- [The 2×2 confusion: where each method's predicted culprit truly lands. [FIG: dashboard top-right — heatmap]]
- [Read it against the salience baseline: contribution tracks salience, not cause.]
- [Cross-model: does it hold under a second model? [NUM: agreement] [FIG: dashboard bottom-right]]
- [Breakdown: where the failure concentrates (substitution type / culprit position).]
- *Strong if:* the CIs are tight enough that the headline is a finding, not a hint — that needs the scaled run.

## 6. Does it matter — the downstream cost  —  ~0.5 page
- [Selective answering: can a confidence signal separate correct from wrong well enough to abstain?]
- [Compare self-confidence, each method's decisiveness, and the oracle's fixability ceiling. [FIG: dashboard bottom-left — risk–coverage]]
- [The gap between the oracle and the deployable signals is the cost of the localisation failure. [CITE: Self-RAG] [CITE: CRAG]]

## 7. Limitations  —  ~0.25 page
- [One model's behaviour at one precision defines the labels; a different model may be fooled differently.]
- [Organic wrong answers are a minority, so the labelled set is smaller than the case set.]
- [Substitution covers years/numbers/entities; English only; one question family.]

## 8. Conclusion  —  2–3 sentences
- [What you showed, and why "attribution conflates salience with cause" being measurable matters.]

---

## Figures to produce (most already exist)
- [ ] 2×2 roles table (hand-drawn / LaTeX) — Section 3.
- [ ] misleading-as-culprit with CIs — `runs/dashboard.png` top-left — Section 5.
- [ ] predicted-role heatmap — `runs/dashboard.png` top-right — Section 5.
- [ ] risk–coverage curves — `runs/dashboard.png` bottom-left — Section 6.
- [ ] cross-model agreement — `runs/dashboard.png` bottom-right — Section 5.
- [ ] (export each panel as its own PDF for a sharp figure, not a screenshot of the grid.)

## References to gather (collect BibTeX into refs.bib)
- ContextCite (2409.00729)
- HotpotQA (Yang et al., 2018; 1809.09600)
- 2WikiMultiHopQA (Ho et al., 2020)
- RAGAS (Es et al., 2023)
- The Power of Noise (Cuconasu et al., 2024; 2401.14887)
- Lost in the Middle (2307.03172)
- RAGOrigin (2509.13772) and RAGCharacter
- Self-RAG (Asai et al., 2023); CRAG (Yan et al., 2024)
- Influence functions (Koh & Liang, 2017)
- TokenShapley (2507.05261)

## Pre-submission checklist
- [ ] Read every section aloud; rewrite anything that sounds like a textbook or a model.
- [ ] Each claim that isn't yours has a citation.
- [ ] Run the similarity checker; confirm matches are only references and standard phrases.
- [ ] Numbers in the text match the figures and tables.
- [ ] The non-circularity paragraph (Section 3) is airtight — it is the question a reviewer will press.
- [ ] Abstract written last, matches the paper.
