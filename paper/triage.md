# Figure & result triage — locked structure for writing week

Target: NeurIPS Datasets & Benchmarks form factor (~9 pp main + unlimited appendix). The main body
carries the arc **ill-posed → mechanism → remedy → doesn't transfer**; everything else is appendix
armour (unlimited, so nothing is wasted). "Old fig" = current filename in `paper/figures/`.

## Main-body arc (6 figures, 3 tables)

| § | Section | Claim | Main exhibit | Old fig/source |
|---|---|---|---|---|
| 1 | Introduction | the problem + contribution | — (prose) | — |
| 2 | The LINEUP benchmark | C1 resource; non-circular oracle | **Fig 1** framework (pipeline + 2×2 roles) | BUILD (reuse deck pipeline + 2×2) |
| 3 | Attribution localizes the culprit | Pillar A | **Table 1** localization by method (+ salience-lift col) | Table 1 + `run_error_direction` |
| 4 | A third of errors have no single culprit | Pillar B (headline) | **Fig 2** no-culprit across 12 cells | fig1 |
| 4 | …and it tracks reasoning structure | Pillar B (deepen) | **Fig 3** ill-posedness by question type | fig11 |
| 5 | Single-chunk attribution is the wrong primitive | Pillar C (flagship) | **Fig 4** 2-panel: dose-response + credit dilution | fig7 + fig10 |
| 6 | A calibrated set-valued remedy | C2 + C3 | **Fig 5** 2-panel: selective risk-coverage + conformal | fig5 + fig6 |
| 7 | Failure is silent and model-specific | Pillar D + E | **Fig 6** model-transfer κ; **Table 2** recall + reliability AUROC | fig4; Table 2 |
| 8 | Validation & limitations | oracle ↔ human | **Table 3** human-validation agreement | pending adjudication |

Figure count: **6**. Table count: **3**. Clean for 9 pages.

## Appendix (the armour — full data, reviewers can check)

| Appendix item | Source | Why it's here not main |
|---|---|---|
| A. Oracle robustness / concordance AUROC 0.90 | fig9, `run_oracle_robustness` | methodology defence; one main-text sentence points here |
| B. No-culprit taxonomy (coalition 54%) | `run_oracle_robustness` | *explains* headline; a sentence in §4 cites it |
| C. Frontier gpt-oss-120B (44%) + no-culprit vs scale | fig8, `run_frontier` | scale robustness; one sentence in §4 |
| D. Natural-retrieval slice (28%, no planting) | `run_natural` | ecological-validity robustness; one sentence in §4 |
| E. Ill-posedness self-detection (entropy 0.66) | `run_illposed_detector` | weak result; one line motivating conformal in §6 |
| F. Abstention targets ill-posed cases (44.5% vs 29%) | `run_abstention_targeting` | validates C2; one line in §6 + appendix table |
| G. Standalone recall@1 vs recall@k bars | fig2 | subsumed by Fig 4 dose-response panel |
| H. Effect-based vs lexical bars | fig3 | subsumed by Table 1 |
| I. Theory proofs (top-1 ≤ 1/m; Shapley non-expressibility; minimal sufficient cause) | `theory.md` | full derivations; results stated inline in §5 |
| J. Full per-cell tables (all 12 cells, every method/metric) | `results_tables.md`, `results_ci.md` | main body shows ranges; appendix shows the grid |
| K. Qualitative examples (clean culprit / coalition / salience trap) | `examples.md` | one example box in main, rest in appendix |

## Cut entirely
Nothing. Appendix is unlimited, and every result moves a claim. The only near-cut is **E (self-
detection, 0.66)** — it survives as a single motivating sentence, not a figure.

## Merge decisions (why 11 figures → 6 main)
- **Fig 4** merges dose-response (fig7) + credit dilution (fig10) — same claim (wrong primitive), one
  shows the symptom (recall@1 decays), the other the mechanism (credit splits toward 1/|R|).
- **Fig 5** merges selective (fig5) + conformal (fig6) — same claim (the remedy), abstain-or-set.
- fig2 → appendix (dose-response panel of Fig 4 already shows recall@1 vs recall@k).
- fig3 → appendix (Table 1 already has the per-method localization numbers).
- fig8 → appendix (the 44% frontier number lands in one sentence; the flat-with-scale plot is armour).

## Build list
1. **Fig 1 framework schematic** — DONE (`fig1_framework.png`, `build_fig1_framework.py`).
2. **Table 3 human-validation** — DONE (66% / 60% inter-rater; culprit 80% vs none 33%; adjudication
   17/8/0). Numbers in `claim_evidence.md` + `validation/table3.md`.
3. **Natural slice rows** — DONE (3 cells, Fig 12).

All main-body figures and tables now exist or have their numbers. Remaining before submission: HF
dataset release (needs token); neutral spot-check of the 8 ambiguous adjudication cases; prose.

## Status of every numbered result (for the author)
- Locked & in `claim_evidence.md`: Pillars A, B, C, D, E; C1, C1b, C2, C3; dose-response; credit
  dilution; structure; error-direction; abstention; oracle robustness; taxonomy; frontier; natural (1 cell).
- Pending data: natural (2 cells), human adjudication (Table 3), HF dataset release (needs token).
