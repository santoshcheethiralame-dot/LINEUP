# Stage 6 — Scorer

## Objective

Stage 6 produces the paper's core evidence. It joins each method's predictions (Stage 5) to the oracle's role labels (Stage 4) and asks, for every method, whether it found the true culprit and how often it blamed a salient-but-innocent chunk instead. This is the first and only stage at which a method's output meets the ground truth; up to here the two were kept in separate files precisely so that the methods were never scored against labels they could see.

## Metrics

The metrics are chosen to be **scale-safe** — different methods produce scores on wholly different scales (ContextCite's Lasso weights, a token-overlap fraction, a one-hot judge vote), so the scorer never compares a raw score across cases or across methods.

- **Top-1 culprit accuracy.** The fraction of cases in which a method's argmax pick is a true culprit. Using the argmax sidesteps any threshold.
- **Predicted-role distribution.** For each method, the true role of the chunk it picked, tallied across cases — culprit, misleading, silent, or inert. This is the 2×2 confusion of the prediction, and it is the empirical form of the figure the project is organised around.
- **Misleading-as-culprit rate.** The *misleading* entry of that distribution: how often a method's top pick is a chunk that looks responsible but is not the cause. This is the headline number — the failure the benchmark was built to measure.
- **Culprit-over-misleading win-rate.** The crux distinction, scored as a within-case pairwise comparison: over every (culprit, misleading) pair *inside the same case*, the fraction in which the method scored the culprit above the decoy (ties counting a half). Because the comparison is always within one case, it never depends on the score's scale, and it is exactly the AUROC for telling the true cause from the look-alike. A genuinely causal method approaches one; a method that conflates salience with causation sits near a half or below.

Top-1 accuracy and the predicted-role distribution use the single argmax prediction; the win-rate uses the full per-chunk scores. Per-role precision, recall, and F1 follow directly from the role distribution and are left to downstream analysis rather than fixed here, so no thresholding choice is baked into the headline.

## Scope

The roles and predictions are joined by question id, and the meaningful population is the organically wrong cases — there is no error to attribute on a case the model answered correctly — so the scorer can restrict to them. Predictions for a case absent from the labels are ignored rather than guessed.

## Output

The scorer prints and writes a results table, one row per method, with the case count, top-1 culprit accuracy, misleading-as-culprit rate, and culprit-over-misleading win-rate. On request it also renders a stacked bar of the predicted-role distribution per method — where each method's predicted culprit actually lands — which is the empirical version of the 2×2 plot. The table is the paper's core table; the figure is its core figure.

## CPU-only and testable

The scoring logic is pure Python over the saved JSONL: no model, no GPU, no heavyweight dependencies (the figure's `matplotlib` is imported only when a figure is requested). It is therefore fully unit-tested — a method that ranks the culprit first scores a perfect top-1 and win-rate, while one that ranks the near-miss first scores zero on both and a misleading-as-culprit rate of one — and runs anywhere.

## Relation to the cited work

- **RAGOrigin** (arXiv:2509.13772) supplies the culprit-identification metric templates — detection accuracy and the false-positive and false-negative framing the role distribution makes concrete.
- **RAGChecker** (Ru et al., 2024) frames noise sensitivity, the comparator for the salience axis that the win-rate operationalises.
- **The RAG error taxonomy E1–E9** (arXiv:2510.13975) locates the four roles within the broader space of known RAG failures, situating the misleading-as-culprit failure among them.
