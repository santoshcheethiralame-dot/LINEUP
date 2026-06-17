# Stage 7 — Downstream Experiment

## Objective

Stages 4–6 establish a negative result: attribution methods are fooled by the near-miss. Stage 7 asks whether that failure has a cost on a task someone would actually build — **selective question answering**, where a system answers when it is confident and abstains otherwise. The natural idea is to abstain using attribution ("if the answer is well-localised to good evidence, trust it"). This stage tests whether that works.

## The task

Selective prediction. Each case carries the model's answer and whether it was correct (Stage 3). A confidence signal orders the cases; the system answers the most confident first and abstains on the tail. A good signal pushes the wrong answers to the bottom, so the answered set stays accurate as coverage grows. The whole experiment is a join over saved outputs and runs on CPU.

## The confidence signals

Three families are compared on the same cases.

- **Self log-probability.** The model's own mean answer log-probability — the baseline a practitioner already has, with no attribution at all.
- **Attribution margin.** Per method, the gap between its top and second chunk scores: how *decisively* it localises a culprit. The premise under test is that a confident, peaked attribution means a trustworthy answer.
- **Oracle.** An upper bound that uses the role labels: a correct answer is trustworthy, and a wrong answer that some chunk-removal turns correct is a detectable, fixable error to abstain on. This is what a *perfect* causal account could flag, and it bounds how well any attribution-based abstention could do.

## Metrics

- **AUROC** of the confidence signal against correctness — how well it separates correct answers from wrong ones (one half is chance).
- **AURC**, the area under the risk–coverage curve — the selective-prediction quality across all coverage levels, lower being better.
- **Selective accuracy at fifty per cent coverage** — the accuracy on the most-confident half.

## What it surfaces

The result the benchmark predicts is that **attribution margin is close to chance**. A method that is fooled by the near-miss is just as decisive on a wrong answer — it points confidently at the look-alike — as on a right one, so its decisiveness carries little information about correctness. The one-hot LLM judge is a sharper case: it always names exactly one passage, so its margin is constant and its AUROC is exactly chance. The **oracle**, by contrast, scores well, because it flags the fixable errors directly. The gap between them is the cost: reliable abstention is available *if* attribution is causal, and the methods under test do not deliver it. That turns the negative result into a concrete reason to get attribution right.

## Scope

Unlike the Stage 6 headline, which is meaningful only on the wrong cases, this experiment needs the full set — correct answers included — because separating correct from wrong is the whole task. The model stages are therefore run without the wrong-only filter when the abstention experiment is wanted.

## Output

A table, one row per signal, with the AUROC, AURC, and selective accuracy, plus an optional risk–coverage figure overlaying the signals. The contrast between the oracle curve and the attribution-margin curves is the figure that makes "this matters" concrete.

## CPU-only and testable

The whole stage is pure Python over the saved JSONL — no model, no GPU — so the AUROC and risk–coverage computations, the margin and oracle signals, and the end-to-end evaluation are unit-tested: a perfectly ordered signal scores an AUROC of one and a low AURC, a constant signal scores a half, and the oracle flags a fixable error while leaving a correct answer trusted.

## Relation to the cited work

- **Self-RAG** (Asai et al., 2023) and **CRAG** (Yan et al., 2024) are the abstention and corrective-retrieval systems this selective-QA framing stands in for; the experiment asks whether attribution could feed such a controller.
- **RAGChecker** (Ru et al., 2024) frames noise robustness, the lens under which an abstention signal that survives a misleading chunk would matter.
