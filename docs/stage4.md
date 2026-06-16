# Stage 4 — Counterfactual Oracle

## Objective

Stage 4 assigns every chunk in every case its true role in the 2×2 — culprit, misleading, silent, or inert — by exact leave-one-out. This is the stage that turns a pile of constructed cases into a labelled benchmark, and it is the one the entire project's credibility rests on: if these labels are trustworthy, the later comparison of attribution methods means something; if they are not, nothing downstream does. The output is the benchmark dataset.

## The two axes

Each chunk is placed on two axes that are measured independently.

**Counterfactual effect (the causal axis).** The chunk is removed from the context, the model is re-run on what remains, and the new answer is compared to the original. If the answer changes, the chunk was causal — it was load-bearing for the answer the model actually gave. This is a threshold-free, directly interpretable criterion that matches the build plan's "answer shift". The stage also records the stronger event of the answer becoming *correct* once the chunk is removed (the wrong→right flip that pinpoints an error's cause) and, as a continuous measure, the drop in the original answer's log-probability when the chunk is removed.

**Apparent salience (the looks-responsible axis).** A chunk is salient if it actually contains the model's answer — if the value the model produced is stated in the chunk. This is the surface cue a salience-based method, or a person skimming, would latch onto, and it is again transparent and threshold-free. Keeping arbitrary cutoffs out of the ground truth is deliberate: a label that depended on a tuned threshold would be a label a reviewer could argue with.

## Why the labels are non-circular

Both axes are derived from only two things: the **known gold answer** (annotated in Stage 1, preserved through Stage 2) and the **model's own behaviour** under removal. Neither axis consults any attribution or traceback method. The methods are what Stage 5 will run and Stage 6 will grade *against* these labels; because the labels never saw those methods, the evaluation is not circular. Exact leave-one-out on cases with a known answer is, in this sense, the clean ground-truth version of the random-ablation approximation that ContextCite computes — which is exactly why a benchmark built this way can adjudicate it.

## The four roles

Crossing the two axes gives the quadrants:

- **culprit** — causal and salient: removing it changes the answer, and it holds the answer. The chunk a good method should name.
- **misleading** — salient but not causal: it holds the answer and *looks* responsible, but removing it changes nothing. The trap; a method that confuses correlation with cause blames it.
- **silent** — causal but not salient: removing it changes the answer, yet it does not visibly contain it. The false negative a salience cue misses.
- **inert** — neither.

The constructed near-miss from Stage 2 can land as either *culprit* (the model genuinely followed it, so removing it repairs the answer) or *misleading* (the model was wrong for some other reason, so the near-miss only looks guilty). Which of the two it becomes is an empirical fact about the model, decided here rather than assumed — and the split between them is among the benchmark's most informative statistics.

## Removal semantics, determinism, and cost

A chunk is removed entirely and the remaining chunks are re-presented; this is the canonical leave-one-out, and it inherits the same position sensitivity any ablation does, which is one more reason Stage 2 randomises order. Generation is greedy and the machine and model are fixed, so the roles are reproducible. This is the most expensive stage: each case costs one regeneration and one teacher-forced score per chunk, so a ten-chunk case is roughly ten model runs. It must therefore run on a single GPU and one model revision, and labelling only the organically wrong cases (with a control of correct ones) is the economical default.

## Output

For each case the stage emits a `CaseRoles` record: the question, the gold and original answers, whether the original was correct, and a `ChunkRole` for every chunk carrying its provenance, its assigned role, the two axis booleans, the post-removal correctness, the log-probability delta, and the answer the model gave once the chunk was removed. Records are written as JSONL. Every signal behind a role is stored, so a later analysis can re-threshold or re-stratify without re-running the model.

## CPU-testability

The oracle is expressed entirely against the abstract `LanguageModel` interface, so the role logic is exercised on CPU against fake models with no GPU. A model that answers wrongly only while the near-miss is present yields that chunk as a *culprit*; a model that answers wrongly regardless yields it as *misleading*. Those two fixtures pin the central distinction the benchmark exists to measure.

## Relation to the cited work

- **ContextCite** (Cohen-Wang et al., arXiv:2409.00729) approximates a chunk's effect with random ablations and a linear surrogate; exact leave-one-out here is the ground-truth effect that approximation targets, which is why the benchmark can score it.
- **RAGonite** (Roy et al., 2024) attributes by counterfactual removal and validates that the signal reflects causation — the same removal-is-causation principle, used here to define ground truth rather than to attribute.
- **Influence functions** (Koh & Liang, ICML 2017) are the classical "which input caused this output" backbone; leave-one-out is the exact, model-agnostic instance of that question.
