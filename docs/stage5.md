# Stage 5 — Method Runners

## Objective

Stage 5 runs the existing attribution methods that the benchmark exists to test. Each method is a runner that, given a scenario and the model's answer, produces a score for every chunk — higher meaning more responsible — and names a predicted culprit as the highest-scoring chunk. These predictions are what Stage 6 grades against the Stage 4 ground truth; the spread of methods is chosen to put a salience-based method, a contributive-attribution method, and the model's own judgement side by side.

## Interface and output

Every method implements one operation, `score_chunks`, returning a per-chunk score aligned to the scenario's chunks; the runner turns that into a `MethodPrediction` with the per-chunk scores and the argmax culprit. Predictions are written as JSONL, joinable to the role labels by question id, so a method never needs to see — and the benchmark never lets it see — the oracle's answers. That separation is what keeps the evaluation honest: the methods are scored against labels they had no access to.

## ContextCite

ContextCite is the headline method and the one whose behaviour the project set out to probe. It samples random ablations of the context — each chunk included with probability one half — teacher-force-scores the *fixed* original answer under each ablation to get the answer's log-probability, and fits a sparse linear surrogate (Lasso) of that log-probability against the binary inclusion mask. The learned weight on each chunk is its attribution; the predicted culprit is the largest weight.

The construction in Stage 2 is what makes this interesting. The near-miss is a near-duplicate of the gold chunk, so the two have strongly correlated ablation effects: including either tends to move the answer's probability in similar ways. A sparse linear surrogate fit from a finite sample of ablations cannot cleanly separate two collinear sources, so it can split or misplace the weight between gold and near-miss — attributing to a chunk that is salient rather than causal. The benchmark is built to surface exactly that failure, which is why exact leave-one-out, not ContextCite, is used for the ground truth.

## Lexical similarity

A deliberately simple salience baseline: the token overlap (F1) between each chunk and the question together with the answer. It uses no model. Because the near-miss is a near-duplicate of the gold evidence and, on a fooled case, contains the very value the model produced, it scores at or near the top — so this method is expected to blame the misleading chunk often. It stands in for the broad family of methods that rank by how related a passage looks.

## LLM-judge culprit

The model is shown the numbered context, the question, and its own answer, and asked which single passage is most responsible. The reply's number selects the predicted culprit. This is the model's own attribution of its behaviour, and it is interesting precisely because a model that was misled may also rationalise by pointing at the passage that looks most relevant.

## What is deliberately not a separate method

The build plan also lists traceback methods, an attention-mass baseline, and a coalition/Shapley runner. Two scoping decisions are worth stating. First, a sanitize-and-replay or remove-and-replay traceback, applied to an *organic* error, reduces to exact leave-one-out — which is the oracle itself; running it as a "method under test" would be scoring the ground truth against itself, so it is not included as one. The adversarial responsibility scores it descends from rely on injected-text signatures that organic errors do not carry, which is the degradation the project predicts rather than a baseline to reimplement faithfully here. Second, an attention-mass baseline needs model internals the backend does not currently expose, and the Shapley runner is the later interaction study; both slot in behind the same `AttributionMethod` interface without disturbing the rest. Building three methods that produce genuinely distinct signals now is worth more than a wider but shallower set.

## CPU-testability, determinism, and cost

Each method depends only on the abstract `LanguageModel` interface, so all of them are exercised on CPU against fake models: a model whose answer probability rises only when the near-miss is present yields the near-miss as ContextCite's top attribution; a model that picks a passage number yields that passage; the lexical baseline needs no model at all. ContextCite's ablation sampling is seeded per question, so its scores are reproducible. Cost is dominated by ContextCite's ablations — one teacher-forced score each — and the single judge generation; the lexical baseline is free.

## Tests

The lexical baseline's preference for the answer-bearing chunk, the judge runner's parsing of the chosen number, the runner's argmax and per-chunk bookkeeping, and the prediction serialization round-trip are covered by fast tests. ContextCite is covered by a test that gates on `scikit-learn`: against a model whose answer is supported only by the near-miss, the Lasso surrogate must place its largest weight there.

## Relation to the cited work

- **ContextCite** (Cohen-Wang et al., arXiv:2409.00729) is the specification for the first method — random ablations, a log-probability target, and a Lasso surrogate whose weights are the attribution.
- **RAGCharacter** (arXiv:2605.01782) and **RAGOrigin** (arXiv:2509.13772) are the traceback methods whose organic-setting behaviour motivates leaving exact-removal traceback to the oracle.
- **RAGChecker** (Ru et al., 2024) frames noise sensitivity, the salience family the lexical baseline represents.
- **TokenShapley** (arXiv:2507.05261) and **RepoShapley** (arXiv:2601.03378) are the coalition-attribution methods the interface is kept open for.
