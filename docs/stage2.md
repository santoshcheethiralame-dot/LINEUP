# Stage 2 — Scenario Builder

## Objective

Stage 2 assembles each test case. Given an example from the data foundation — a question, its gold answer, the gold supporting chunks, and a pool of distractors — it produces the retrieved context the model will actually see: the gold chunks, a controlled number of distractors, and exactly one constructed **near-miss** chunk, presented in randomized order. The output is a set of raw, unlabelled test cases; the causal role of each chunk is decided later by the oracle (Stage 4). What this stage fixes is the *material* of each case and, crucially, the construction of the misleading chunk that the rest of the benchmark turns on.

## The misleading chunk, and why it is the core

The benchmark is organised around a 2×2 of counterfactual effect against apparent salience. Three of the four cells are supplied for free by ordinary multi-hop data: gold chunks that carry the answer, distractors that do not, and chunks the model ignores. The cell that does not occur naturally in a controlled, labelled form is the **misleading** chunk — one that *looks* responsible for an answer without *being* its cause. To measure whether an attribution method can avoid blaming such a chunk, the benchmark has to contain such chunks by design. Constructing them is therefore the scientific core of the stage.

## What counts as a near-miss

A near-miss is deliberately positioned between two failure modes that are easier to build but uninteresting here:

- It is **not random noise.** An unrelated passage is trivially non-causal and trivially non-salient; it lands in the *inert* cell and tests nothing. The relevant finding from *The Power of Noise* (Cuconasu et al., arXiv:2401.14887) is that semantically related but wrong passages are the ones that actually damage RAG answers, so a useful misleading chunk must be topically adjacent, not off-topic.
- It is **not adversarial injection.** The construction does not echo the question, issue instructions, or plant a flag the way an attack would; an injected signature is exactly what organic errors lack, and what poison-traceback methods key on. The template of injecting a distracting statement is borrowed from Shi et al. (*Distracted by Irrelevant Context*, arXiv:2302.00093), but made subtler: a single plausible wrong value inside otherwise-faithful prose.

In short, a near-miss is a passage that is topically indistinguishable from the gold evidence and reads as authoritative, but asserts a believable wrong value.

## Construction by value substitution

The reference construction clones the gold paragraph that states the answer and replaces the answer with a plausible, same-type wrong value. It proceeds in three steps.

**Locating the answer.** The builder searches the gold chunks for the sentence that states the answer, preferring the annotated supporting sentences and falling back to any gold sentence. Matching is case-insensitive and anchored at word boundaries only where the answer begins or ends with an alphanumeric character, so an answer containing punctuation (such as "U.S.") is still found while a short answer is not matched inside a longer word. If no gold sentence contains the answer, the example is not eligible for substitution.

**Choosing the wrong value.** The answer is typed as a year, a number, or an entity. Years are shifted by a small plausible offset; numbers are scaled by a plausible factor; entities are sampled from the pool of real same-type answers observed elsewhere in the corpus, preferring candidates of the same surface shape (token count and capitalisation) and excluding the answer itself and its sub- or super-strings. Drawing the replacement from genuine same-type answers keeps it believable rather than invented, and keeps the construction free of any external model.

**Building the host paragraph.** Every occurrence of the answer in the source gold paragraph is replaced by the wrong value, so the resulting paragraph is internally consistent. It retains the source title — modelling the common and entirely organic case of a stale or duplicated copy of the same article sitting in the index alongside the current one — and is marked with provenance `misleading`. Because the chunk is identical to the gold evidence except for the value it asserts, it is maximally confusable with it: a salience-based method will score it almost exactly as it scores the gold chunk, which is precisely what makes the *misleading*-versus-*culprit* distinction hard.

**The intended wrong answer is recorded.** The replacement value is the answer the misleading chunk is built to induce. Storing it gives the later stages a concrete hypothesis: if the model emits that value, the misleading chunk is implicated, and the oracle's counterfactual test can confirm or refute it.

## Eligibility and scoping

Not every question admits a clean substitution. Boolean (`yes`/`no`) answers have no value to perturb inside a sentence, and some answers do not appear verbatim in the gold context. These cases are detected up front and excluded, with the reason recorded, so that every constructed near-miss genuinely differs from its gold source by exactly one believable value. This keeps the misleading set clean at the cost of coverage, which is the right trade for a benchmark whose value rests on the integrity of its labels.

## Assembling the scenario

The context is composed of the gold chunks, the one misleading chunk, and enough distractors to reach a target size `k` (default ten, matching the HotpotQA distractor setting). Distractors are drawn from the example's own pool. The chunks are then shuffled into a random presentation order. Order randomisation is a requirement rather than a detail: *Lost in the Middle* (Liu et al., arXiv:2307.03172) shows that a model's use of a passage depends on where it sits in the context, so a fixed position would confound which chunk the model relies on with where it happened to be placed.

## Reproducibility

Determinism is enforced per case. Each question draws a private random stream seeded from a stable digest of the base seed and the question id — the built-in `hash` is salted per process and would not reproduce across runs, so the seed is derived from SHA-256 instead. The same stream drives both the choice of wrong value and the shuffle, so a fixed corpus and seed regenerate an identical benchmark. Every case also carries a **recipe**: the seed, the target size, the original and substituted values and the substitution type, the source chunk and sentence, the identifiers of the gold, misleading, and distractor chunks, and the final presentation order. The recipe is sufficient to reconstruct the case and to audit exactly how it was made.

## Provenance is not role

Each chunk records its **provenance** — gold, distractor, or misleading — which states where it came from. This is kept distinct from its causal **role** (culprit, misleading, silent, inert), which is not yet known: a constructed near-miss is built to mislead, but whether it actually causes a given model to err is an empirical question answered by the oracle. The stage therefore emits unlabelled cases with provenance annotations, not role labels.

## Output

The stage delivers raw test cases, written as JSONL with their full recipes, ready for generation in Stage 3. `scripts/build_scenarios.py` builds a batch, reports how many cases were constructed and why any were skipped, and prints one assembled case showing the randomized order of gold, misleading, and distractor chunks.

## Tests

The substitution logic, the near-miss construction, the eligibility checks, the assembled-context invariants (one misleading chunk, the gold chunks present, the target size met), determinism under a fixed seed, and the serialization round-trip are all covered by fast, model-free unit tests over hand-constructed fixtures.

## Limitations and planned extensions

The reference construction is the value-substituted near-duplicate. The builder is written behind a small strategy interface so that two planned alternatives slot in without disturbing the rest of the pipeline: a *topical-host* strategy that places the wrong value inside a genuinely different but related article rather than a clone of the gold, and a *paraphrase* strategy that rewrites the near-duplicate for surface diversity. Boolean and non-appearing answers are out of scope for substitution and are excluded rather than forced. These choices are deliberate: one rigorous, auditable construction now, with clearly marked seams for the variations the paper's ablations will need.

## Relation to the cited work

- **The Power of Noise** (Cuconasu et al., arXiv:2401.14887) defines the near-miss: semantically related but wrong passages are what hurt, so the misleading chunk is topically adjacent rather than noisy.
- **Distracted by Irrelevant Context** (Shi et al., arXiv:2302.00093) supplies the injection template, made subtler here — organic, not blatant.
- **Lost in the Middle** (Liu et al., arXiv:2307.03172) motivates randomising chunk position so order does not confound which chunk is used.
- **RAGonite / ConfQuestions** (Roy et al., 2024) is precedent for sourcing conflicting or misleading evidence in a controlled way.
