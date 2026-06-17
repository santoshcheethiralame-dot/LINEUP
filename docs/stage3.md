# Stage 3 — Generation and Correctness

## Objective

Stage 3 runs the model on each assembled case and labels the answer correct or wrong. This is where the phenomenon the benchmark is about — a model answering confidently but wrongly because of a bad chunk — actually occurs. The output is, for every case, the model's answer and a correctness verdict; the confidently-wrong cases are the ones the later stages dissect, and the correct ones form a control.

## Rendering a scenario into a prompt

Each scenario is turned into a chat prompt: a system instruction, then a user turn containing the numbered context blocks followed by the question. The chunks are presented in the scenario's stored order, so the randomization decided in Stage 2 is preserved exactly. The system instruction asks for the shortest possible answer — a name, date, or phrase — which keeps the model's output close to the reference form and makes automatic correctness judging tractable. The prompt builder is a pure function of the scenario, so the same case always produces the same prompt.

## Generation

Generation uses the Stage 0 backend in its greedy, temperature-0 setting, so a fixed model and machine give a reproducible answer. Alongside the text, the backend reports the per-token log-probabilities of the generated answer; their mean is recorded as a confidence proxy, which is what lets a later analysis separate *confidently* wrong answers from hesitant ones.

## Judging correctness

Automatic correctness for short-answer QA has a well-known pitfall, noted in *The Power of Noise*: a strict string comparison marks "President Roosevelt" wrong when the reference is "Roosevelt". The stage therefore judges in two tiers:

1. **Normalized exact match** — lowercase, strip punctuation and articles, fold whitespace (the SQuAD convention), then compare. This tier is high-precision and model-free.
2. **LLM judge** — for answers that miss exact match, an LLM judge is asked whether the model answer is correct given the reference, and replies yes or no. This recovers genuine phrasing variants without loosening the exact tier into the false positives that naive substring matching would admit. The verdict is read from the first word of the reply, so "Yes, correct." is parsed as readily as a bare "yes".

Each verdict records which tier decided it. The judge is optional: with it disabled the stage falls back to exact match alone, which is the fast path used when developing on CPU. By default the judge is the same model that produced the answer; because the exact tier settles the bulk of cases and the judge only adjudicates phrasing variants, the scope for self-preference bias is narrow, and since the judge accepts any backend a separate or stronger model can be substituted where the compute allows. An empty answer is counted wrong outright, without a judge call.

## The planted-value signal

Beyond correct-or-wrong, the stage records whether the model's answer echoes the **intended wrong answer** — the value the Stage 2 misleading chunk was built to induce. The match is on whole tokens, so a short planted value such as a year is not spuriously found inside a longer number. This is not a correctness label; it is a diagnostic. When a wrong answer matches the planted value, it is strong evidence that the misleading chunk, and not some other distractor, drove the error — a hypothesis the Stage 4 oracle then confirms or refutes by counterfactual removal.

## Output

For each case the stage emits a `GenerationResult`: the question, the gold answer, the model's answer, the correctness verdict and which tier produced it, the planted-value flag, and the answer's mean log-probability. Results are written as JSONL, joinable to the scenarios by question id, so the two stages stay decoupled — Stage 3 reads scenarios and writes generations, and nothing larger than identifiers needs to move between them.

## Reproducibility and where it runs

Generation is the first stage that needs the GPU, and it must run on a single machine and pinned model revision so the labels are internally comparable. Everything around the model, however, is expressed against the abstract `LanguageModel` interface, so the entire orchestration — prompt building, the two-tier judging, the planted-value diagnostic — is exercised by unit tests against a fake model on CPU, with no GPU or model download. Only the final run uses the real model.

## Tests

The prompt builder, the normalization and matching functions, the LLM-judge parsing, and the end-to-end generate-and-judge flow are all covered by fast tests over a fake backend that returns canned answers: a correct answer is labeled by exact match, a planted wrong value is flagged, and a phrasing variant is recovered by the judge.

## Limitations

Correctness on the variants exact match misses rests on a model judge, and by default that judge is the answering model itself. The bias is bounded — exact match settles the bulk of cases and the judge only adjudicates phrasing — but a separate judge is the cleaner choice where the compute allows, and the interface accepts one. An abstention ("I don't know") is counted simply as wrong rather than as its own category; the recorded answer log-probability is what later lets a confident error be told apart from a hesitant one.

## Relation to the cited work

- **Lewis et al., 2020** (arXiv:2005.11401) is the retrieval-augmented generation setup being run here.
- **RAGAS** (Es et al., 2023) is the source of answer-correctness conventions, and later the detection-only baseline this work contrasts against.
- **The Power of Noise** (Cuconasu et al., arXiv:2401.14887) supplies the binary correctness protocol and the "Roosevelt" pitfall that motivates the two-tier judge.
