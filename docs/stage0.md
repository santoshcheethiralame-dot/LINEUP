# Stage 0 — Infrastructure and Model Access

## Objective

Stage 0 establishes the engineering substrate for the benchmark: a reproducible Python package and a language-model backend that exposes not only generated text but also **teacher-forced token log-probabilities**. The latter requirement is not incidental. Both the counterfactual oracle (Stage 4) and the contributive-attribution method under test (ContextCite; Cohen-Wang et al., arXiv:2409.00729) operate by measuring how the probability the model assigns to a *fixed* response changes as the retrieved context is perturbed. A backend that can only return sampled text is therefore insufficient; the scoring interface is the load-bearing component of the entire pipeline.

## Repository and Environment

The package is organised under a `src/` layout and installed in editable mode. Runtime dependencies are version-pinned in `requirements.txt`; PyTorch is installed separately from its CUDA index so that the GPU build is selected explicitly rather than resolved by chance. Determinism is enforced in two ways: fixed seeds (`lineup.config.set_seed`) applied to Python, NumPy, and PyTorch, and greedy decoding, which removes sampling as a source of run-to-run variation.

## The Backend Abstraction

All model access passes through a single abstract interface, `LanguageModel`, with two operations:

- `generate(messages)` returns the greedy continuation together with the per-token log-probabilities of the generated tokens.
- `score(messages, response)` returns the per-token log-probabilities of a **caller-supplied** response under teacher forcing.

Keeping this surface small is deliberate. Every later stage — perturbing the context and re-scoring a fixed answer, computing leave-one-out effects, running attribution baselines — is expressible in terms of `score`, and any alternative backend (vLLM on Linux, or a hosted API that exposes log-probabilities) can be introduced by implementing these two methods alone.

## Teacher-Forced Scoring

The reference backend is built on Hugging Face `transformers`. For an instruction-tuned model the prompt is assembled with the tokenizer's chat template and the generation prefix; for a base model without a template the messages are concatenated as a plain transcript. The supplied response is tokenized without special tokens and appended to the prompt, and a single forward pass yields the logits for the full sequence.

The one subtlety is alignment. In a causal language model the distribution that predicts the token at absolute position *p* is emitted at position *p − 1*. Consequently the log-probability of the *i*-th response token, whose absolute position is `prompt_len + i`, is read from the distribution at `prompt_len + i − 1`. This indexing is isolated in a small pure function, `_gather_response_logprobs`, so that it can be unit-tested independently of any model.

A second, deliberate, modelling choice concerns tokenization: the response is tokenized independently of the prompt rather than by re-tokenizing the concatenated string. This matches the convention used by the attribution method the benchmark evaluates and keeps response token boundaries stable as the surrounding context is perturbed, at the cost of a possible boundary effect on the first response token. The trade-off is recorded here so that it is visible at the point where it matters.

## Generation and Determinism

Generation uses greedy decoding (`do_sample=False`, a single beam), which is the temperature-0 setting and is deterministic given a fixed model and numerical precision. The per-token log-probabilities reported alongside the generated text are taken from the model's own decoding scores; for greedy decoding without additional logits processors these coincide, up to floating-point error, with the teacher-forced scores produced by `score` — a property the `hello_world` script exercises end to end.

## Why `transformers` rather than vLLM

The build plan names vLLM as a candidate server. vLLM has no supported native Windows build, and — more to the point — the property the project actually requires is exact teacher-forced scoring of an arbitrary response, which a direct forward pass in `transformers` provides cleanly. The backend interface does not preclude a vLLM implementation on Linux; it simply is not the reference path. The default model is `Qwen/Qwen2.5-7B-Instruct`, an openly licensed 7B instruction-tuned model that requires no gated access.

## Output

The stage delivers a clean, installable repository in which `scripts/hello_world.py` produces a greedy answer to a context-grounded question and prints the teacher-forced log-probability of each answer token. This is the concrete artefact the plan specifies as the Stage 0 output: a model that generates and returns log-probabilities.

## Tests

The alignment logic is covered by a fast, model-free unit test over hand-constructed tensors. Backend behaviour — deterministic greedy generation, and deterministic, finite scoring — is covered by tests marked `slow`, which fetch a small model from the Hub and are excluded from the default test run.

## Relation to the cited work

- **ContextCite** (Cohen-Wang et al., arXiv:2409.00729) motivates the scoring interface: its attribution scores are derived from changes in response log-probability under context ablation, which is impossible without teacher-forced log-probabilities.
- **TokenShapley** (arXiv:2507.05261) informs the choice of an open 7–8B instruction-tuned model as the default backend.
