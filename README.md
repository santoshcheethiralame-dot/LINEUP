# lineup

A controlled benchmark for **organic chunk-role attribution** in retrieval-augmented generation.

When a RAG system answers confidently but wrongly, the failure usually traces back to a single retrieved passage. Existing tools tell you *that* an answer is unfaithful; they do not reliably tell you *which* chunk caused it, and they are easiest to fool when the offending passage is not adversarial but simply stale, near-duplicate, or topically adjacent. This project builds a benchmark whose chunk roles are known by construction, so attribution and traceback methods can be measured on whether they recover the true causal chunk.

Each retrieved chunk sits on two axes — its **counterfactual effect** (does removing it change the answer?) and its **apparent salience** (does it look responsible?) — which gives four roles:

|              | salient    | not salient |
| ------------ | ---------- | ----------- |
| **causal**   | culprit    | silent      |
| **not causal** | misleading | inert     |

The hard case is the *misleading* chunk: salient but not causal. A method that blames it has confused correlation with cause.

## Status

This repository currently covers the first two stages of the build:

- **Stage 0 — infrastructure and model access.** A model backend that returns both generated text and teacher-forced token log-probabilities ([docs/stage0.md](docs/stage0.md)).
- **Stage 1 — data foundation.** A loader over multi-hop QA that yields, per question, the gold answer, the gold supporting chunks, and a pool of realistic distractors ([docs/stage1.md](docs/stage1.md)).

## Layout

```
src/lineup/
  config.py        run configuration and seeding
  backends/        language-model backends (generation + scoring)
  data/            dataset schema, HotpotQA loader, distractor retriever
scripts/           runnable entry points for each stage
tests/             unit tests
docs/              per-stage write-ups
```

## Setup

Python 3.10+ and a CUDA-capable GPU are assumed for the default 7B model. Create the environment outside any cloud-synced folder.

```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

PyTorch is installed first, from its CUDA index, because the version in `requirements.txt` deliberately omits it.

## Quickstart

Generate an answer and inspect its teacher-forced log-probabilities:

```
python scripts/hello_world.py
```

Load one multi-hop question with its gold chunks and retrieved distractors:

```
python scripts/inspect_example.py
```

## Tests

```
pytest             # fast unit tests
pytest -m slow     # also runs backend tests, which fetch a tiny model
```

## Notes

Generation is greedy (temperature 0) and seeded, so a fixed model and precision give reproducible log-probabilities. The default backend uses Hugging Face `transformers`; the backend interface is small enough that a vLLM or hosted-API backend can be substituted without touching the rest of the pipeline.
