# lineup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santoshcheethiralame-dot/LINEUP/blob/main/notebooks/lineup_colab.ipynb)

A controlled benchmark for **organic chunk-role attribution** in retrieval-augmented generation.

When a RAG system answers confidently but wrongly, the failure usually traces back to a single retrieved passage. Existing tools tell you *that* an answer is unfaithful; they do not reliably tell you *which* chunk caused it, and they are easiest to fool when the offending passage is not adversarial but simply stale, near-duplicate, or topically adjacent. This project builds a benchmark whose chunk roles are known by construction, so attribution and traceback methods can be measured on whether they recover the true causal chunk.

Each retrieved chunk sits on two axes — its **counterfactual effect** (does removing it change the answer?) and its **apparent salience** (does it look responsible?) — which gives four roles:

|              | salient    | not salient |
| ------------ | ---------- | ----------- |
| **causal**   | culprit    | silent      |
| **not causal** | misleading | inert     |

The hard case is the *misleading* chunk: salient but not causal. A method that blames it has confused correlation with cause.

## Status

This repository currently covers the first five stages of the build:

- **Stage 0 — infrastructure and model access.** A model backend that returns both generated text and teacher-forced token log-probabilities ([docs/stage0.md](docs/stage0.md)).
- **Stage 1 — data foundation.** A loader over multi-hop QA that yields, per question, the gold answer, the gold supporting chunks, and a pool of realistic distractors ([docs/stage1.md](docs/stage1.md)).
- **Stage 2 — scenario construction.** For each question, the retrieved context is assembled from the gold chunks, distractors, and one constructed organic near-miss chunk, in randomized order and with a reproducible recipe ([docs/stage2.md](docs/stage2.md)).
- **Stage 3 — generation and correctness.** Run the model on each case, label the answer correct or wrong (exact match plus an LLM judge), and flag when the answer echoes the planted misleading value ([docs/stage3.md](docs/stage3.md)).
- **Stage 4 — counterfactual oracle.** Assign every chunk its true role (culprit / misleading / silent / inert) by exact leave-one-out — the ground-truth labels the benchmark is built to provide ([docs/stage4.md](docs/stage4.md)).

## Layout

```
src/lineup/
  config.py        run configuration and seeding
  backends/        language-model backends (generation + scoring)
  data/            schema, HotpotQA loader, distractor retriever, near-miss construction, scenario assembly
  textnorm.py      shared answer normalization
  prompt.py        render a scenario into a prompt
  correctness.py   answer matching and the LLM judge
  generation.py    run the model and label correctness
  oracle.py        leave-one-out role assignment
scripts/           runnable entry points for each stage
notebooks/         cloud-GPU notebook for the model stages
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

## Running the model stages

Only Stages 3–5 need a GPU; the data and scenario stages, the tests, and the scorer run on CPU. Without a local GPU, run the model stages on a free Colab or Kaggle T4 (16 GB) using the notebook in `notebooks/`, which loads the 7B model in 4-bit. See [docs/running.md](docs/running.md) for the full compute split and how a second machine reproduces the run.

## Quickstart

Generate an answer and inspect its teacher-forced log-probabilities:

```
python scripts/hello_world.py
```

Load one multi-hop question with its gold chunks and retrieved distractors:

```
python scripts/inspect_example.py
```

Build a batch of test cases (each is gold chunks + distractors + one near-miss):

```
python scripts/build_scenarios.py
```

Run the model on the cases and label them correct or wrong (needs a GPU — use the Colab notebook):

```
python scripts/run_generation.py
```

Assign every chunk its ground-truth role by leave-one-out (needs a GPU):

```
python scripts/run_oracle.py --wrong-only
```

## Tests

```
pytest             # fast unit tests
pytest -m slow     # also runs backend tests, which fetch a tiny model
```

## Notes

Generation is greedy (temperature 0) and seeded, so a fixed model and precision give reproducible log-probabilities. Scenario construction is likewise seeded and model-free, so a fixed corpus and seed give an identical benchmark. The default backend uses Hugging Face `transformers`; the backend interface is small enough that a vLLM or hosted-API backend can be substituted without touching the rest of the pipeline.
