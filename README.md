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

This repository covers the build's nine stages (0 through 8):

- **Stage 0 — infrastructure and model access.** A model backend that returns both generated text and teacher-forced token log-probabilities ([docs/stage0.md](docs/stage0.md)).
- **Stage 1 — data foundation.** A loader over multi-hop QA that yields, per question, the gold answer, the gold supporting chunks, and a pool of distractor passages — HotpotQA's own distractor paragraphs, with a BM25 retriever included for custom corpora ([docs/stage1.md](docs/stage1.md)).
- **Stage 2 — scenario construction.** For each question, the retrieved context is assembled from the gold chunks, distractors, and one constructed organic near-miss chunk, in randomized order and with a reproducible recipe ([docs/stage2.md](docs/stage2.md)).
- **Stage 3 — generation and correctness.** Run the model on each case, label the answer correct or wrong (exact match plus an LLM judge), and flag when the answer echoes the planted misleading value ([docs/stage3.md](docs/stage3.md)).
- **Stage 4 — counterfactual oracle.** Assign every chunk its true role (culprit / misleading / silent / inert) by exact leave-one-out — the ground-truth labels the benchmark is built to provide ([docs/stage4.md](docs/stage4.md)).
- **Stage 5 — method runners.** Run the attribution methods under test (ContextCite, a lexical-similarity baseline, an LLM judge), each predicting a culprit per chunk ([docs/stage5.md](docs/stage5.md)).
- **Stage 6 — scorer.** Join the predictions to the oracle roles and compute the headline misleading-as-culprit rate, top-1 culprit accuracy, and the 2×2 confusion ([docs/stage6.md](docs/stage6.md)).
- **Stage 7 — downstream experiment.** A selective-QA / abstention study testing whether the attribution failure has a cost: can a confidence signal (self-confidence, attribution decisiveness, or an oracle upper bound) tell correct answers from wrong ones ([docs/stage7.md](docs/stage7.md)).
- **Stage 8 — release and write-up.** Package a run as a HuggingFace dataset with a data card and a reproducibility manifest, a one-command pipeline harness, and the paper outline ([docs/stage8.md](docs/stage8.md)).

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
  methods.py       attribution methods under test (ContextCite, baselines)
  scoring.py       grade predictions against the oracle roles
  downstream.py    selective-QA / abstention experiment
  release.py       package a run as a dataset with a data card
scripts/           runnable entry points for each stage
notebooks/         cloud-GPU notebook for the model stages
tests/             unit tests
docs/              per-stage write-ups
app/               streamlit case explorer over the saved JSONL (no GPU)
paper/             paper outline and claim-to-evidence map
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

Stages 3–5 need a GPU; the data and scenario stages, the scorer, the abstention study, and the tests run on CPU. Without a local GPU, run the model stages on a free Colab or Kaggle T4 (16 GB) using the notebook in `notebooks/`, which loads the 7B model in 4-bit. `notebooks/lineup_study.ipynb` runs the whole study at scale across two models and renders the result charts in one pass. See [docs/running.md](docs/running.md) for the full compute split and how a second machine reproduces the run.

## Explore the results

The `app/` directory is a small Streamlit explorer over the saved JSONL — an overview (the per-method table, the role-distribution bars, the selective-answering AUROCs) and a per-case view that colours each chunk by its true role and shows which chunk every method blamed. It needs no GPU or model: `pip install streamlit && streamlit run app/app.py` opens it on a bundled sample, and pointing the sidebar at a run's `outputs/` shows real results. See [app/README.md](app/README.md) for deployment.

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
python scripts/run_oracle.py
```

Run the attribution methods under test and record their predicted culprits (needs a GPU):

```
python scripts/run_methods.py
```

Grade the methods against the ground-truth roles, and test whether attribution helps a system abstain (CPU-only):

```
python scripts/run_scoring.py --wrong-only
python scripts/run_abstention.py
```

## Release

`run_pipeline.py` runs every stage end to end on a GPU and writes the four JSONL files:

```
python scripts/run_pipeline.py --limit 200 --k 6 --load-in-4bit
```

`build_release.py` packages a finished run as a HuggingFace dataset — the `chunks` and `cases` configurations, a generated data card, and a `manifest.json` recording the model, seed, source split, and package versions:

```
python scripts/build_release.py --out release
```

See [docs/stage8.md](docs/stage8.md) for the dataset schema and reproduction notes, and [paper/outline.md](paper/outline.md) for the write-up.

## Analysis

Add 95% bootstrap confidence intervals to the scorer, with an optional error-bar figure of the headline rate:

```
python scripts/run_scoring.py --wrong-only --bootstrap 2000 --ci-figure outputs/ci.png
```

Run the pipeline under a second model and compare where the two models' errors land:

```
python scripts/run_agreement.py --a runs/qwen/roles.jsonl --b runs/llama/roles.jsonl
```

Slice the scores by near-miss type and culprit position, probe for two-chunk coalition effects, and sample labelled passages for human validation:

```
python scripts/run_breakdowns.py
python scripts/run_interactions.py --wrong-only --load-in-4bit
python scripts/make_review_sheet.py --n 30   # fill human_role, then score_review.py --sheet outputs/review.csv
```

See [docs/analysis.md](docs/analysis.md) for what each number means.

## Tests

```
pytest             # fast unit tests
pytest -m slow     # also runs backend tests, which fetch a tiny model
```

## Notes

Generation is greedy (temperature 0) and seeded, so a fixed model and precision give reproducible log-probabilities. Scenario construction is likewise seeded and model-free, so a fixed corpus and seed give an identical benchmark. The default backend uses Hugging Face `transformers`; the backend interface is small enough that a vLLM or hosted-API backend can be substituted without touching the rest of the pipeline.
