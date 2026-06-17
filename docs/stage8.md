# Stage 8 — Release and Write-up

Stage 8 turns the working pipeline into something another researcher can pick up: a packaged dataset with a data card, a one-command harness, a reproducibility record, and a paper organised around the project's three claims. Nothing here changes the science of Stages 0–7; it makes that science portable.

## The released dataset

The benchmark's product is the per-passage role label produced by the Stage 4 oracle. The release exposes it in two configurations, both written as JSON Lines and described in the data card.

- **`chunks`** — one row per passage, the unit a method is scored against. Each row carries the case it belongs to (`qid`, `question`, `gold_answer`, `model_answer`, `original_correct`), the passage itself (`position`, `chunk_id`, `title`, `text`, `provenance`), and the ground-truth label (`role`, `is_causal`, `is_salient`, `delta_logprob`). Passages from cases the model answered correctly carry a null `role`, because leave-one-out is only run where there is a wrong answer to explain.
- **`cases`** — one row per case, with question-level metadata: the construction parameters (`k`, `intended_wrong_answer`, `substitution_type`), the presented order (`chunk_order`), and a `has_culprit` flag.

Grouping `chunks` by `qid` and ordering by `position` reconstructs the exact context the model saw, so a user can present the same cases to their own system and compare their predicted culprit against the recorded role.

`lineup.release` builds these records as plain dictionaries from the saved `scenarios` and `roles`; it imports only the schema, so it stays pure Python and is covered by `tests/test_release.py` without a model or a GPU.

## The data card

`lineup.release.data_card` renders a HuggingFace dataset card from a run's statistics. It opens with the YAML front-matter the Hub expects — licence, task category, language, tags, and the two `configs` pointing at the JSONL files — and then documents the 2×2 roles, the field schema of each configuration, how to use the benchmark, the run statistics, the reproduction steps, the limitations, the licence, and a citation. Because the statistics block is filled from the actual run, the card never drifts from the data it ships with.

## The reproducibility record

Every release writes a `manifest.json` next to the data. It records the `lineup` version, the model, the seed, the source dataset and split, the wall-clock creation time, the current git commit, the versions of the scientifically load-bearing packages (`transformers`, `datasets`, `torch`, `numpy`, `scikit-learn`), and the dataset statistics. The manifest is what lets a reader establish whether a second run was produced under the same conditions, which matters because the labels depend on one model's behaviour at one precision.

## The harness

Two entry points cover the two things a user wants to do.

- `scripts/run_pipeline.py` runs the whole pipeline end to end on a GPU — load the questions, build the cases, generate, label with leave-one-out, run the methods, score, and evaluate abstention — writing the four JSONL files the rest of the project consumes. It is the command-line equivalent of the Colab notebook, with `--limit`, `--k`, `--model`, `--seed`, and `--load-in-4bit` flags.
- `scripts/build_release.py` packages a finished run into the release directory: the two JSONL configurations, the data card, and the manifest.

## Reproducibility

Scenario construction is deterministic given the seed and the source corpus: `tests/test_release.py` builds the same cases twice from one seed and asserts the presented order and the planted wrong answer are identical. Generation is greedy (temperature 0), so a fixed model and precision give reproducible answers and log-probabilities. The one rule a reproducer must follow is to run every model stage on a single machine and a single model revision: greedy decoding is deterministic per machine, but log-probabilities — which the oracle and ContextCite both depend on — drift across GPUs and numeric precisions.

## The write-up

`paper/outline.md` carries the paper's structure, built on the three pillars the project was designed to support:

1. **The framing** — chunk roles as a 2×2 of counterfactual effect against apparent salience, and the misleading passage as the case existing tools conflate.
2. **The negative result** — salience-based attribution blames the misleading passage at a high rate and loses the culprit-versus-misleading comparison, measured by the Stage 6 metrics.
3. **The downstream cost** — the Stage 7 selective-answering study, where the oracle's fixable-error signal separates correct from wrong answers far better than any deployable attribution signal does.

The outline maps each claim to the table, figure, or script that supports it, and lists what remains before submission (scale, a second model, and a cross-dataset check).

## Limitations

The release packages whatever run it is given; a credible public dataset needs a larger run than the few-dozen-case demos, a second model so the labels are not tied to one model's quirks, and ideally a second multi-hop source. The tooling is ready for that scale; the scientific work of producing and validating the larger run is what the paper depends on.
