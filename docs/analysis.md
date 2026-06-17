# Analysis — confidence intervals and cross-model agreement

Two analyses strengthen the headline result beyond a single point estimate on a single model. Both are pure Python and run on CPU.

## Confidence intervals

The per-method metrics are computed over a finite set of wrong cases, so each number carries sampling noise. `lineup.scoring.bootstrap_intervals` quantifies it by resampling cases with replacement and recomputing the metrics, then reporting a percentile interval. The case is the resampling unit, which matches how the benchmark would be redrawn: a different sample of questions.

The point estimate and the bootstrap share one per-case computation (`_case_metric` and `_aggregate`), so the interval cannot drift from the number it brackets. `tests/test_bootstrap.py` checks that the interval contains the point estimate and that a fixed seed is reproducible.

```
python scripts/run_scoring.py --wrong-only --bootstrap 2000 --ci-figure outputs/ci.png
```

This prints the usual table, then a second table with 95% intervals, and optionally a bar chart of the misleading-as-culprit rate with error bars. On a small run the intervals are wide — that width is the honest statement that the run needs more cases before the rates can be read closely.

## Cross-model agreement

A benchmark whose labels come from one model's behaviour invites the question of whether the result is a quirk of that model. Running the pipeline under a second model and comparing the two label sets answers it. `lineup.agreement.compare_models` takes two `roles.jsonl` files over the same cases and reports:

- the wrong-answer rate of each model,
- the number of cases both models answered wrongly,
- among those, how often the two models' culprit sets are identical (`same_culprit_rate`) and their average overlap (`culprit_jaccard`),
- the per-passage role agreement and Cohen's kappa over the cases both got wrong.

High same-culprit agreement is a strong result: the misleading passage fools both models, so the failure is a property of the setting rather than of one model. Low agreement is equally publishable — it bounds how far a single-model label set generalises.

```
python scripts/run_pipeline.py --model Qwen/Qwen2.5-7B-Instruct --out runs/qwen --load-in-4bit
python scripts/run_pipeline.py --model meta-llama/Llama-3.1-8B-Instruct --out runs/llama --load-in-4bit
python scripts/run_agreement.py --a runs/qwen/roles.jsonl --b runs/llama/roles.jsonl --label-a qwen --label-b llama
```

Both model stages must run on one machine and precision, for the same reason the rest of the pipeline does: log-probabilities, and therefore the leave-one-out labels, are only comparable within a fixed numeric setting.
