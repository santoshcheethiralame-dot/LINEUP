# Evaluating your attribution method on LINEUP

LINEUP grades a per-passage attribution method against **non-circular causal ground truth**: for each
wrong RAG answer, the oracle labels every passage with its role (culprit / misleading / silent / inert).
To benchmark your own method, score each passage for each wrong case and call `evaluate` — the same
scoring used for the paper, so your numbers are directly comparable to the [leaderboard](../paper/leaderboard.md).

## The three steps

```python
from lineup.data.serialization import read_roles, read_scenarios
from lineup.evaluate import evaluate, predictions_from_scores, leaderboard_markdown, no_culprit_rate

# 1. Load the benchmark: the ground-truth roles, and the passages your method reads.
cases      = read_roles("roles.jsonl")           # one CaseRoles per wrong case (the labels)
scenarios  = {s.qid: s for s in read_scenarios("scenarios.jsonl")}   # the question + 6 passages

# 2. Run YOUR method on each case: return a score per passage (higher = more responsible).
def my_method(question, passages):               # passages: list of (chunk_id, text)
    ...                                          # your attribution logic
    return {chunk_id: score for chunk_id, _ in passages}

scores_by_qid = {}
for case in cases:
    s = scenarios[case.qid]
    passages = [(c.chunk_id, c.text) for c in s.chunks]
    scores_by_qid[case.qid] = my_method(case.question, passages)

# 3. Score it.
preds  = predictions_from_scores("my_method", scores_by_qid)
result = evaluate(cases, preds)
print(leaderboard_markdown(result, no_culprit=no_culprit_rate(cases)))
```

If your method ranks the passages instead of scoring them, just emit descending scores (e.g. `6, 5, 4, …`).

## What the metrics mean

| Metric | Question it answers |
|---|---|
| **top-1 culprit acc** | When a single culprit exists, does your top pick find it? |
| **recall@1** | What fraction of the *responsible set* does your single pick recover? (capped at 1/\|R\| when blame is shared) |
| **recall@k** | What fraction does a set of the right size recover? (the set ceiling) |
| **reliability AUROC** | Does your score margin flag your own wrong picks? (≈0.5 = no) |
| **single-culprit AUROC** | Does your score margin flag cases that have no single culprit? |

The **no-culprit rate** (~37% pooled) is a property of the benchmark, not your method: that fraction of
wrong cases has no passage that is both causal and salient, so a single top-1 answer is ill-posed there.
The headline takeaway LINEUP is built to surface: a large `recall@k − recall@1` gap, and a low
single-culprit AUROC, mean your method should return a **set** (with abstention / conformal calibration),
not a single passage.

## Using the released HuggingFace dataset

```python
from datasets import load_dataset
chunks = load_dataset("<org>/lineup", "chunks")   # one row per passage, grouped by case_id
cases  = load_dataset("<org>/lineup", "cases")    # one row per case
```
Group `chunks` by `case_id`, run your method per case, and score as above. See the dataset card for the
full field schema.

## Reproduce the published leaderboard

```
python scripts/build_leaderboard.py
```
Regenerates [`paper/leaderboard.md`](../paper/leaderboard.md) for the four reference methods (ContextCite,
SingleChunk, LexicalSimilarity, LLM judge) across all 12 cells.
