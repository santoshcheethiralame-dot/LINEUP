"""Generate the LINEUP leaderboard: every method scored on the headline metrics, pooled across the
full 12-cell matrix. Regenerate with `python scripts/build_leaderboard.py`."""
from __future__ import annotations

from dataclasses import replace

from lineup.config import PROJECT_ROOT
from lineup.data.serialization import read_predictions, read_roles
from lineup.evaluate import evaluate, leaderboard_markdown, no_culprit_rate

DATA = PROJECT_ROOT / "paper" / "results_data"
OUT = PROJECT_ROOT / "paper" / "leaderboard.md"
MODELS = ["qwen", "phi", "mistral"]
DATASETS = ["hotpotqa", "2wiki"]
CONDS = ["baseline", "hardtraps"]


def main():
    cases, preds = [], []
    for ds in DATASETS:
        for cond in CONDS:
            for model in MODELS:
                d = DATA / ds / cond / model
                if (d / "roles.jsonl").exists() and (d / "predictions.jsonl").exists():
                    # qids repeat across cells, so namespace by cell to avoid a corrupting join.
                    cell = f"{ds}/{cond}/{model}"
                    cases += [replace(c, qid=f"{cell}::{c.qid}") for c in read_roles(d / "roles.jsonl") if not c.original_correct]
                    preds += [replace(p, qid=f"{cell}::{p.qid}") for p in read_predictions(d / "predictions.jsonl")]

    scores = evaluate(cases, preds)
    ncr = no_culprit_rate(cases)
    md = leaderboard_markdown(scores, ncr)

    OUT.write_text(
        "# LINEUP leaderboard\n\n"
        "Headline metrics for each attribution method, pooled across all 12 cells "
        "(3 model families x 2 datasets x 2 conditions), wrong cases only. "
        "recall@1 is the single-pick ceiling; recall@k is the set. Regenerate with "
        "`python scripts/build_leaderboard.py`; add your own method via `lineup.evaluate.evaluate`.\n\n"
        + md
        + "\n",
        encoding="utf-8",
    )
    print(md)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
