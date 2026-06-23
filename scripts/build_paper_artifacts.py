"""Turn the 12-cell matrix into the paper's figures + a CI table.

Reads paper/results_data/<dataset>/<condition>/<model>/ and writes:
    paper/figures/fig1_no_culprit.png      ill-posedness across the matrix
    paper/figures/fig2_set_recovery.png    recall@1 vs recall@k (hard-traps)
    paper/figures/fig3_methods.png         top-1 accuracy by method
    paper/figures/fig4_agreement.png       cross-model role kappa
    paper/results_ci.md                     no-culprit rate with 95% CIs
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lineup.agreement import compare_models
from lineup.data.serialization import read_predictions, read_roles
from lineup.scoring import score_predictions
from lineup.setvalued import attribution_recovery

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
FIG = ROOT / "paper" / "figures"

DATASETS = ["hotpotqa", "2wiki"]
CONDITIONS = ["baseline", "hardtraps"]
MODELS = ["qwen", "phi", "mistral"]
DS_LABEL = {"hotpotqa": "HotpotQA", "2wiki": "2Wiki"}
MODEL_LABEL = {"qwen": "Qwen-7B", "phi": "Phi-3.5", "mistral": "Mistral-7B"}
MODEL_COLOR = {"qwen": "#3b6fb6", "phi": "#e08a1e", "mistral": "#4a9d5b"}
GROUPS = [(d, c) for d in DATASETS for c in CONDITIONS]
METHODS = ["contextcite", "single_chunk", "llm_judge", "lexical_similarity"]
METHOD_LABEL = ["ContextCite", "SingleChunk", "LLM-judge", "Lexical"]


def load(dataset, condition, model):
    cell = DATA / dataset / condition / model
    if not (cell / "roles.jsonl").exists():
        return None
    roles = read_roles(cell / "roles.jsonl")
    wrong = [case for case in roles if not case.original_correct]
    preds = read_predictions(cell / "predictions.jsonl")
    return {
        "roles": roles,
        "wrong": wrong,
        "scores": {r.method: r for r in score_predictions(wrong, preds)},
        "recovery": {r.method: r for r in attribution_recovery(wrong, preds)},
    }


def no_culprit_rate(wrong):
    nc = sum(1 for case in wrong if not any(role.role == "culprit" for role in case.chunk_roles))
    return nc / len(wrong), len(wrong)


def binom_halfwidth(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


def fig_no_culprit(cells):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.25
    for offset, model in enumerate(MODELS):
        xs, ys, errs = [], [], []
        for group_index, (dataset, condition) in enumerate(GROUPS):
            cell = cells[(dataset, condition, model)]
            rate, n = no_culprit_rate(cell["wrong"])
            xs.append(group_index + (offset - 1) * width)
            ys.append(rate * 100)
            errs.append(binom_halfwidth(rate, n) * 100)
        ax.bar(xs, ys, width, yerr=errs, capsize=3, label=MODEL_LABEL[model], color=MODEL_COLOR[model])
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([f"{DS_LABEL[d]}\n{c}" for d, c in GROUPS])
    ax.set_ylabel("errors with no single culprit (%)")
    ax.set_title("Roughly a third of organic errors have no single causal culprit")
    ax.set_ylim(0, 62)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_no_culprit.png", dpi=300)
    plt.close(fig)


def fig_set_recovery(cells):
    selection = [(d, m) for d in DATASETS for m in MODELS]
    labels = [f"{MODEL_LABEL[m]}\n{DS_LABEL[d]}" for d, m in selection]
    r1 = [cells[(d, "hardtraps", m)]["recovery"]["contextcite"].recall_at_1 for d, m in selection]
    rk = [cells[(d, "hardtraps", m)]["recovery"]["contextcite"].recall_at_k for d, m in selection]
    x = range(len(selection))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar([i - width / 2 for i in x], r1, width, label="single pick (recall@1)", color="#c2c6cc")
    ax.bar([i + width / 2 for i in x], rk, width, label="effect set (recall@k)", color="#3b6fb6")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of the culprit set recovered")
    ax.set_title("Under redundancy, one pick can't cover the culprit set — a set can")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_set_recovery.png", dpi=300)
    plt.close(fig)


def fig_methods(cells):
    data = []
    for method in METHODS:
        values = []
        for key, cell in cells.items():
            report = cell["scores"].get(method)
            if report and report.top1_culprit_accuracy is not None:
                values.append(report.top1_culprit_accuracy)
        data.append(values)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(METHOD_LABEL) + 1))
    ax.set_xticklabels(METHOD_LABEL)
    ax.set_ylabel("top-1 culprit accuracy (12 settings)")
    ax.set_title("Effect-based attribution localizes the culprit; lexical overlap does not")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_methods.png", dpi=300)
    plt.close(fig)


def fig_agreement(cells):
    pairs = [("qwen", "phi"), ("qwen", "mistral"), ("phi", "mistral")]
    pair_label = ["Qwen–Phi", "Qwen–Mistral", "Phi–Mistral"]
    colors = ["#3b6fb6", "#e08a1e", "#4a9d5b"]
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for offset, (a, b) in enumerate(pairs):
        xs = [gi + (offset - 1) * width for gi in range(len(GROUPS))]
        ys = [compare_models(cells[(d, c, a)]["roles"], cells[(d, c, b)]["roles"]).role_kappa for d, c in GROUPS]
        ax.bar(xs, ys, width, label=pair_label[offset], color=colors[offset])
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([f"{DS_LABEL[d]}\n{c}" for d, c in GROUPS])
    ax.set_ylabel("per-passage role agreement (Cohen's κ)")
    ax.set_title("Different model families disagree on which passage is to blame")
    ax.set_ylim(0, 0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_agreement.png", dpi=300)
    plt.close(fig)


def ci_table(cells):
    lines = [
        "# No-culprit rate with 95% confidence intervals",
        "",
        "| dataset | condition | model | n_wrong | no-culprit % [95% CI] |",
        "|---|---|---|--:|--:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for model in MODELS:
                cell = cells[(dataset, condition, model)]
                rate, n = no_culprit_rate(cell["wrong"])
                half = binom_halfwidth(rate, n)
                lo, hi = max(0, rate - half), min(1, rate + half)
                lines.append(
                    f"| {dataset} | {condition} | {MODEL_LABEL[model]} | {n} | "
                    f"{100 * rate:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] |"
                )
    (ROOT / "paper" / "results_ci.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cells = {
        (d, c, m): load(d, c, m)
        for d in DATASETS for c in CONDITIONS for m in MODELS
    }
    missing = [key for key, value in cells.items() if value is None]
    if missing:
        raise SystemExit(f"missing cells: {missing}")
    FIG.mkdir(parents=True, exist_ok=True)
    fig_no_culprit(cells)
    fig_set_recovery(cells)
    fig_methods(cells)
    fig_agreement(cells)
    ci_table(cells)
    print(f"wrote 4 figures to {FIG} and {ROOT / 'paper' / 'results_ci.md'}")


if __name__ == "__main__":
    main()
