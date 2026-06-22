"""Calibrated selective attribution: abstain when no single culprit is evident, fall back to a set.

Pools the wrong cases across every run cell, picks the best confidence signal on a train split,
and reports on a held-out test split: how much abstaining raises the accuracy of the answers the
method *does* give, and how well the effect-set covers the culprits on the cases it abstains on.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lineup.data.serialization import read_predictions, read_roles
from lineup.selective import best_signal, build_cases, risk_coverage, set_recall_on_abstained, signal_auroc, split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
FIG = ROOT / "paper" / "figures"
DATASETS = ["hotpotqa", "2wiki"]
CONDITIONS = ["baseline", "hardtraps"]
MODELS = ["qwen", "phi", "mistral"]
COVERAGES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def load_all():
    pooled = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for model in MODELS:
                cell = DATA / dataset / condition / model
                if not (cell / "roles.jsonl").exists():
                    continue
                roles = read_roles(cell / "roles.jsonl")
                wrong = [case for case in roles if not case.original_correct]
                preds = read_predictions(cell / "predictions.jsonl")
                pooled += build_cases(wrong, preds, source=f"{dataset}/{condition}/{model}")
    return pooled


def main():
    pooled = load_all()
    train, test = split(pooled)
    signal = best_signal(train)
    auroc_test = signal_auroc(test, signal)
    curve = risk_coverage(test, signal, COVERAGES)
    base = curve[1.0]
    rec1_half, reck_half = set_recall_on_abstained(test, signal, 0.5)

    print(f"pooled wrong cases: {len(pooled)}  (train {len(train)} / test {len(test)})")
    print(f"chosen confidence signal (max train AUROC): {signal}")
    print(f"test AUROC (signal vs top-1 is correct):    {auroc_test:.2f}")
    print("\nselective accuracy of the answers it gives, by coverage (test):")
    for coverage in COVERAGES:
        print(f"  coverage {int(coverage * 100):3d}%  ->  accuracy {curve[coverage]:.2f}")
    print(f"\nbase (always answer) accuracy : {base:.2f}")
    print(f"accuracy at 50% coverage      : {curve[0.5]:.2f}   (+{curve[0.5] - base:.2f})")
    print(f"on the abstained half, effect-set recall: single pick {rec1_half:.2f} -> set {reck_half:.2f}")

    coverages = list(COVERAGES)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([c * 100 for c in coverages], [curve[c] for c in coverages], marker="o", color="#3b6fb6", label="calibrated abstention")
    ax.axhline(base, ls="--", color="grey", lw=1, label="answer everything (no abstention)")
    ax.set_xlabel("coverage (% of cases answered)")
    ax.set_ylabel("accuracy of the answers given")
    ax.set_title("Abstaining on low-confidence cases raises attribution accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig5_selective.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {FIG / 'fig5_selective.png'}")


if __name__ == "__main__":
    main()
