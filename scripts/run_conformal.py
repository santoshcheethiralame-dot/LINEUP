"""Conformal attribution sets: calibrate a set size that provably covers the culprit at 1 - alpha,
validate the coverage on a held-out split, and show the set must grow under redundancy.
"""
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lineup.conformal import build_ranks, calibrate_tau, coverage_and_size, top1_coverage
from lineup.data.serialization import read_predictions, read_roles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
FIG = ROOT / "paper" / "figures"
DATASETS = ["hotpotqa", "2wiki"]
CONDITIONS = ["baseline", "hardtraps"]
MODELS = ["qwen", "phi", "mistral"]


def load_items():
    items = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for model in MODELS:
                cell = DATA / dataset / condition / model
                if not (cell / "roles.jsonl").exists():
                    continue
                roles = read_roles(cell / "roles.jsonl")
                wrong = [case for case in roles if not case.original_correct]
                preds = read_predictions(cell / "predictions.jsonl")
                items += build_ranks(wrong, preds, source=f"{dataset}/{condition}/{model}", condition=condition)
    return items


def split(items):
    cal, test = [], []
    for item in items:
        digest = int(hashlib.sha256(item.key.encode()).hexdigest(), 16)
        (cal if digest % 2 == 0 else test).append(item)
    return cal, test


def main():
    items = load_items()
    cal, test = split(items)
    alpha = 0.1
    tau = calibrate_tau(cal, alpha)
    coverage, size = coverage_and_size(test, tau)
    base = top1_coverage(test)

    print(f"well-posed (single-culprit) cases: {len(items)}  (cal {len(cal)} / test {len(test)})")
    print(f"target coverage 1 - alpha = {1 - alpha:.2f}")
    print(f"calibrated set size tau    = {tau}")
    print(f"test coverage @ tau        = {coverage:.2f}   (avg set size {size:.1f})")
    print(f"top-1 pick coverage        = {base:.2f}   (set size 1)")

    print("\nset size needed per condition (same 90% guarantee):")
    for condition in CONDITIONS:
        cal_c = [i for i in cal if i.condition == condition]
        test_c = [i for i in test if i.condition == condition]
        tau_c = calibrate_tau(cal_c, alpha)
        cov_c, size_c = coverage_and_size(test_c, tau_c)
        print(f"  {condition:9s} tau={tau_c}  test coverage {cov_c:.2f}  (avg size {size_c:.1f})")

    # coverage vs set size as alpha sweeps: the conformal trade-off, with top-1 marked
    sizes, covers = [], []
    for a in [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]:
        t = calibrate_tau(cal, a)
        c, s = coverage_and_size(test, t)
        sizes.append(s)
        covers.append(c)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sizes, covers, marker="o", color="#3b6fb6", label="conformal set")
    ax.scatter([1], [base], color="#e08a1e", zorder=5, label="top-1 pick")
    ax.axhline(0.9, ls="--", color="grey", lw=1)
    ax.set_xlabel("average set size")
    ax.set_ylabel("culprit coverage")
    ax.set_title("Conformal attribution sets buy a coverage guarantee a single pick can't")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig6_conformal.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {FIG / 'fig6_conformal.png'}")


if __name__ == "__main__":
    main()
