import argparse
from pathlib import Path

from lineup.agreement import compare_models
from lineup.data.serialization import read_generations, read_predictions, read_roles
from lineup.downstream import abstention_curves
from lineup.scoring import bootstrap_intervals, score_predictions

ROLE_ORDER = ["culprit", "misleading", "silent", "inert"]


def render_dashboard(run_dirs, *, seed: int = 0, n_boot: int = 2000, out: str = "dashboard.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for name, directory in run_dirs.items():
        base = Path(directory)
        roles = read_roles(base / "roles.jsonl")
        wrong = [case for case in roles if not case.original_correct]
        preds = read_predictions(base / "predictions.jsonl")
        data[name] = {
            "roles": roles,
            "wrong": wrong,
            "preds": preds,
            "gens": read_generations(base / "generations.jsonl"),
            "reports": {r.method: r for r in score_predictions(wrong, preds)},
            "cis": bootstrap_intervals(wrong, preds, n_boot=n_boot, seed=seed),
        }

    methods = sorted(next(iter(data.values()))["reports"])
    names = list(run_dirs)
    primary = names[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        reps, cis = data[name]["reports"], data[name]["cis"]
        vals = [reps[m].misleading_as_culprit_rate for m in methods]
        lo, hi = [], []
        for j, m in enumerate(methods):
            low, high = cis[m]["misleading_as_culprit_rate"]
            lo.append(vals[j] - (low if low is not None else vals[j]))
            hi.append((high if high is not None else vals[j]) - vals[j])
        ax.bar([x + i * width for x in range(len(methods))], vals, width, yerr=[lo, hi], capsize=4, label=name)
    ax.set_xticks([x + width * (len(names) - 1) / 2 for x in range(len(methods))])
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("misleading-as-culprit rate")
    ax.set_title("How often each method blames the near-miss")
    ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[0, 1]
    reps = data[primary]["reports"]
    matrix = [[reps[m].predicted_role_rate.get(r, 0.0) for r in ROLE_ORDER] for m in methods]
    image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ROLE_ORDER)))
    ax.set_xticklabels(ROLE_ORDER)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    for r in range(len(methods)):
        for c in range(len(ROLE_ORDER)):
            ax.text(c, r, f"{matrix[r][c]:.2f}", ha="center", va="center",
                    color="white" if matrix[r][c] < 0.5 else "black", fontsize=9)
    ax.set_title(f"Where {primary}'s predicted culprit truly lands")
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    for signal, (cov, risk) in abstention_curves(data[primary]["gens"], data[primary]["preds"], data[primary]["roles"]).items():
        if cov:
            ax.plot(cov, risk, label=signal, linewidth=1.5)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (error of the answered set)")
    ax.set_title("Selective answering -- does attribution help abstain")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")

    def fmt(value):
        return f"{value:.2f}" if value is not None else "n/a"

    if len(names) >= 2:
        rep = compare_models(data[names[0]]["roles"], data[names[1]]["roles"])
        lines = [
            f"cross-model agreement: {names[0]} vs {names[1]}", "",
            f"cases wrong in both:     {rep.n_both_wrong} / {rep.n_common}",
            f"same culprit set:        {fmt(rep.same_culprit_rate)}",
            f"culprit-set Jaccard:     {fmt(rep.culprit_jaccard)}",
            f"per-passage role kappa:  {fmt(rep.role_kappa)}",
        ]
        ax.text(0.0, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=12)
    else:
        ax.text(0.0, 0.95, "add a second run to fill this panel", va="top", fontsize=12)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")

    for name in names:
        print(f"\n== {name} ==")
        for m in methods:
            report = data[name]["reports"][m]
            low, high = data[name]["cis"][m]["misleading_as_culprit_rate"]
            band = f"  [{low:.2f}, {high:.2f}]" if low is not None else ""
            top1 = f"{report.top1_culprit_accuracy:.2f}" if report.top1_culprit_accuracy is not None else "n/a"
            print(f"  {m:18s} top1={top1}  misleading-as-culprit={report.misleading_as_culprit_rate:.2f}{band}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the results dashboard from one or more runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="run directories; the folder name is the label")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out", default="dashboard.png")
    args = parser.parse_args()

    render_dashboard({Path(d).name: d for d in args.runs}, seed=args.seed, n_boot=args.n_boot, out=args.out)


if __name__ == "__main__":
    main()
