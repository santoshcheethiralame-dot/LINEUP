import argparse
from pathlib import Path

from lineup.config import OUTPUT_DIR
from lineup.data.serialization import read_generations, read_predictions, read_roles
from lineup.downstream import abstention_curves, bootstrap_abstention_auroc, evaluate_abstention


def _cell(value) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _markdown_table(reports) -> str:
    lines = [
        "| signal | n | correct | AUROC | AURC | selective acc @50% |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        lines.append(
            f"| {report.signal} | {report.n} | {report.n_correct} | "
            f"{_cell(report.auroc)} | {_cell(report.aurc)} | {_cell(report.selective_accuracy_at_50)} |"
        )
    return "\n".join(lines)


def _fmt_ci(point, interval) -> str:
    if point is None:
        return "n/a"
    low, high = interval
    if low is None:
        return f"{point:.3f}"
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def _ci_table(reports, intervals) -> str:
    lines = ["| signal | AUROC (95% CI) |", "| --- | --- |"]
    for report in reports:
        lines.append(f"| {report.signal} | {_fmt_ci(report.auroc, intervals.get(report.signal, (None, None)))} |")
    return "\n".join(lines)


def _render_figure(generations, predictions, roles, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    for name, (coverages, risks) in abstention_curves(generations, predictions, roles).items():
        if coverages:
            axis.plot(coverages, risks, label=name)
    axis.set_xlabel("coverage (fraction answered)")
    axis.set_ylabel("risk (error rate of the answered set)")
    axis.set_title("Selective QA risk–coverage by confidence signal")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, default=OUTPUT_DIR / "generations.jsonl")
    parser.add_argument("--predictions", type=Path, default=OUTPUT_DIR / "predictions.jsonl")
    parser.add_argument("--roles", type=Path, default=OUTPUT_DIR / "roles.jsonl")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "abstention.md")
    parser.add_argument("--figure", type=Path, default=None, help="optional PNG of the risk-coverage curves")
    parser.add_argument("--bootstrap", type=int, default=0, help="resamples for AUROC confidence intervals (0 = off)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the bootstrap resampling")
    args = parser.parse_args()

    generations = read_generations(args.generations)
    predictions = read_predictions(args.predictions)
    roles = read_roles(args.roles) if args.roles.exists() else None
    reports = evaluate_abstention(generations, predictions, roles)

    table = _markdown_table(reports)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    if args.figure is not None:
        _render_figure(generations, predictions, roles, args.figure)

    if args.bootstrap:
        intervals = bootstrap_abstention_auroc(
            generations, predictions, roles, n_boot=args.bootstrap, seed=args.seed
        )
        ci_table = _ci_table(reports, intervals)
        print("\n" + ci_table)
        ci_path = args.out.with_name(args.out.stem + "_ci.md")
        ci_path.write_text(ci_table + "\n", encoding="utf-8")
        print(f"wrote {ci_path}")


if __name__ == "__main__":
    main()
