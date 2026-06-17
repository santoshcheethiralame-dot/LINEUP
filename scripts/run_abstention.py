import argparse
from pathlib import Path

from lineup.config import OUTPUT_DIR
from lineup.data.serialization import read_generations, read_predictions, read_roles
from lineup.downstream import abstention_curves, evaluate_abstention


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


if __name__ == "__main__":
    main()
