import argparse
from pathlib import Path

from lineup.config import OUTPUT_DIR
from lineup.data.serialization import read_predictions, read_roles
from lineup.scoring import score_predictions

_ROLES = ("culprit", "misleading", "silent", "inert")


def _markdown_table(reports) -> str:
    lines = [
        "| method | n | top-1 culprit acc | misleading-as-culprit | culprit > misleading |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        winrate = (
            f"{report.culprit_over_misleading_winrate:.2f}"
            if report.culprit_over_misleading_winrate is not None
            else "n/a"
        )
        lines.append(
            f"| {report.method} | {report.n_cases} | {report.top1_culprit_accuracy:.2f} | "
            f"{report.misleading_as_culprit_rate:.2f} | {winrate} |"
        )
    return "\n".join(lines)


def _render_figure(reports, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [report.method for report in reports]
    bottoms = [0.0] * len(reports)
    figure, axis = plt.subplots(figsize=(7, 4))
    for role in _ROLES:
        values = [report.predicted_role_rate.get(role, 0.0) for report in reports]
        axis.bar(methods, values, bottom=bottoms, label=role)
        bottoms = [base + value for base, value in zip(bottoms, values)]
    axis.set_ylabel("share of predicted culprits")
    axis.set_title("Where each method's predicted culprit actually lands")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, default=OUTPUT_DIR / "roles.jsonl")
    parser.add_argument("--predictions", type=Path, default=OUTPUT_DIR / "predictions.jsonl")
    parser.add_argument("--wrong-only", action="store_true", help="score only the organically wrong cases")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "results.md")
    parser.add_argument("--figure", type=Path, default=None, help="optional PNG of the role distribution")
    args = parser.parse_args()

    cases = read_roles(args.roles)
    if args.wrong_only:
        cases = [case for case in cases if not case.original_correct]
    predictions = read_predictions(args.predictions)
    reports = score_predictions(cases, predictions)

    table = _markdown_table(reports)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    if args.figure is not None:
        _render_figure(reports, args.figure)


if __name__ == "__main__":
    main()
