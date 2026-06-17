import argparse
from pathlib import Path

from lineup.config import OUTPUT_DIR
from lineup.data.serialization import read_predictions, read_roles
from lineup.scoring import bootstrap_intervals, score_predictions

_ROLES = ("culprit", "misleading", "silent", "inert")


def _cell(value) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _markdown_table(reports) -> str:
    lines = [
        "| method | n | n(culprit) | top-1 | misleading-as-culprit | culprit > misleading | culprit > rest |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        lines.append(
            f"| {report.method} | {report.n_cases} | {report.n_with_culprit} | "
            f"{_cell(report.top1_culprit_accuracy)} | {report.misleading_as_culprit_rate:.2f} | "
            f"{_cell(report.culprit_over_misleading_winrate)} | {_cell(report.culprit_over_rest_winrate)} |"
        )
    return "\n".join(lines)


def _fmt_ci(point, interval) -> str:
    if point is None:
        return "n/a"
    low, high = interval
    if low is None:
        return f"{point:.2f}"
    return f"{point:.2f} [{low:.2f}, {high:.2f}]"


def _ci_table(reports, intervals) -> str:
    lines = [
        "| method | top-1 culprit | misleading-as-culprit | culprit > misleading | culprit > rest |",
        "| --- | --- | --- | --- | --- |",
    ]
    none = (None, None)
    for report in reports:
        ci = intervals.get(report.method, {})
        lines.append(
            f"| {report.method} | "
            f"{_fmt_ci(report.top1_culprit_accuracy, ci.get('top1_culprit_accuracy', none))} | "
            f"{_fmt_ci(report.misleading_as_culprit_rate, ci.get('misleading_as_culprit_rate', none))} | "
            f"{_fmt_ci(report.culprit_over_misleading_winrate, ci.get('culprit_over_misleading_winrate', none))} | "
            f"{_fmt_ci(report.culprit_over_rest_winrate, ci.get('culprit_over_rest_winrate', none))} |"
        )
    return "\n".join(lines)


def _render_ci_figure(reports, intervals, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [report.method for report in reports]
    points = [report.misleading_as_culprit_rate for report in reports]
    lows, highs = [], []
    for report in reports:
        low, high = intervals.get(report.method, {}).get("misleading_as_culprit_rate", (None, None))
        lows.append(report.misleading_as_culprit_rate - low if low is not None else 0.0)
        highs.append(high - report.misleading_as_culprit_rate if high is not None else 0.0)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(methods, points, yerr=[lows, highs], capsize=6)
    axis.set_ylabel("misleading-as-culprit rate")
    axis.set_title("How often each method blames the near-miss (95% bootstrap CI)")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    print(f"wrote {path}")


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
    parser.add_argument("--bootstrap", type=int, default=0, help="resamples for confidence intervals (0 = off)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the bootstrap resampling")
    parser.add_argument("--ci-figure", type=Path, default=None, help="optional PNG of misleading-as-culprit with CIs")
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

    if args.bootstrap:
        intervals = bootstrap_intervals(cases, predictions, n_boot=args.bootstrap, seed=args.seed)
        ci_table = _ci_table(reports, intervals)
        print("\n" + ci_table)
        ci_path = args.out.with_name(args.out.stem + "_ci.md")
        ci_path.write_text(ci_table + "\n", encoding="utf-8")
        print(f"wrote {ci_path}")
        if args.ci_figure is not None:
            _render_ci_figure(reports, intervals, args.ci_figure)


if __name__ == "__main__":
    main()
