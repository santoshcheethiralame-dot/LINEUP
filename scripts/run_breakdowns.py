import argparse
from pathlib import Path

from lineup.breakdowns import by_culprit_position, by_substitution_type
from lineup.config import OUTPUT_DIR
from lineup.data.serialization import read_predictions, read_roles, read_scenarios


def _table(grouped) -> str:
    lines = [
        "| group | method | n | top-1 | misleading-as-culprit |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for label, reports in grouped.items():
        for report in reports:
            top1 = f"{report.top1_culprit_accuracy:.2f}" if report.top1_culprit_accuracy is not None else "n/a"
            lines.append(
                f"| {label} | {report.method} | {report.n_cases} | {top1} | {report.misleading_as_culprit_rate:.2f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Break the scores down by substitution type and culprit position.")
    parser.add_argument("--scenarios", type=Path, default=OUTPUT_DIR / "scenarios.jsonl")
    parser.add_argument("--roles", type=Path, default=OUTPUT_DIR / "roles.jsonl")
    parser.add_argument("--predictions", type=Path, default=OUTPUT_DIR / "predictions.jsonl")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "breakdowns.md")
    args = parser.parse_args()

    scenarios = read_scenarios(args.scenarios)
    cases = [case for case in read_roles(args.roles) if not case.original_correct]
    predictions = read_predictions(args.predictions)

    text = (
        "## By substitution type\n\n"
        + _table(by_substitution_type(scenarios, cases, predictions))
        + "\n\n## By culprit position\n\n"
        + _table(by_culprit_position(scenarios, cases, predictions))
    )
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
