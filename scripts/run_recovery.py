import argparse
from pathlib import Path

from lineup.config import OUTPUT_DIR
from lineup.data.serialization import read_predictions, read_roles
from lineup.setvalued import attribution_recovery


def _cell(value) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _markdown_table(reports) -> str:
    lines = [
        "| method | n | mean|R| | recall@1 | recall@k | innocent-gold | reliability AUROC | single-culprit AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        lines.append(
            f"| {report.method} | {report.n} | {report.mean_responsible:.2f} | "
            f"{_cell(report.recall_at_1)} | {_cell(report.recall_at_k)} | "
            f"{_cell(report.innocent_gold_rate)} | {_cell(report.reliability_auroc)} | "
            f"{_cell(report.single_culprit_auroc)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, default=OUTPUT_DIR / "roles.jsonl")
    parser.add_argument("--predictions", type=Path, default=OUTPUT_DIR / "predictions.jsonl")
    parser.add_argument("--wrong-only", action="store_true", help="score only the organically wrong cases")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "recovery.md")
    args = parser.parse_args()

    cases = read_roles(args.roles)
    if args.wrong_only:
        cases = [case for case in cases if not case.original_correct]
    predictions = read_predictions(args.predictions)
    reports = attribution_recovery(cases, predictions)

    table = _markdown_table(reports)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
