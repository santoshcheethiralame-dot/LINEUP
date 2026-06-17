import argparse
from pathlib import Path

from lineup.agreement import compare_models
from lineup.data.serialization import read_roles


def _cell(value) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two models' role labels over the same cases.")
    parser.add_argument("--a", type=Path, required=True, help="roles.jsonl from the first model")
    parser.add_argument("--b", type=Path, required=True, help="roles.jsonl from the second model")
    parser.add_argument("--label-a", default="model A")
    parser.add_argument("--label-b", default="model B")
    args = parser.parse_args()

    report = compare_models(read_roles(args.a), read_roles(args.b))

    print(f"common cases: {report.n_common}  both wrong: {report.n_both_wrong}")
    print(f"wrong rate  {args.label_a}: {report.wrong_rate_a:.2f}  {args.label_b}: {report.wrong_rate_b:.2f}")
    print(f"same culprit set (both wrong): {_cell(report.same_culprit_rate)}")
    print(f"culprit-set Jaccard:           {_cell(report.culprit_jaccard)}")
    print(f"per-passage role agreement:    {_cell(report.role_agreement)}")
    print(f"role Cohen's kappa:            {_cell(report.role_kappa)}")


if __name__ == "__main__":
    main()
