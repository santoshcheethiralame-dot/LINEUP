import argparse
from pathlib import Path

from lineup.config import OUTPUT_DIR
from lineup.data.serialization import read_roles, read_scenarios
from lineup.review import sample_review_rows, write_review_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample labelled passages into a CSV for human review.")
    parser.add_argument("--scenarios", type=Path, default=OUTPUT_DIR / "scenarios.jsonl")
    parser.add_argument("--roles", type=Path, default=OUTPUT_DIR / "roles.jsonl")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "review.csv")
    args = parser.parse_args()

    scenarios = read_scenarios(args.scenarios)
    cases = [case for case in read_roles(args.roles) if not case.original_correct]
    rows = sample_review_rows(scenarios, cases, n=args.n, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_review_sheet(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    print("fill the human_role column with culprit / misleading / silent / inert, then run score_review.py")


if __name__ == "__main__":
    main()
