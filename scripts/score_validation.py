"""Score filled blind-review sheets against the hidden key.

    python scripts/score_validation.py --key paper/validation/review_key.csv \
        --sheets review_santosh.csv review_nivas.csv review_rushi.csv
"""
import argparse
from pathlib import Path

from lineup.review import read_review_sheet, score_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--sheets", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    key = read_review_sheet(args.key)
    sheets = {sheet.stem: read_review_sheet(sheet) for sheet in args.sheets}
    result = score_scenario(key, sheets)

    print("human vs oracle (picked the same culprit):")
    for name, stat in result["per_reviewer"].items():
        value = f"{stat['agreement']:.2f}" if stat["agreement"] is not None else "n/a"
        print(f"  {name:20s} {value}  (n={stat['n']})")
    maj, inter, none = result["majority_vs_oracle"], result["inter_rater"], result["none_recall"]
    print(f"\nmajority vote vs oracle : {maj:.2f}  (n={result['majority_n']})" if maj is not None else "\nmajority: n/a")
    print(f"no-culprit agreement    : {none:.2f}  (n={result['none_n']})" if none is not None else "no-culprit: n/a")
    print(f"inter-rater agreement   : {inter:.2f}" if inter is not None else "inter-rater: n/a")


if __name__ == "__main__":
    main()
