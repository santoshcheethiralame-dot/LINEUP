import argparse
from pathlib import Path

from lineup.review import read_review_sheet, review_agreement


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a filled-in human review sheet against the oracle labels.")
    parser.add_argument("--sheet", type=Path, required=True)
    args = parser.parse_args()

    result = review_agreement(read_review_sheet(args.sheet))
    if result["agreement"] is None:
        print("no rows have a human_role yet — fill the column and rerun")
    else:
        print(f"human/oracle agreement: {result['agreement']:.2f} over {result['n']} reviewed passages")


if __name__ == "__main__":
    main()
