"""Build a blind human-validation sheet, pooling contested passages across model runs.

Reads the ingested run cells under paper/results_data/<dataset>/<condition>/<model>/ and writes
a blank sheet (no oracle label, no qid) plus a hidden answer key. Hand the same blank sheet to
every reviewer; score the filled copies with score_validation.py.
"""
import argparse
from pathlib import Path

from lineup.data.serialization import read_roles, read_scenarios
from lineup.review import KEY_FIELDS, SCENARIO_FIELDS, sample_scenario_rows, write_csv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"

# Topics an Indian student is more likely to have context for. The datasets are Wikipedia-based
# and skew Western, so this only re-orders what exists -- it floats familiar cases to the front.
FAMILIAR = [
    "india", "indian", "hindi", "bollywood", "cricket", "mumbai", "bombay", "delhi", "kolkata",
    "calcutta", "chennai", "madras", "bengal", "tamil", "telugu", "punjab", "kerala", "gujarat",
    "maharashtra", "rajasthan", "hyderabad", "bangalore", "bengaluru", "pune", "gandhi", "nehru",
    "tagore", "mughal", "himalaya", "ganges", "ganga", "rupee", "ipl", "tendulkar", "bharat",
    "pakistan", "nepal", "bangladesh", "sri lanka",
    "olympic", "fifa", "world cup", "nasa", "einstein", "newton", "physics", "chemistry",
    "biology", "mathematics", "isro", "nobel prize",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki"])
    parser.add_argument("--conditions", nargs="+", default=["baseline", "hardtraps"])
    parser.add_argument("--models", nargs="+", default=["qwen", "phi", "mistral"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-prefer", action="store_true", help="do not bias toward familiar topics")
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "validation")
    args = parser.parse_args()

    sources = []
    for dataset in args.datasets:
        for condition in args.conditions:
            for model in args.models:
                cell = DATA / dataset / condition / model
                roles_path, scen_path = cell / "roles.jsonl", cell / "scenarios.jsonl"
                if not (roles_path.exists() and scen_path.exists()):
                    continue
                scenarios = read_scenarios(scen_path)
                cases = [case for case in read_roles(roles_path) if not case.original_correct]
                sources.append((f"{dataset}/{condition}/{model}", scenarios, cases))

    if not sources:
        raise SystemExit("no run cells found — ingest some zips first")
    print(f"pooling {len(sources)} run cells")

    prefer = None if args.no_prefer else FAMILIAR
    blank, key = sample_scenario_rows(sources, n=args.n, seed=args.seed, prefer=prefer)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "review_blank.csv", blank, SCENARIO_FIELDS)
    write_csv(args.out / "review_key.csv", key, KEY_FIELDS)
    print(f"wrote {len(blank)} rows -> {args.out / 'review_blank.csv'}")
    print(f"hidden key            -> {args.out / 'review_key.csv'}")


if __name__ == "__main__":
    main()
