import argparse
from pathlib import Path

from lineup.config import OUTPUT_DIR
from lineup.data.hotpotqa import load_examples
from lineup.data.misleading import substitution_check
from lineup.data.scenario import ScenarioBuilder
from lineup.data.serialization import write_scenarios
from lineup.data.substitution import build_answer_pool

_MARK = {"gold": "G", "misleading": "M", "distractor": "."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "scenarios.jsonl")
    args = parser.parse_args()

    examples = list(load_examples(args.split, limit=args.limit))
    pool = build_answer_pool(examples)
    builder = ScenarioBuilder(answer_pool=pool, k=args.k, seed=args.seed)

    scenarios = []
    skipped: dict[str, int] = {}
    for example in examples:
        reason = substitution_check(example)
        if reason is None:
            scenario = builder.build(example)
            if scenario is not None:
                scenarios.append(scenario)
                continue
            reason = "construction failed"
        skipped[reason] = skipped.get(reason, 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_scenarios(args.out, scenarios)
    print(f"built {len(scenarios)} scenarios from {len(examples)} examples -> {args.out}")
    if skipped:
        print("skipped:", dict(sorted(skipped.items())))

    if scenarios:
        scenario = scenarios[0]
        print(f"\nexample {scenario.qid}")
        print(f"Q: {scenario.question}")
        print(
            f"gold answer: {scenario.gold_answer}   "
            f"intended wrong answer: {scenario.recipe.intended_wrong_answer} "
            f"({scenario.recipe.substitution_type})"
        )
        print("context order:")
        for position, chunk in enumerate(scenario.chunks):
            print(f"  {position:2d} [{_MARK.get(chunk.provenance, '?')}] {chunk.title}")


if __name__ == "__main__":
    main()
