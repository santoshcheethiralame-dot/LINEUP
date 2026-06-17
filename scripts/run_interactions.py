import argparse
from pathlib import Path

from lineup.config import DEFAULT_MODEL, OUTPUT_DIR, set_seed
from lineup.data.serialization import read_generations, read_scenarios
from lineup.interaction import pairwise_interactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe for pairwise coalition effects among chunks (needs a GPU).")
    parser.add_argument("--scenarios", type=Path, default=OUTPUT_DIR / "scenarios.jsonl")
    parser.add_argument("--generations", type=Path, default=OUTPUT_DIR / "generations.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--wrong-only", action="store_true", help="probe only the wrong cases")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "interactions.md")
    args = parser.parse_args()

    set_seed()
    from lineup.backends import TransformersModel

    model = TransformersModel(args.model, max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit)

    scenarios = {scenario.qid: scenario for scenario in read_scenarios(args.scenarios)}
    generations = read_generations(args.generations)
    if args.wrong_only:
        generations = [generation for generation in generations if not generation.is_correct]

    synergy_cases = 0
    probed = 0
    lines = ["| qid | pairs | synergy pairs |", "| --- | ---: | ---: |"]
    for generation in generations:
        scenario = scenarios.get(generation.qid)
        if scenario is None:
            continue
        probed += 1
        result = pairwise_interactions(model, scenario, generation)
        synergy_cases += 1 if result.has_synergy else 0
        lines.append(f"| {result.qid} | {len(result.pairs)} | {sum(1 for pair in result.pairs if pair.synergy)} |")

    header = f"cases with a coalition effect: {synergy_cases}/{probed}"
    table = "\n".join(lines)
    print(header)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"{header}\n\n{table}\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
