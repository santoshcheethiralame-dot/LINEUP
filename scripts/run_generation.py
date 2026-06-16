import argparse
from pathlib import Path

from lineup.backends import TransformersModel
from lineup.config import DEFAULT_MODEL, OUTPUT_DIR, set_seed
from lineup.correctness import LLMJudge
from lineup.data.hotpotqa import load_examples
from lineup.data.misleading import substitution_check
from lineup.data.scenario import ScenarioBuilder
from lineup.data.serialization import write_generations
from lineup.data.substitution import build_answer_pool
from lineup.generation import generate_and_judge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="exact match only, skip the LLM judge")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "generations.jsonl")
    args = parser.parse_args()

    set_seed(args.seed)
    examples = list(load_examples(args.split, limit=args.limit))
    pool = build_answer_pool(examples)
    builder = ScenarioBuilder(answer_pool=pool, k=args.k, seed=args.seed)

    scenarios = []
    for example in examples:
        if substitution_check(example) is None:
            scenario = builder.build(example)
            if scenario is not None:
                scenarios.append(scenario)

    model = TransformersModel(args.model, load_in_4bit=args.load_in_4bit)
    judge = None if args.no_judge else LLMJudge(model)

    results = [generate_and_judge(model, scenario, llm_judge=judge) for scenario in scenarios]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_generations(args.out, results)

    correct = [r for r in results if r.is_correct]
    wrong = [r for r in results if not r.is_correct]
    fooled = [r for r in wrong if r.matched_intended_wrong]
    print(f"ran {len(results)} cases -> {args.out}")
    print(f"correct {len(correct)}  wrong {len(wrong)}  (of the wrong, {len(fooled)} echoed the planted value)")

    if fooled:
        example = max(fooled, key=lambda r: r.answer_logprob)
        print(f"\nmost confident error: {example.qid}")
        print(f"  Q: {example.question}")
        print(f"  gold: {example.gold_answer}   model: {example.model_answer}")


if __name__ == "__main__":
    main()
