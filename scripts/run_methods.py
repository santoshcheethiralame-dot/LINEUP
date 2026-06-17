import argparse
from collections import Counter, defaultdict
from pathlib import Path

from lineup.backends import TransformersModel
from lineup.config import DEFAULT_MODEL, OUTPUT_DIR, set_seed
from lineup.correctness import LLMJudge
from lineup.data.hotpotqa import load_examples
from lineup.data.misleading import substitution_check
from lineup.data.scenario import ScenarioBuilder
from lineup.data.serialization import write_predictions
from lineup.data.substitution import build_answer_pool
from lineup.generation import generate_and_judge
from lineup.methods import ContextCite, LexicalSimilarity, LLMJudgeCulprit, run_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--wrong-only", action="store_true", help="run only on the organically wrong cases")
    parser.add_argument("--n-ablations", type=int, default=32)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "predictions.jsonl")
    args = parser.parse_args()

    set_seed(args.seed)
    examples = list(load_examples(args.split, limit=args.limit))
    pool = build_answer_pool(examples)
    builder = ScenarioBuilder(answer_pool=pool, k=args.k, seed=args.seed)

    model = TransformersModel(args.model, load_in_4bit=args.load_in_4bit)
    judge = None if args.no_judge else LLMJudge(model)
    methods = [
        ContextCite(n_ablations=args.n_ablations, seed=args.seed),
        LexicalSimilarity(),
        LLMJudgeCulprit(),
    ]

    predictions = []
    blamed = defaultdict(Counter)
    for example in examples:
        if substitution_check(example) is not None:
            continue
        scenario = builder.build(example)
        if scenario is None:
            continue
        original = generate_and_judge(model, scenario, llm_judge=judge)
        if args.wrong_only and original.is_correct:
            continue
        provenance = {chunk.chunk_id: chunk.provenance for chunk in scenario.chunks}
        for method in methods:
            prediction = run_method(method, model, scenario, original.model_answer)
            predictions.append(prediction)
            blamed[method.name][provenance.get(prediction.predicted_culprit_id, "?")] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out, predictions)
    print(f"wrote {len(predictions)} predictions -> {args.out}")
    print("provenance of each method's predicted culprit:")
    for method in methods:
        print(f"  {method.name}: {dict(blamed[method.name])}")


if __name__ == "__main__":
    main()
