import argparse
from pathlib import Path

from lineup.config import DEFAULT_MODEL, DEFAULT_SEED, OUTPUT_DIR, set_seed
from lineup.correctness import LLMJudge
from lineup.data.hotpotqa import load_examples
from lineup.data.misleading import substitution_check
from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import CaseRoles
from lineup.data.serialization import (
    write_generations,
    write_predictions,
    write_roles,
    write_scenarios,
)
from lineup.data.substitution import build_answer_pool
from lineup.downstream import evaluate_abstention
from lineup.generation import generate_and_judge
from lineup.methods import ContextCite, LexicalSimilarity, LLMJudgeCulprit, run_method
from lineup.oracle import leave_one_out
from lineup.scoring import score_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark pipeline end to end (needs a GPU).")
    parser.add_argument("--limit", type=int, default=200, help="number of source questions to draw from")
    parser.add_argument("--k", type=int, default=6, help="passages per case")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--n-ablations", type=int, default=32)
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit weights for a 16 GB GPU")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    set_seed(args.seed)
    from lineup.backends import TransformersModel

    model = TransformersModel(
        args.model, max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit
    )

    examples = list(load_examples(args.split, limit=args.limit))
    pool = build_answer_pool(examples)
    builder = ScenarioBuilder(answer_pool=pool, k=args.k, seed=args.seed)
    judge = LLMJudge(model)

    scenarios, generations = [], []
    for example in examples:
        if substitution_check(example):
            continue
        scenario = builder.build(example)
        if scenario is None:
            continue
        scenarios.append(scenario)
        generations.append(generate_and_judge(model, scenario, llm_judge=judge))

    role_cases = []
    for scenario, generation in zip(scenarios, generations):
        if generation.is_correct:
            role_cases.append(
                CaseRoles(
                    scenario.qid,
                    scenario.question,
                    scenario.gold_answer,
                    generation.model_answer,
                    True,
                    [],
                )
            )
        else:
            role_cases.append(leave_one_out(model, scenario, generation))

    methods = [ContextCite(n_ablations=args.n_ablations, seed=args.seed), LexicalSimilarity(), LLMJudgeCulprit()]
    predictions = []
    for scenario, generation in zip(scenarios, generations):
        for method in methods:
            predictions.append(run_method(method, model, scenario, generation.model_answer))

    args.out.mkdir(parents=True, exist_ok=True)
    write_scenarios(args.out / "scenarios.jsonl", scenarios)
    write_generations(args.out / "generations.jsonl", generations)
    write_roles(args.out / "roles.jsonl", role_cases)
    write_predictions(args.out / "predictions.jsonl", predictions)

    wrong = [case for case in role_cases if not case.original_correct]
    print(f"cases: {len(scenarios)}  wrong: {len(wrong)}  predictions: {len(predictions)}")
    for report in score_predictions(wrong, predictions):
        print(f"{report.method:18s} misleading-as-culprit={report.misleading_as_culprit_rate:.2f}")
    for report in evaluate_abstention(generations, predictions, role_cases):
        auroc = f"{report.auroc:.2f}" if report.auroc is not None else "n/a"
        print(f"{report.signal:22s} AUROC={auroc}")
    for name in ("scenarios", "generations", "roles", "predictions"):
        print(f"wrote {args.out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
