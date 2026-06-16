import argparse
from collections import Counter
from pathlib import Path

from lineup.backends import TransformersModel
from lineup.config import DEFAULT_MODEL, OUTPUT_DIR, set_seed
from lineup.correctness import LLMJudge
from lineup.data.hotpotqa import load_examples
from lineup.data.misleading import substitution_check
from lineup.data.scenario import ScenarioBuilder
from lineup.data.serialization import write_roles
from lineup.data.substitution import build_answer_pool
from lineup.generation import generate_and_judge
from lineup.oracle import leave_one_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--wrong-only", action="store_true", help="label only the organically wrong cases")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "roles.jsonl")
    args = parser.parse_args()

    set_seed(args.seed)
    examples = list(load_examples(args.split, limit=args.limit))
    pool = build_answer_pool(examples)
    builder = ScenarioBuilder(answer_pool=pool, k=args.k, seed=args.seed)

    model = TransformersModel(args.model, load_in_4bit=args.load_in_4bit)
    judge = None if args.no_judge else LLMJudge(model)

    cases = []
    role_counts: Counter = Counter()
    misleading_chunk_roles: Counter = Counter()
    for example in examples:
        if substitution_check(example) is not None:
            continue
        scenario = builder.build(example)
        if scenario is None:
            continue
        original = generate_and_judge(model, scenario, llm_judge=judge)
        if args.wrong_only and original.is_correct:
            continue
        case = leave_one_out(model, scenario, original, llm_judge=judge)
        cases.append(case)
        for chunk_role in case.chunk_roles:
            role_counts[chunk_role.role] += 1
            if chunk_role.provenance == "misleading" and not case.original_correct:
                misleading_chunk_roles[chunk_role.role] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_roles(args.out, cases)
    print(f"labeled {len(cases)} cases -> {args.out}")
    print("chunk roles:", dict(role_counts))
    print("on wrong cases, the constructed misleading chunk lands as:", dict(misleading_chunk_roles))


if __name__ == "__main__":
    main()
