"""Free frontier-scale slice: run the oracle on a large API model (Groq 70B, Gemini) to check that
the ill-posedness holds beyond 7B open models. Generation-only — no GPU, no teacher-forced
logprobs, so ContextCite/SingleChunk are skipped and we use the oracle + lexical + LLM-judge.

    export LINEUP_API_KEY=...          # a free Groq or Google AI Studio key
    pip install openai
    python scripts/run_frontier.py --provider groq --model llama-3.3-70b-versatile --smoke
    python scripts/run_frontier.py --provider groq --model llama-3.3-70b-versatile --limit 150
"""
import argparse
from pathlib import Path

from lineup.backends.api_backend import PROVIDERS, APIModel
from lineup.backends.base import Message
from lineup.config import set_seed
from lineup.correctness import LLMJudge
from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import CaseRoles
from lineup.data.serialization import (
    read_generations,
    read_predictions,
    read_roles,
    write_generations,
    write_predictions,
    write_roles,
    write_scenarios,
)
from lineup.data.sources import load_examples
from lineup.data.substitution import build_answer_pool
from lineup.generation import generate_and_judge
from lineup.methods import LexicalSimilarity, LLMJudgeCulprit, run_method
from lineup.oracle import leave_one_out

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "frontier_data"


def _progress(label, i, total, every=25):
    if (i + 1) % every == 0 or (i + 1) == total:
        print(f"  {label} {i + 1}/{total}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=list(PROVIDERS), default="groq")
    parser.add_argument("--model", required=True, help="e.g. llama-3.3-70b-versatile / gemini-2.0-flash")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-interval", type=float, default=0.0, help="seconds between calls (use ~4 for Gemini free tier's 15 req/min)")
    parser.add_argument("--max-tokens", type=int, default=512, help="output budget; reasoning models need room to think before answering")
    parser.add_argument("--smoke", action="store_true", help="one generation to verify the key/endpoint")
    args = parser.parse_args()

    model = APIModel(args.model, base_url=PROVIDERS[args.provider], max_new_tokens=args.max_tokens, min_interval=args.min_interval)
    if args.smoke:
        print("smoke reply:", model.generate([Message("user", "Reply with the single word: ok")]).text)
        return

    set_seed(args.seed)
    examples = list(load_examples(args.dataset, args.split, limit=args.limit))
    builder = ScenarioBuilder(answer_pool=build_answer_pool(examples), k=6, seed=args.seed, n_decoys=0)
    scenarios = [s for s in (builder.build(e) for e in examples) if s is not None]
    print(f"built {len(scenarios)} cases from {len(examples)} questions")

    judge = LLMJudge(model)
    tag = f"{args.provider}-{args.model}".replace("/", "_")
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    write_scenarios(out / "scenarios.jsonl", scenarios)

    # Every phase resumes from its checkpoint (scenario order is deterministic by seed), so a
    # rate-limit cap mid-run never loses work — just re-run the same command and it continues.
    gen_path = out / "generations.jsonl"
    generations = list(read_generations(gen_path)) if gen_path.exists() else []
    if generations:
        print(f"  resuming: {len(generations)}/{len(scenarios)} generations already done")
    for i in range(len(generations), len(scenarios)):
        generations.append(generate_and_judge(model, scenarios[i], llm_judge=judge))
        if (i + 1) % 10 == 0 or i + 1 == len(scenarios):
            write_generations(gen_path, generations)
        _progress("generate", i, len(scenarios))
    write_generations(gen_path, generations)
    n_wrong = sum(not g.is_correct for g in generations)
    print(f"{n_wrong} wrong of {len(scenarios)}")

    roles_path = out / "roles.jsonl"
    role_cases = list(read_roles(roles_path)) if roles_path.exists() else []
    if role_cases:
        print(f"  resuming: {len(role_cases)}/{len(scenarios)} oracle cases already done")
    for i in range(len(role_cases), len(scenarios)):
        scenario, generation = scenarios[i], generations[i]
        if generation.is_correct:
            role_cases.append(CaseRoles(scenario.qid, scenario.question, scenario.gold_answer, generation.model_answer, True, []))
        else:
            role_cases.append(leave_one_out(model, scenario, generation, llm_judge=judge, score_logprobs=False))
        if (i + 1) % 5 == 0 or i + 1 == len(scenarios):
            write_roles(roles_path, role_cases)
        _progress("oracle", i, len(scenarios))
    write_roles(roles_path, role_cases)

    methods = [LexicalSimilarity(), LLMJudgeCulprit()]
    preds_path = out / "predictions.jsonl"
    predictions = list(read_predictions(preds_path)) if preds_path.exists() else []
    done_qids = {p.qid for p in predictions}
    remaining = [(s, g) for s, g in zip(scenarios, generations) if not g.is_correct and s.qid not in done_qids]
    if predictions:
        print(f"  resuming: {len(done_qids)} method cases already done")
    for i, (scenario, generation) in enumerate(remaining):
        for method in methods:
            try:
                predictions.append(run_method(method, model, scenario, generation.model_answer))
            except NotImplementedError:
                pass        # a method that needs logprobs cannot run on an API model
        if (i + 1) % 10 == 0 or i + 1 == len(remaining):
            write_predictions(preds_path, predictions)
        _progress("methods", i, len(remaining))
    write_predictions(preds_path, predictions)

    wrong = [c for c in role_cases if not c.original_correct]
    no_culprit = sum(1 for c in wrong if not any(r.role == "culprit" for r in c.chunk_roles))
    print(f"\nFRONTIER {tag}: errors {n_wrong}/{len(scenarios)}; "
          f"no single culprit {no_culprit}/{len(wrong)} = {no_culprit / len(wrong):.0%}")
    print("(open 7B baselines were 28-49% no-culprit — does the frontier model match?)")
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
