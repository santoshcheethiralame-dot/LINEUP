"""Leave-two-out proof that the 'no single cause' cases are real redundant coalitions (needs a GPU).

The no-culprit breakdown reports a 'no single cause' bucket: wrong cases where single-chunk
leave-one-out finds no causal passage at all. A single-removal oracle cannot, by construction, tell
whether that is a true redundant coalition -- two passages each sufficient, so removing one leaves the
other to sustain the error -- or just model noise. This runs the pairwise (leave-two-out) test on
exactly those cases and reports how often removing a *pair* flips the answer though neither passage
alone does. That synergy is the thing single leave-one-out is blind to; showing it directly closes
the objection that the redundancy is inferred rather than demonstrated.

Fast without losing value: every chunk in a no-single-cause case is already non-causal under single
leave-one-out (that is the filter), so any pair that flips the answer is a synergy pair -- the
single-removal generations are not recomputed. Pairs are ordered by how close each chunk came to
flipping alone (its single-removal log-prob shift, already in roles.jsonl) and the search stops at the
first synergy pair. Confirming a case has *no* synergy still tests every pair, so nothing is sampled
away. Pass several cell directories so the model loads once for all of them.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from lineup.config import DEFAULT_MODEL, set_seed
from lineup.data.serialization import read_generations, read_scenarios
from lineup.oracle import answer_key
from lineup.prompt import build_messages_for


def no_single_cause(roles_path: Path) -> dict:
    """qid -> {chunk_id: |single-removal logprob shift|} for wrong cases with no causal chunk."""
    out = {}
    for line in roles_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("original_correct", True):
            continue
        if any(c["causal"] for c in r["chunk_roles"]):
            continue
        out[r["qid"]] = {c["chunk_id"]: abs(c.get("delta_logprob", 0.0)) for c in r["chunk_roles"]}
    return out


def process(model, d: Path) -> None:
    targets = no_single_cause(d / "roles.jsonl")
    scenarios = {s.qid: s for s in read_scenarios(d / "scenarios.jsonl")}
    generations = {g.qid: g for g in read_generations(d / "generations.jsonl") if g.qid in targets}

    records = []
    synergy_cases = 0
    for qid, deltas in targets.items():
        scenario = scenarios.get(qid)
        generation = generations.get(qid)
        if scenario is None or generation is None:
            continue
        gold = scenario.gold_answer
        intended_wrong = scenario.recipe.intended_wrong_answer
        original_key = answer_key(generation.model_answer, gold, intended_wrong)
        chunks = scenario.chunks
        pairs = sorted(
            combinations(range(len(chunks)), 2),
            key=lambda ij: deltas.get(chunks[ij[0]].chunk_id, 0.0) + deltas.get(chunks[ij[1]].chunk_id, 0.0),
            reverse=True,
        )
        found, flipped, tested = None, None, 0
        for i, j in pairs:
            tested += 1
            keep = [c for k, c in enumerate(chunks) if k not in (i, j)]
            answer = model.generate(build_messages_for(scenario.question, keep)).text.strip()
            if answer_key(answer, gold, intended_wrong) != original_key:
                found, flipped = (chunks[i].chunk_id, chunks[j].chunk_id), answer
                break
        synergy_cases += int(found is not None)
        records.append(
            {
                "qid": qid,
                "gold_answer": gold,
                "model_answer": generation.model_answer,
                "has_synergy": found is not None,
                "synergy_pair": list(found) if found else None,
                "answer_without_pair": flipped,
                "pairs_tested": tested,
            }
        )

    n = len(records)
    rate = 100 * synergy_cases / n if n else 0.0
    (d / "coalition_proof.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"[{d.as_posix().split('results_data/')[-1]}] no-single-cause {n}  synergy {synergy_cases} ({rate:.0f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast leave-two-out coalition proof on the no-single-cause slice (needs a GPU).")
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True, help="one or more cell dirs (model loads once for all)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    set_seed()
    from lineup.backends import TransformersModel

    model = TransformersModel(args.model, max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit)
    for d in args.data_dir:
        process(model, d)


if __name__ == "__main__":
    main()
