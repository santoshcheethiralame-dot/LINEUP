"""Does the no-single-culprit finding survive stochastic (temperature>0) decoding? (needs a GPU).

The oracle defines roles by whether removing a passage changes the answer under greedy decoding.
Referee B notes deployed RAG usually samples, where "the answer" is a distribution and the
counterfactual is noisy. This re-runs the causal test with sampling: for each case we draw K samples
at temperature T and take the *majority* answer value (mode of answer_key over the K draws) as "the
answer"; a passage is causal if removing it changes the majority value. Only salient chunks can be
culprits, so only they are tested (the same fast path used elsewhere). We compare the no-single-
culprit rate under sampling to the greedy labels on the cases wrong under both.

Sampling is done by calling the HF model directly (num_return_sequences=K, do_sample), so the
backend's greedy path is untouched. Runs in 4-bit on one T4; pass several cell dirs to load once.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from random import Random

import torch

from lineup.config import DEFAULT_MODEL, set_seed
from lineup.correctness import judge_correct
from lineup.data.serialization import read_roles, read_scenarios
from lineup.oracle import answer_key, is_salient
from lineup.prompt import build_messages, build_messages_for


@torch.no_grad()
def majority_key(model, messages, gold, iw, *, k, temperature, max_new_tokens):
    """Sample K answers, return (majority answer_key, one representative answer text)."""
    ids = model._encode_prompt(messages)
    out = model.model.generate(
        ids, do_sample=True, temperature=temperature, top_p=0.9,
        num_return_sequences=k, max_new_tokens=max_new_tokens,
        pad_token_id=model.tokenizer.eos_token_id,
    )
    texts = [model.tokenizer.decode(o[ids.shape[1]:], skip_special_tokens=True).strip() for o in out]
    keys = [answer_key(t, gold, iw) for t in texts]
    key = Counter(keys).most_common(1)[0][0]
    rep = next(t for t, kk in zip(texts, keys) if kk == key)
    return key, rep


def process(model, d: Path, judge, args) -> dict:
    stored = {c.qid: c for c in read_roles(d / "roles.jsonl") if not c.original_correct}
    scenarios = {s.qid: s for s in read_scenarios(d / "scenarios.jsonl")}
    qids = sorted(q for q in stored if q in scenarios)
    Random(args.seed).shuffle(qids)
    qids = qids[: args.limit] if args.limit else qids

    n = nc_greedy = nc_sample = agree = still_wrong = 0
    for qid in qids:
        sc = scenarios[qid]
        gold, iw = sc.gold_answer, sc.recipe.intended_wrong_answer
        orig_key, orig_ans = majority_key(model, build_messages(sc), gold, iw,
                                          k=args.samples, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        correct, _ = judge_correct(sc.question, gold, orig_ans, judge)
        if correct:
            continue
        still_wrong += 1
        salient = [i for i, c in enumerate(sc.chunks) if is_salient(c, orig_ans, gold, iw)]
        culprit = False
        for i in salient:
            keep = [c for j, c in enumerate(sc.chunks) if j != i]
            key, _ = majority_key(model, build_messages_for(sc.question, keep), gold, iw,
                                  k=args.samples, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
            if key != orig_key:
                culprit = True
                break
        nc_s = not culprit
        nc_g = not any(c.role == "culprit" for c in stored[qid].chunk_roles)
        n += 1
        nc_greedy += int(nc_g)
        nc_sample += int(nc_s)
        agree += int(nc_g == nc_s)

    cell = d.as_posix().split("results_data/")[-1]
    pct = lambda a, b: 100 * a / b if b else 0.0
    print(f"[{cell}] wrong@both n={n} (still-wrong@sample {still_wrong}/{len(qids)})\n"
          f"    no-culprit:  greedy {pct(nc_greedy, n):.0f}%   sampling(T={args.temperature},K={args.samples}) {pct(nc_sample, n):.0f}%   agreement {pct(agree, n):.0f}%")
    return {"n": n, "g": nc_greedy, "s": nc_sample, "agree": agree}


def main() -> None:
    p = argparse.ArgumentParser(description="No-single-culprit under temperature>0 sampling (needs a GPU).")
    p.add_argument("--data-dir", type=Path, nargs="+", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--samples", type=int, default=5, help="K draws per prompt; majority answer is 'the answer'")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--fp16", dest="load_in_4bit", action="store_false")
    args = p.parse_args()

    set_seed(args.seed)
    from lineup.backends import TransformersModel
    from lineup.correctness import LLMJudge

    model = TransformersModel(args.model, max_new_tokens=args.max_new_tokens, load_in_4bit=args.load_in_4bit)
    judge = None if args.no_judge else LLMJudge(model)

    tot = {"n": 0, "g": 0, "s": 0, "agree": 0}
    for d in args.data_dir:
        r = process(model, d, judge, args)
        for kk in tot:
            tot[kk] += r[kk]
    pct = lambda a, b: 100 * a / b if b else 0.0
    print(f"\n=== POOLED sampling ({args.model}, T={args.temperature}, K={args.samples}) ===\n"
          f"no-single-culprit:  greedy {pct(tot['g'], tot['n']):.0f}%   sampling {pct(tot['s'], tot['n']):.0f}%   "
          f"(n={tot['n']})   per-case agreement {pct(tot['agree'], tot['n']):.0f}%\n"
          f"If sampling reproduces greedy, the finding is not an artifact of deterministic decoding (Ref B).")


if __name__ == "__main__":
    main()
