"""Position-controlled ablation: is a chunk's causality its content, or its position? (needs a GPU).

Leave-one-out *deletes* a chunk, which removes its content but also shifts every later chunk one slot
earlier and shortens the context. A position-sensitive model ("lost in the middle") could then flip
its answer for a *positional* reason, and the chunk would be mislabelled causal. This re-runs the
causal test with length-matched *masking* instead of deletion: the chunk stays in its slot [i], but
its title and body are replaced by content-free filler of the same token length, so every other
chunk keeps its position and the context keeps its length. Only the information is removed.

For each wrong case we recompute, on the salient chunks (a culprit must be salient, so only they can
change the has-culprit label -- the same fast path used elsewhere, 0/1766 mismatch offline):
  * delete-causal: answer changes when the chunk is removed (standard oracle)
  * mask-causal:   answer changes when the chunk is masked in place (position-controlled)
and report the no-single-culprit rate and ContextCite culprit-accuracy under each, with per-case
agreement. If masking reproduces deletion, the causal axis is content, not position; if it moves, the
gap is the position confound R1 asked us to quantify.

Runs in 4-bit (fits one T4; the fp16 sanity showed 4-bit does not distort the labels), so no
sharding is needed. Pass several cell dirs to load the model once; --limit subsamples per cell.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from lineup.config import DEFAULT_MODEL, set_seed
from lineup.data.schema import Chunk
from lineup.data.serialization import read_predictions, read_roles, read_scenarios
from lineup.generation import generate_and_judge
from lineup.oracle import answer_key, is_salient
from lineup.prompt import build_messages_for

FILLER_SENTENCE = "This passage has been removed for this evaluation. "


def mask_chunk(chunk: Chunk, tokenizer) -> Chunk:
    """Replace a chunk's title+body with content-free filler of the same token length, in place."""
    n = len(tokenizer.encode(chunk.text, add_special_tokens=False))
    if n <= 0:
        return Chunk(chunk_id=chunk.chunk_id, title="[removed]", text="", provenance=chunk.provenance)
    reps = (FILLER_SENTENCE * (n // len(tokenizer.encode(FILLER_SENTENCE, add_special_tokens=False)) + 2))
    ids = tokenizer.encode(reps, add_special_tokens=False)[:n]
    return Chunk(chunk_id=chunk.chunk_id, title="[removed]", text=tokenizer.decode(ids), provenance=chunk.provenance)


def causal_under(model, scenario, original_key, gold, iw, index, *, mask: bool) -> bool:
    """Does removing (mask=False) or masking (mask=True) chunk `index` change the answer's value?"""
    chunks = scenario.chunks
    if mask:
        kept = [mask_chunk(c, model.tokenizer) if j == index else c for j, c in enumerate(chunks)]
    else:
        kept = [c for j, c in enumerate(chunks) if j != index]
    answer = model.generate(build_messages_for(scenario.question, kept)).text.strip()
    return answer_key(answer, gold, iw) != original_key


def process(model, d: Path, judge, limit: int, seed: int) -> dict:
    stored = {c.qid: c for c in read_roles(d / "roles.jsonl") if not c.original_correct}
    scenarios = {s.qid: s for s in read_scenarios(d / "scenarios.jsonl")}
    cc_top = {p.qid: p.predicted_culprit_id for p in read_predictions(d / "predictions.jsonl") if p.method == "contextcite"}
    qids = sorted(q for q in stored if q in scenarios)
    Random(seed).shuffle(qids)
    qids = qids[:limit] if limit else qids

    n = nc_del = nc_mask = agree = 0
    wp_del = wp_mask = cc_hit_del = cc_hit_mask = 0
    for qid in qids:
        sc = scenarios[qid]
        gold, iw = sc.gold_answer, sc.recipe.intended_wrong_answer
        original = generate_and_judge(model, sc, llm_judge=judge)
        if original.is_correct:
            continue
        key = answer_key(original.model_answer, gold, iw)
        salient = [i for i, c in enumerate(sc.chunks) if is_salient(c, original.model_answer, gold, iw)]
        del_causal = {i: causal_under(model, sc, key, gold, iw, i, mask=False) for i in salient}
        mask_causal = {i: causal_under(model, sc, key, gold, iw, i, mask=True) for i in salient}

        has_del = any(del_causal.values())
        has_mask = any(mask_causal.values())
        n += 1
        nc_del += int(not has_del)
        nc_mask += int(not has_mask)
        agree += int((not has_del) == (not has_mask))

        top = cc_top.get(qid)
        top_i = next((i for i, c in enumerate(sc.chunks) if c.chunk_id == top), None)
        top_culprit_del = top_i in salient and del_causal.get(top_i, False)
        top_culprit_mask = top_i in salient and mask_causal.get(top_i, False)
        if has_del:
            wp_del += 1
            cc_hit_del += int(top_culprit_del)
        if has_mask:
            wp_mask += 1
            cc_hit_mask += int(top_culprit_mask)

    cell = d.as_posix().split("results_data/")[-1]
    pct = lambda a, b: 100 * a / b if b else 0.0
    print(
        f"[{cell}] wrong n={n}\n"
        f"    no-culprit:  delete {pct(nc_del, n):.0f}%   mask {pct(nc_mask, n):.0f}%   agreement {pct(agree, n):.0f}%\n"
        f"    CC culprit-acc: delete {pct(cc_hit_del, wp_del):.0f}% (wp={wp_del})   mask {pct(cc_hit_mask, wp_mask):.0f}% (wp={wp_mask})"
    )
    return {"n": n, "nc_del": nc_del, "nc_mask": nc_mask, "agree": agree,
            "wp_del": wp_del, "wp_mask": wp_mask, "cc_del": cc_hit_del, "cc_mask": cc_hit_mask}


def main() -> None:
    p = argparse.ArgumentParser(description="Position-controlled (length-matched masking) ablation vs deletion (needs a GPU).")
    p.add_argument("--data-dir", type=Path, nargs="+", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=60, help="wrong cases per cell (0 = all)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--fp16", dest="load_in_4bit", action="store_false", help="use fp16 instead of 4-bit (needs 2x T4)")
    p.add_argument("--device-map", default=None, help="'auto' to shard fp16 across GPUs")
    args = p.parse_args()

    set_seed(args.seed)
    from lineup.backends import TransformersModel
    from lineup.correctness import LLMJudge

    model = TransformersModel(
        args.model, max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit, device_map=args.device_map,
    )
    judge = None if args.no_judge else LLMJudge(model)

    tot = {k: 0 for k in ("n", "nc_del", "nc_mask", "agree", "wp_del", "wp_mask", "cc_del", "cc_mask")}
    for d in args.data_dir:
        r = process(model, d, judge, args.limit, args.seed)
        for k in tot:
            tot[k] += r[k]

    pct = lambda a, b: 100 * a / b if b else 0.0
    print(
        f"\n=== POOLED position control ({args.model}) ===\n"
        f"no-single-culprit:  deletion {pct(tot['nc_del'], tot['n']):.0f}%   masking {pct(tot['nc_mask'], tot['n']):.0f}%   "
        f"(n={tot['n']})   per-case agreement {pct(tot['agree'], tot['n']):.0f}%\n"
        f"ContextCite culprit-accuracy:  deletion {pct(tot['cc_del'], tot['wp_del']):.0f}%   masking {pct(tot['cc_mask'], tot['wp_mask']):.0f}%\n"
        f"If masking reproduces deletion, the causal axis is content, not position (R1)."
    )


if __name__ == "__main__":
    main()
