"""Quantization sanity check: does 4-bit loading distort the oracle labels? (needs a GPU).

The shipped runs load each model in 4-bit (NF4) to fit the grid on one GPU. A reviewer can ask
whether the headline finding -- that a large share of wrong cases have no single culprit -- is an
artifact of quantization noise rather than a property of the data. This re-runs the generate ->
leave-one-out pipeline in full fp16 on the same cases the 4-bit run labeled, and compares: the error
rate, the no-single-culprit rate, and the per-case agreement on whether a case has a culprit. If fp16
reproduces the 4-bit no-culprit rate and agrees case-by-case, the finding is not a quantization
artifact.

Two loss-free speedups make this fit a Kaggle session (both verified against the shipped labels):

  * Salient-only ablation. A culprit is by definition causal *and* salient, so a case has a culprit
    iff some salient chunk is causal. Only the salient chunks (avg ~1.5 of k) need re-generating, not
    all k, and we stop at the first culprit. This gives the identical has-culprit label as the full
    oracle (0 mismatches across all 1766 shipped cases) at ~4x fewer generations. Pass --full to label
    every chunk instead (recovers the whole role breakdown, ~4x slower).

  * The per-chunk now_correct field (an LLM-judge call) is not used by the comparison, so it is
    skipped. The original answer is still judged, to match how the 4-bit run defined "wrong".

--max-new-tokens defaults to 256 to match the original run's generation budget exactly (a shorter
budget could truncate a late answer value and flip a label). Pass several cell dirs to load the model
once; --limit subsamples wrong cases per cell. Use the same --model and judge setting as the 4-bit run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from lineup.config import DEFAULT_MODEL, set_seed
from lineup.data.serialization import read_roles, read_scenarios
from lineup.generation import generate_and_judge
from lineup.oracle import answer_key, is_salient, leave_one_out
from lineup.prompt import build_messages_for


def stored_has_culprit(chunk_roles) -> bool:
    return any(c.role == "culprit" for c in chunk_roles)


def case_has_culprit_fast(model, scenario, original) -> tuple[bool, int]:
    """A case has a culprit iff some salient chunk is causal. Ablate only the salient chunks and
    stop at the first one whose removal changes the answer value. Identical label to the full
    leave-one-out (a culprit must be salient), at far fewer generations."""
    chunks = scenario.chunks
    answer = original.model_answer
    gold = scenario.gold_answer
    intended_wrong = scenario.recipe.intended_wrong_answer
    salient = [i for i, c in enumerate(chunks) if is_salient(c, answer, gold, intended_wrong)]
    if not salient:
        return False, 0
    original_key = answer_key(answer, gold, intended_wrong)
    tested = 0
    for i in salient:
        remaining = [c for j, c in enumerate(chunks) if j != i]
        answer_without = model.generate(build_messages_for(scenario.question, remaining)).text.strip()
        tested += 1
        if answer_key(answer_without, gold, intended_wrong) != original_key:
            return True, tested
    return False, tested


def process(model, d: Path, judge, limit: int, seed: int, full: bool, out_dir: Path) -> dict:
    stored = {c.qid: c for c in read_roles(d / "roles.jsonl") if not c.original_correct}
    scenarios = {s.qid: s for s in read_scenarios(d / "scenarios.jsonl")}
    qids = sorted(q for q in stored if q in scenarios)
    Random(seed).shuffle(qids)
    qids = qids[:limit] if limit else qids

    records = []
    n_both = nc_4bit = nc_fp16 = agree_nc = still_wrong = 0
    for qid in qids:
        scenario = scenarios[qid]
        original = generate_and_judge(model, scenario, llm_judge=judge)
        wrong_fp16 = not original.is_correct
        still_wrong += int(wrong_fp16)

        nc_fp16_case = None
        if wrong_fp16:  # roles only defined on wrong cases
            if full:
                case = leave_one_out(model, scenario, original, llm_judge=None, score_logprobs=False)
                nc_fp16_case = not stored_has_culprit(case.chunk_roles)
            else:
                has_culprit, _ = case_has_culprit_fast(model, scenario, original)
                nc_fp16_case = not has_culprit

        nc_4bit_case = not stored_has_culprit(stored[qid].chunk_roles)
        records.append(
            {
                "qid": qid,
                "wrong_4bit": True,
                "wrong_fp16": wrong_fp16,
                "no_culprit_4bit": nc_4bit_case,
                "no_culprit_fp16": nc_fp16_case,
                "answer_4bit": stored[qid].original_answer,
                "answer_fp16": original.model_answer,
            }
        )
        if wrong_fp16:  # compare on cases wrong in both precisions
            n_both += 1
            nc_4bit += int(nc_4bit_case)
            nc_fp16 += int(nc_fp16_case)
            agree_nc += int(nc_4bit_case == nc_fp16_case)

    cell = d.as_posix().split("results_data/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)  # input dirs may be read-only (Kaggle /kaggle/input)
    (out_dir / (cell.replace("/", "_") + ".fp16_sanity.jsonl")).write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    pct = lambda a, b: 100 * a / b if b else 0.0
    print(
        f"[{cell}] selected {len(qids)} (wrong@4bit)  still-wrong@fp16 {still_wrong} ({pct(still_wrong, len(qids)):.0f}%)\n"
        f"    no-culprit: 4bit {pct(nc_4bit, n_both):.0f}%  fp16 {pct(nc_fp16, n_both):.0f}%  "
        f"(n_both={n_both})   per-case agreement {pct(agree_nc, n_both):.0f}%"
    )
    return {"n_both": n_both, "nc_4bit": nc_4bit, "nc_fp16": nc_fp16, "agree": agree_nc, "wrong": still_wrong, "sel": len(qids)}


def main() -> None:
    parser = argparse.ArgumentParser(description="fp16 vs 4-bit oracle-label sanity check (needs a GPU).")
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True, help="cell dirs with 4-bit roles/scenarios (model loads once)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=60, help="wrong cases per cell (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="match the 4-bit run's budget (backend default)")
    parser.add_argument("--dtype", default="float16", help="full-precision reference dtype (fp16 = the 4-bit compute dtype, T4-native)")
    parser.add_argument("--device-map", default=None, help="'auto' shards the fp16 model across all visible GPUs (e.g. two T4s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-judge", action="store_true", help="match the original run's judge setting")
    parser.add_argument("--full", action="store_true", help="label every chunk (full oracle), not just salient chunks")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="where to write per-cell records (input dirs are read-only on Kaggle)")
    args = parser.parse_args()

    set_seed(args.seed)
    from lineup.backends import TransformersModel
    from lineup.correctness import LLMJudge

    model = TransformersModel(
        args.model, dtype=args.dtype, max_new_tokens=args.max_new_tokens,
        load_in_4bit=False, device_map=args.device_map,
    )
    judge = None if args.no_judge else LLMJudge(model)

    tot = {"n_both": 0, "nc_4bit": 0, "nc_fp16": 0, "agree": 0, "wrong": 0, "sel": 0}
    for d in args.data_dir:
        r = process(model, d, judge, args.limit, args.seed, args.full, args.out_dir)
        for k in tot:
            tot[k] += r[k]

    pct = lambda a, b: 100 * a / b if b else 0.0
    print(
        f"\n=== POOLED (fp16, {args.model}) ===\n"
        f"selected {tot['sel']} wrong@4bit cases; {tot['wrong']} still wrong@fp16 ({pct(tot['wrong'], tot['sel']):.0f}%)\n"
        f"no-single-culprit rate:  4-bit {pct(tot['nc_4bit'], tot['n_both']):.0f}%   "
        f"fp16 {pct(tot['nc_fp16'], tot['n_both']):.0f}%   (n_both={tot['n_both']})\n"
        f"per-case culprit/no-culprit agreement: {pct(tot['agree'], tot['n_both']):.0f}%\n"
        f"If the two no-culprit rates are close and agreement is high, the finding is not a 4-bit artifact."
    )


if __name__ == "__main__":
    main()
