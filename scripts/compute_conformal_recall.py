"""De-oracled recall: recover the planted set with a CONFORMAL-calibrated set size, not |R| (local).

recall@k (A4/A8) presumes you know m = |R| per case. This calibrates a single set size tau on a
calibration split (so the top-tau set covers the whole planted set in >= 1-alpha of cases), then
reports, on a disjoint test split, how much of the planted set the top-tau pick recovers -- a recall
number that never consults the oracle's per-case size. Addresses reviewer item #3.
"""
from __future__ import annotations
import json, glob, math, os
from random import Random

ROOT = r"C:/Users/carbo/projects/lineup/paper/results_data"
PLANT = {"misleading", "decoy"}
ALPHA = 0.1

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]

def gather(cond):
    """Per case: sorted ranks (1-based) of planted chunks, for contextcite."""
    out = []
    for rp in glob.glob(ROOT + f"/*/{cond}/*/roles.jsonl"):
        roles = {r["qid"]: r for r in load(rp) if not r["original_correct"] and r["chunk_roles"]}
        pp = os.path.join(os.path.dirname(rp), "predictions.jsonl")
        preds = {r["qid"]: r for r in load(pp) if r["method"] == "contextcite"}
        for qid, role in roles.items():
            R = {c["chunk_id"] for c in role["chunk_roles"] if c["provenance"] in PLANT}
            pr = preds.get(qid)
            if not R or not pr:
                continue
            ranked = [c["chunk_id"] for c in sorted(pr["chunk_scores"], key=lambda x: -x["score"])]
            ranks = sorted(ranked.index(cid) + 1 for cid in R if cid in ranked)
            if len(ranks) == len(R):
                out.append((R, ranks, ranked))
    return out

def calibrate_tau(cal):
    # smallest tau s.t. top-tau covers ALL planted in >= 1-alpha of cal cases (split conformal order stat)
    worst = sorted(max(ranks) for _, ranks, _ in cal)
    n = len(worst)
    idx = math.ceil((n + 1) * (1 - ALPHA)) - 1
    return worst[min(idx, n - 1)]

print(f"{'cond':<10}{'n':>5}{'tau':>5}{'cov@tau':>9}{'recall@tau':>11}{'recall@1':>10}{'recall@k(oracle)':>18}")
for cond in ["baseline", "hardtraps"]:
    data = gather(cond)
    Random(0).shuffle(data)
    half = len(data) // 2
    cal, test = data[:half], data[half:]
    tau = calibrate_tau(cal)
    cov = sum(max(ranks) <= tau for _, ranks, _ in test) / len(test)
    rtau = sum(len({r for r in ranks if r <= tau}) / len(R) for R, ranks, _ in test) / len(test)
    r1 = sum(len({r for r in ranks if r <= 1}) / len(R) for R, ranks, _ in test) / len(test)
    rk = sum(len({r for r in ranks if r <= len(R)}) / len(R) for R, ranks, _ in test) / len(test)
    print(f"{cond:<10}{len(test):>5}{tau:>5}{cov:>9.2f}{rtau:>11.2f}{r1:>10.2f}{rk:>18.2f}")
print("\ntau = conformal-calibrated set size (covers whole planted set in >=90% of calibration cases).")
print("recall@tau de-oracles recall@k: it uses a fixed calibrated size, never the per-case oracle |R|.")
