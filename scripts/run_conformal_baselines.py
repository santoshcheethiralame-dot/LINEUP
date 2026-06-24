"""Stress-test the conformal remedy against naive set baselines (local, no Kaggle).

Honest finding: naive set rules (a fixed top-k, a score-threshold set) reach comparable coverage on
this benchmark, so conformal is not *more efficient*. Its contribution is the distribution-free
*guarantee*: only conformal hits a chosen coverage level 1-alpha by construction; the naive rules'
coverage depends on a hand-tuned k or threshold with no recourse and no promise off-distribution.

We also test an adaptive per-case variant (APS, include chunks until their cumulative score mass
passes a calibrated threshold). It bloats here -- because ContextCite's effect scores are diffuse,
capturing the mass needs many chunks -- so the fixed-tau conformal set is the right object. Reported
as a negative result.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from lineup.conformal import build_ranks, calibrate_tau, coverage_and_size, top1_coverage
from lineup.data.serialization import read_predictions, read_roles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
OUT = ROOT / "paper" / "conformal_baselines.md"
DATASETS = ["hotpotqa", "2wiki"]
CONDITIONS = ["baseline", "hardtraps"]
MODELS = ["qwen", "phi", "mistral"]
ALPHA = 0.1


def _cum_mass(scores):
    pos = [max(s, 0.0) for s in scores]
    tot = sum(pos) or 1.0
    cum, c = [], 0.0
    for p in pos:
        c += p / tot
        cum.append(c)
    return cum


def load_cases():
    out = []
    for ds in DATASETS:
        for cond in CONDITIONS:
            for model in MODELS:
                cell = DATA / ds / cond / model
                if not ((cell / "roles.jsonl").exists() and (cell / "predictions.jsonl").exists()):
                    continue
                roles = {c.qid: c for c in read_roles(cell / "roles.jsonl") if not c.original_correct}
                for pr in read_predictions(cell / "predictions.jsonl"):
                    if pr.method != "contextcite" or pr.qid not in roles:
                        continue
                    case = roles[pr.qid]
                    ranked = sorted(pr.chunk_scores, key=lambda s: s.score, reverse=True)
                    ids = [s.chunk_id for s in ranked]
                    culprits = [r.chunk_id for r in case.chunk_roles if r.role == "culprit"]
                    rank = (ids.index(culprits[0]) + 1) if len(culprits) == 1 and culprits[0] in ids else None
                    out.append({
                        "key": f"{ds}/{cond}/{model}/{pr.qid}", "condition": cond, "n": len(ids),
                        "rank": rank, "cum": _cum_mass([s.score for s in ranked]),
                        "top_score": ranked[0].score if ranked else 0.0, "scores": [s.score for s in ranked],
                    })
    return out


def is_cal(key):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0


def threshold_size(case, frac=0.5):
    thr = frac * max(case["top_score"], 0.0)
    return max(1, sum(1 for s in case["scores"] if s >= thr and s > 0))


def aps(cal_wp, test_wp):
    masses = sorted(c["cum"][c["rank"] - 1] for c in cal_wp)
    n = len(masses)
    q = masses[min(math.ceil((n + 1) * (1 - ALPHA)), n) - 1]
    def size(case):
        for j, c in enumerate(case["cum"], start=1):
            if c >= q:
                return j
        return case["n"]
    cov = sum(1 for c in test_wp if c["cum"][c["rank"] - 1] <= q) / len(test_wp)
    sz = sum(size(c) for c in test_wp) / len(test_wp)
    return cov, sz


def main():
    cases = load_cases()
    wp = [c for c in cases if c["rank"] is not None]
    cal_wp = [c for c in wp if is_cal(c["key"])]
    test_wp = [c for c in wp if not is_cal(c["key"])]

    items = []
    for ds in DATASETS:
        for cond in CONDITIONS:
            for model in MODELS:
                cell = DATA / ds / cond / model
                if (cell / "roles.jsonl").exists():
                    wrong = [c for c in read_roles(cell / "roles.jsonl") if not c.original_correct]
                    items += build_ranks(wrong, read_predictions(cell / "predictions.jsonl"),
                                         source=f"{ds}/{cond}/{model}", condition=cond)
    cal_i = [i for i in items if is_cal(i.key)]
    test_i = [i for i in items if not is_cal(i.key)]
    tau = calibrate_tau(cal_i, ALPHA)
    conf_cov, conf_size = coverage_and_size(test_i, tau)

    def cov_fixed(k):
        return sum(1 for c in test_wp if c["rank"] <= k) / len(test_wp)
    rows = [
        ("single pick (top-1)", top1_coverage(test_i), 1.0, "uncalibrated"),
        ("fixed top-2", cov_fixed(2), 2.0, "uncalibrated"),
        ("score-threshold (>=0.5*top)",
         sum(1 for c in test_wp if c["rank"] <= threshold_size(c)) / len(test_wp),
         sum(threshold_size(c) for c in test_wp) / len(test_wp), "uncalibrated (tuned frac)"),
        (f"split-conformal (tau={tau})", conf_cov, conf_size, "GUARANTEED >= 1-alpha"),
    ]
    aps_cov, aps_size = aps(cal_wp, test_wp)

    per_cond = {}
    for cond in CONDITIONS:
        ci = [i for i in cal_i if i.condition == cond]
        ti = [i for i in test_i if i.condition == cond]
        tau_c = calibrate_tau(ci, ALPHA)
        cov_c, _ = coverage_and_size(ti, tau_c)
        per_cond[cond] = (sum(1 for c in test_wp if c["condition"] == cond and c["rank"] <= 2)
                          / max(1, sum(1 for c in test_wp if c["condition"] == cond)), tau_c, cov_c)

    L = ["# Conformal vs naive set baselines (honest comparison)", ""]
    L.append(f"Well-posed test cases: **{len(test_wp)}**; target coverage 1-alpha = {1-ALPHA:.0%}.")
    L.append("")
    L.append("| set rule | coverage | avg size | guarantee |")
    L.append("|---|---:|---:|---|")
    for name, cov, size, guar in rows:
        L.append(f"| {name} | {cov:.2f} | {size:.2f} | {guar} |")
    L.append("")
    L.append("Naive rules reach comparable coverage on this data, so conformal is **not more efficient**.")
    L.append("Its contribution is the **distribution-free guarantee**: only conformal hits a *chosen* level")
    L.append("by construction. The fixed-k and threshold rules give whatever coverage they give — their")
    L.append("number depends on a hand-tuned k/threshold, with no promise it holds at another level or off")
    L.append("this distribution. This is the right way to motivate conformal (calibration, not performance),")
    L.append("and it pre-empts the obvious 'why not just top-k?' question.")
    L.append("")
    L.append("**Per condition (does it adapt?):** fixed top-2 covers "
             f"{per_cond['baseline'][0]:.2f} (baseline) and {per_cond['hardtraps'][0]:.2f} (hard-traps) — it")
    L.append("does *not* collapse under redundancy; conformal simply keeps a safer margin by growing "
             f"tau {per_cond['baseline'][1]}->{per_cond['hardtraps'][1]} (coverage "
             f"{per_cond['baseline'][2]:.2f}->{per_cond['hardtraps'][2]:.2f}).")
    L.append("")
    L.append(f"**Negative result:** an adaptive per-case APS set bloats (coverage {aps_cov:.2f}, avg size "
             f"{aps_size:.2f}) because ContextCite's effect scores are diffuse, so a fixed-tau conformal set")
    L.append("is preferable. (Adaptive sizing is future work, perhaps with a less diffuse base scorer.)")
    OUT.write_text("\n".join(L), encoding="utf-8")

    print(f"well-posed test: {len(test_wp)}")
    for name, cov, size, guar in rows:
        print(f"  {name:30s} cov {cov:.2f} size {size:.2f}  [{guar}]")
    print(f"  adaptive APS (negative)        cov {aps_cov:.2f} size {aps_size:.2f}")
    print(f"  per-condition fixed-top2: baseline {per_cond['baseline'][0]:.2f} / hardtraps {per_cond['hardtraps'][0]:.2f}"
          f"  | conformal tau {per_cond['baseline'][1]}->{per_cond['hardtraps'][1]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
