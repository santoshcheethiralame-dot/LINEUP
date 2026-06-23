"""#3 Does selective attribution abstain on the *right* cases? (local)

The selective remedy abstains on low-confidence cases (smallest top1-top2 margin in the method's
scores). The killer validation: are the abstained cases disproportionately the ill-posed
(no-culprit / coalition) ones? If so, the method stays silent exactly where the problem is
ill-posed -- abstaining for the right reason, not just dropping hard cases at random.

We rank pooled wrong cases by the ContextCite margin, abstain the lowest-margin fraction at several
coverage levels, and compare the no-culprit rate of answered vs abstained.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
OUT = ROOT / "paper" / "abstention_targeting.md"

MODELS = ["qwen", "phi", "mistral"]
DATASETS = ["hotpotqa", "2wiki"]
CONDS = ["baseline", "hardtraps"]


def main():
    cases = []  # (margin, no_culprit)
    for ds in DATASETS:
        for cond in CONDS:
            for model in MODELS:
                d = DATA / ds / cond / model
                rp, pp = d / "roles.jsonl", d / "predictions.jsonl"
                if not (rp.exists() and pp.exists()):
                    continue
                nc = {}
                for l in rp.read_text(encoding="utf-8").splitlines():
                    if l.strip():
                        r = json.loads(l)
                        if not r.get("original_correct", True):
                            nc[r["qid"]] = int(not any(c["causal"] and c["salient"] for c in r["chunk_roles"]))
                for l in pp.read_text(encoding="utf-8").splitlines():
                    if not l.strip():
                        continue
                    pr = json.loads(l)
                    if pr.get("method") != "contextcite" or pr["qid"] not in nc:
                        continue
                    s = sorted((c["score"] for c in pr["chunk_scores"]), reverse=True)
                    margin = (s[0] - s[1]) if len(s) >= 2 else 0.0
                    cases.append((margin, nc[pr["qid"]]))

    cases.sort()  # ascending margin -> the front is what gets abstained first
    n = len(cases)
    base = sum(y for _, y in cases) / n

    L = ["# Does abstention target the ill-posed cases?", ""]
    L.append(f"Pooled wrong cases: **{n}**. Overall no-culprit rate: **{100*base:.1f}%**.")
    L.append("Cases are ranked by ContextCite margin (top1-top2); the lowest-margin fraction is abstained.")
    L.append("")
    L.append("| coverage (answered) | no-culprit % in ANSWERED | no-culprit % in ABSTAINED |")
    L.append("|---:|---:|---:|")
    for cov in (0.9, 0.8, 0.7, 0.6, 0.5):
        n_abst = int(round((1 - cov) * n))
        abst = cases[:n_abst]
        ans = cases[n_abst:]
        nc_abst = 100 * sum(y for _, y in abst) / len(abst) if abst else float("nan")
        nc_ans = 100 * sum(y for _, y in ans) / len(ans) if ans else float("nan")
        L.append(f"| {cov:.0%} | {nc_ans:.1f} | {nc_abst:.1f} |")
    L.append("")
    L.append("Reading: if the no-culprit rate in the abstained set is far above the answered set, the")
    L.append("remedy concentrates the ill-posed cases into exactly the cases it declines -- it abstains for")
    L.append("the right reason. This is the direct validation of the selective remedy (C2).")
    OUT.write_text("\n".join(L), encoding="utf-8")

    print(f"pooled wrong cases: {n}  overall no-culprit {100*base:.1f}%")
    print("coverage | no-culprit ANSWERED | no-culprit ABSTAINED")
    for cov in (0.9, 0.8, 0.7, 0.6, 0.5):
        n_abst = int(round((1 - cov) * n))
        abst = cases[:n_abst]; ans = cases[n_abst:]
        nc_abst = 100 * sum(y for _, y in abst) / len(abst) if abst else float("nan")
        nc_ans = 100 * sum(y for _, y in ans) / len(ans) if ans else float("nan")
        print(f"  {cov:.0%}      | {nc_ans:5.1f}%             | {nc_abst:5.1f}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
