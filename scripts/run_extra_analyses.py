"""Two quick depth checks for the appendix (local, no Kaggle, no new main-body figures):

  (7) Position effect: does ContextCite find the culprit better in some context positions?
      (a 'lost in the middle' check for attribution; culprit position is randomized by construction).
  (8) Graded correctness: do near-miss errors (answer shares a token with gold) differ in no-culprit
      rate from total-miss errors?

Both pooled across the 12 cells. Output is a short markdown note; null results are reported honestly.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
OUT = ROOT / "paper" / "extra_analyses.md"
MODELS = ["qwen", "phi", "mistral"]
DATASETS = ["hotpotqa", "2wiki"]
CONDS = ["baseline", "hardtraps"]
_W = re.compile(r"\w+")


def toks(s):
    return {w for w in _W.findall((s or "").lower()) if len(w) >= 3}


def main():
    # position: top-1 hit by culprit position third
    pos_hit = {"early": [0, 0], "middle": [0, 0], "late": [0, 0]}   # [hits, n]
    # graded: no-culprit rate by near-miss vs total-miss
    graded = {"near-miss": [0, 0], "total-miss": [0, 0]}            # [no_culprit, n]

    for ds in DATASETS:
        for cond in CONDS:
            for model in MODELS:
                d = DATA / ds / cond / model
                if not all((d / f).exists() for f in ("roles.jsonl", "scenarios.jsonl", "generations.jsonl", "predictions.jsonl")):
                    continue
                order = {}
                for line in (d / "scenarios.jsonl").read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        s = json.loads(line)
                        order[s["qid"]] = [c["chunk_id"] for c in s["chunks"]]
                gen = {}
                for line in (d / "generations.jsonl").read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        g = json.loads(line)
                        gen[g["qid"]] = g
                cc_top = {}
                for line in (d / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        p = json.loads(line)
                        if p.get("method") == "contextcite":
                            cc_top[p["qid"]] = p["predicted_culprit_id"]
                for line in (d / "roles.jsonl").read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["original_correct"]:
                        continue
                    qid = r["qid"]
                    culprits = [c["chunk_id"] for c in r["chunk_roles"] if c["role"] == "culprit"]
                    # graded correctness
                    g = gen.get(qid, {})
                    near = bool(toks(g.get("model_answer", "")) & toks(r.get("gold_answer", "")))
                    key = "near-miss" if near else "total-miss"
                    graded[key][1] += 1
                    if not culprits:
                        graded[key][0] += 1
                    # position effect (well-posed only)
                    if len(culprits) == 1 and qid in order and culprits[0] in order[qid]:
                        ids = order[qid]
                        frac = ids.index(culprits[0]) / max(1, len(ids))
                        bucket = "early" if frac < 1 / 3 else "middle" if frac < 2 / 3 else "late"
                        pos_hit[bucket][1] += 1
                        if cc_top.get(qid) == culprits[0]:
                            pos_hit[bucket][0] += 1

    L = ["# Appendix: position and graded-correctness checks", ""]
    L.append("## Position effect (ContextCite top-1 by culprit position)")
    L.append("Culprit position is randomized by construction, so this tests attribution's positional bias.")
    L.append("")
    L.append("| position | top-1 culprit acc | n |")
    L.append("|---|---:|---:|")
    for b in ("early", "middle", "late"):
        h, n = pos_hit[b]
        L.append(f"| {b} | {h/n:.2f} | {n} |" if n else f"| {b} | -- | 0 |")
    accs = [pos_hit[b][0] / pos_hit[b][1] for b in pos_hit if pos_hit[b][1]]
    spread = (max(accs) - min(accs)) if accs else 0
    L.append("")
    L.append(f"Spread across thirds: **{spread:.2f}**. "
             + ("Small — no strong 'lost in the middle' effect for attribution." if spread < 0.1
                else "Notable — attribution accuracy varies with culprit position."))
    L.append("")
    L.append("## Graded correctness (no-culprit rate by error type)")
    L.append("A *near-miss* error shares a content token with the gold answer; a *total-miss* does not.")
    L.append("")
    L.append("| error type | no-culprit % | n |")
    L.append("|---|---:|---:|")
    for k in ("near-miss", "total-miss"):
        nc, n = graded[k]
        L.append(f"| {k} | {100*nc/n:.0f} | {n} |" if n else f"| {k} | -- | 0 |")
    gvals = [graded[k][0] / graded[k][1] for k in graded if graded[k][1]]
    gdiff = (max(gvals) - min(gvals)) if gvals else 0
    L.append("")
    L.append(f"Difference: **{100*gdiff:.0f} points**. "
             + ("Both error types are comparably ill-posed." if gdiff < 0.1
                else "No-culprit rate differs by error type."))
    OUT.write_text("\n".join(L), encoding="utf-8")

    print("position (top-1 by third):", {b: (f"{pos_hit[b][0]/pos_hit[b][1]:.2f}" if pos_hit[b][1] else "na") for b in pos_hit})
    print(f"  spread {spread:.2f}")
    print("graded (no-culprit by type):", {k: (f"{100*graded[k][0]/graded[k][1]:.0f}% (n={graded[k][1]})" if graded[k][1] else "na") for k in graded})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
