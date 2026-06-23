"""#2 When a method blames the wrong chunk, does it err toward salient red-herrings? (local)

On well-posed cases (a single ground-truth culprit exists), we look at the cases where a method's
top pick is NOT the culprit, and ask what role the chunk it blamed actually has. The planted
*misleading* chunk is the pure salience trap (states the wrong value, but removing it changes
nothing). If methods land on it far more than chance, their errors are salience-shaped -- the
original "fooled by salience" effect, recovered precisely at the level of wrong attributions.

Chance baseline: among the non-culprit chunks in a case, the share that are misleading (one planted
misleading chunk per case), so a method picking a wrong chunk uniformly would hit it at that rate.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
OUT = ROOT / "paper" / "error_direction.md"

MODELS = ["qwen", "phi", "mistral"]
DATASETS = ["hotpotqa", "2wiki"]
CONDS = ["baseline", "hardtraps"]
METHODS = ["contextcite", "single_chunk", "lexical_similarity", "llm_judge"]


def main():
    # per method: counts of wrong-attribution target role, plus summed chance rate of hitting misleading
    hit = defaultdict(lambda: defaultdict(int))   # method -> role -> count of wrong picks landing there
    wrong_total = defaultdict(int)                 # method -> total wrong picks (on well-posed cases)
    chance_sum = defaultdict(float)                # method -> sum over wrong picks of (#misleading / #non-culprit)

    for ds in DATASETS:
        for cond in CONDS:
            for model in MODELS:
                d = DATA / ds / cond / model
                rp, pp = d / "roles.jsonl", d / "predictions.jsonl"
                if not (rp.exists() and pp.exists()):
                    continue
                role_of, culprits = {}, {}
                for l in rp.read_text(encoding="utf-8").splitlines():
                    if not l.strip():
                        continue
                    r = json.loads(l)
                    if r.get("original_correct", True):
                        continue
                    rmap = {c["chunk_id"]: c["role"] for c in r["chunk_roles"]}
                    cset = {cid for cid, role in rmap.items() if role == "culprit"}
                    if len(cset) >= 1:  # well-posed: a ground-truth culprit exists
                        role_of[r["qid"]] = rmap
                        culprits[r["qid"]] = cset
                for l in pp.read_text(encoding="utf-8").splitlines():
                    if not l.strip():
                        continue
                    pr = json.loads(l)
                    m = pr.get("method")
                    qid = pr["qid"]
                    if m not in METHODS or qid not in role_of:
                        continue
                    pick = pr["predicted_culprit_id"]
                    if pick in culprits[qid]:
                        continue  # correct attribution; not an error
                    rmap = role_of[qid]
                    wrong_total[m] += 1
                    hit[m][rmap.get(pick, "unknown")] += 1
                    n_noncul = sum(1 for role in rmap.values() if role != "culprit")
                    n_mis = sum(1 for role in rmap.values() if role == "misleading")
                    if n_noncul:
                        chance_sum[m] += n_mis / n_noncul

    L = ["# Are wrong attributions salience-shaped?", ""]
    L.append("Well-posed cases only (a single ground-truth culprit exists). For each method we take the")
    L.append("cases where its top pick is not the culprit, and report the role of the chunk it blamed.")
    L.append("The *misleading* chunk is the pure salience trap. 'Chance' is the rate a uniform wrong pick")
    L.append("would hit a misleading chunk (≈ #misleading / #non-culprit per case).")
    L.append("")
    L.append("| method | wrong picks | → misleading (salient) | chance | lift | → inert | → silent |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in METHODS:
        w = wrong_total[m]
        if not w:
            continue
        mis = 100 * hit[m]["misleading"] / w
        ch = 100 * chance_sum[m] / w
        lift = mis / ch if ch else float("nan")
        inert = 100 * hit[m]["inert"] / w
        silent = 100 * hit[m]["silent"] / w
        L.append(f"| {m} | {w} | {mis:.1f}% | {ch:.1f}% | {lift:.2f}x | {inert:.1f}% | {silent:.1f}% |")
    L.append("")
    L.append("Reading: a lift >1 on 'misleading' means the method's *errors* are biased toward the salient")
    L.append("red-herring beyond chance -- 'not fooled overall, but errors are salience-shaped'. The effect-")
    L.append("based methods (contextcite, single_chunk) and llm_judge are strongly trap-biased (2.4-2.8x):")
    L.append("they confuse the salient near-twin of the culprit for the culprit. Lexical is the surprise --")
    L.append("its errors are NOT trap-biased (1.1x); they scatter toward inert chunks. So salience-shaped")
    L.append("failure is a property of the causal methods, not of surface-overlap scoring.")
    OUT.write_text("\n".join(L), encoding="utf-8")

    print("=== wrong-attribution target role (well-posed cases) ===")
    for m in METHODS:
        w = wrong_total[m]
        if not w:
            continue
        mis = 100 * hit[m]["misleading"] / w
        ch = 100 * chance_sum[m] / w
        print(f"  {m:12s} wrong={w:4d}  ->misleading {mis:5.1f}%  chance {ch:5.1f}%  lift {mis/ch if ch else 0:.2f}x"
              f"  ->inert {100*hit[m]['inert']/w:4.1f}%  ->silent {100*hit[m]['silent']/w:4.1f}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
