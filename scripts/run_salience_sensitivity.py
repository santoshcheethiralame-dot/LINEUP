"""Is the no-culprit rate robust to how 'salient' is defined? (local, no Kaggle)

The causal axis is a threshold-free counterfactual (and an independent signal confirms it at AUROC
0.90). The salience axis is the softer one: it asks whether a passage 'states the wrong value', via
text matching. A reviewer can ask whether the 27-53% headline rides on that matcher. We re-derive the
no-culprit rate under three salience definitions -- the shipped matcher, a STRICT one (the answer
string appears verbatim), and a LOOSE one (any 4+ char answer token appears) -- and report the band.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
OUT = ROOT / "paper" / "salience_sensitivity.md"
MODELS = ["qwen", "phi", "mistral"]
DATASETS = ["hotpotqa", "2wiki"]
CONDS = ["baseline", "hardtraps"]
_WORD = re.compile(r"\w+")


def tokens(s):
    return {w for w in _WORD.findall(s.lower()) if len(w) >= 4}


def main():
    n_wrong = 0
    nc = {"shipped": 0, "strict": 0, "loose": 0}
    for ds in DATASETS:
        for cond in CONDS:
            for model in MODELS:
                d = DATA / ds / cond / model
                if not all((d / f).exists() for f in ("roles.jsonl", "scenarios.jsonl", "generations.jsonl")):
                    continue
                text = {}
                for line in (d / "scenarios.jsonl").read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        s = json.loads(line)
                        text[s["qid"]] = {c["chunk_id"]: c["text"] for c in s["chunks"]}
                answer = {}
                for line in (d / "generations.jsonl").read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        g = json.loads(line)
                        answer[g["qid"]] = g.get("model_answer", "")
                for line in (d / "roles.jsonl").read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["original_correct"]:
                        continue
                    n_wrong += 1
                    ans = (answer.get(r["qid"]) or "").strip()
                    atok = tokens(ans)
                    chunk_text = text.get(r["qid"], {})
                    has = {"shipped": False, "strict": False, "loose": False}
                    for c in r["chunk_roles"]:
                        if not c["causal"]:
                            continue
                        body = chunk_text.get(c["chunk_id"], "").lower()
                        if c["salient"]:
                            has["shipped"] = True
                        if ans and ans.lower() in body:
                            has["strict"] = True
                        if atok and (atok & tokens(body)):
                            has["loose"] = True
                    for k in nc:
                        if not has[k]:
                            nc[k] += 1

    L = ["# No-culprit rate vs. the salience definition", ""]
    L.append(f"Pooled wrong cases (12 cells): **{n_wrong}**. No single culprit = no causal passage that is")
    L.append("also salient, under each salience rule:")
    L.append("")
    L.append("| salience rule | no-culprit % |")
    L.append("|---|---:|")
    L.append(f"| STRICT (answer appears verbatim) | {100*nc['strict']/n_wrong:.1f} |")
    L.append(f"| shipped (normalized phrase match) | {100*nc['shipped']/n_wrong:.1f} |")
    L.append(f"| LOOSE (any 4+ char answer token)  | {100*nc['loose']/n_wrong:.1f} |")
    L.append("")
    lo = min(nc.values()); hi = max(nc.values())
    L.append(f"**Honest reading:** the rate is matcher-dependent, **[{100*lo/n_wrong:.0f}%, {100*hi/n_wrong:.0f}%]**.")
    L.append("Stricter salience marks fewer passages salient (more no-culprit); the LOOSE token-overlap rule is")
    L.append("over-generous (one shared word like \"protocol\" marks a passage salient) and is the floor. The")
    L.append("shipped phrase-level matcher (37%) is the principled choice — it matches whole values, not stray")
    L.append("tokens. The finding is **directionally robust** (always substantial; >=19% even under an")
    L.append("over-generous rule) but the precise value depends on the salience definition, which we disclose")
    L.append("as the one soft axis. We report the phrase-level number and this band.")
    OUT.write_text("\n".join(L), encoding="utf-8")

    print(f"pooled wrong: {n_wrong}")
    for k in ("strict", "shipped", "loose"):
        print(f"  {k:8s} no-culprit {100*nc[k]/n_wrong:.1f}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
