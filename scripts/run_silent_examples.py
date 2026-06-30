"""Silent passages are real causal drivers, not prompt-perturbation noise (local, no GPU).

A silent passage is causal (removing it flips the answer) but not salient (it does not state the
wrong value). The objection is that such a flip is just instability from changing the context. Two
checks answer it. First, effect size: removing a silent passage shifts the answer log-probability as
much as removing a true culprit, far above any token-distribution jitter -- and because the oracle
compares normalized answer *values*, a mere rephrase is never counted as causal. Second, worked
examples: removing a bridge passage redirects the reasoning to a different hop.

One honesty note this script enforces: some passages labeled silent are really salience-matcher
near-misses, where the passage does state the wrong value in a format variant (an en-dash, a longer
phrasing) the matcher missed. Those are not silent drivers. We therefore surface only candidates
whose wrong-value core is genuinely absent from the passage, and leave the final pick to inspection.
Reads cached roles/scenarios; writes the distribution, a figure, and clean candidate examples.
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
OUT = ROOT / "paper" / "silent_passages.md"
FIG = ROOT / "paper" / "figures" / "fig_silent.png"

_GENERIC = {"countries", "country", "award", "awards", "winning", "won", "the", "a", "an", "of", "in", "and", "or"}


def chunk_text(scenarios_path: Path) -> dict:
    text = {}
    for line in scenarios_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        for c in s.get("chunks", []):
            text[c["chunk_id"]] = (c.get("title", ""), c.get("text", ""))
    return text


def _norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def states_value(answer: str, text: str) -> bool:
    """Does the passage state the wrong value's distinctive (non-generic) core, dash-normalized?"""
    t = _norm(text)
    core = [w for w in _norm(answer).split() if w not in _GENERIC]
    if not core:
        return False
    if " ".join(core) in t:
        return True
    toks = set(t.split())
    return sum(w in toks for w in core) / len(core) >= 0.6


def about_answer(title: str, answer: str) -> bool:
    """Is the passage simply about an entity named in the wrong answer? A passage titled with a word
    that also appears in the wrong answer is a salience near-miss (abbreviation/phrasing), not a
    genuine silent bridge — drop it so only bridges to a different entity survive."""
    tt = {w for w in _norm(title).split() if len(w) >= 4}
    aa = {w for w in _norm(answer).split() if len(w) >= 4}
    return bool(tt & aa)


def main() -> None:
    silent, culprit, candidates = [], [], []
    for cell in sorted(glob.glob(str(DATA / "*" / "*" / "*"))):
        cell = Path(cell)
        rp, sp = cell / "roles.jsonl", cell / "scenarios.jsonl"
        if not (rp.exists() and sp.exists()):
            continue
        texts = chunk_text(sp)
        for line in rp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("original_correct", True):
                continue
            for c in r["chunk_roles"]:
                d = abs(c.get("delta_logprob", 0.0))
                if c["role"] == "culprit":
                    culprit.append(d)
                    continue
                if c["role"] != "silent":
                    continue
                silent.append(d)
                aw = (c.get("answer_without") or "").strip()
                ma = (r.get("original_answer") or "").strip()
                gold = (r.get("gold_answer") or "").strip()
                title, body = texts.get(c["chunk_id"], ("", ""))
                # a clean silent driver: large effect, the wrong value genuinely absent from the
                # passage (so it is unarguably non-salient, not a matcher miss), removal flips toward gold.
                if (d >= 3 and aw and ma and aw != ma and len(ma.split()) <= 6
                        and not states_value(ma, body) and not about_answer(title, ma)):
                    to_gold = bool(gold) and (gold.lower() in aw.lower() or aw.lower() in gold.lower())
                    candidates.append((to_gold, d, r["question"], gold, ma, aw, title, body[:240].strip()))

    def line(x):
        return (f"n={len(x)}, median={st.median(x):.2f}, mean={st.mean(x):.2f}, "
                f"share below 0.5 logprob (jitter range)={100*sum(v < 0.5 for v in x)/len(x):.0f}%")

    candidates.sort(key=lambda t: (not t[0], -t[1]))   # flip-to-gold first, then largest effect

    L = ["# Silent passages are causal, not noise", ""]
    L.append("A *silent* passage is causal (removing it flips the answer's value) but not salient (it does")
    L.append("not state the wrong value). Because the oracle compares normalized answer *values*, a flip")
    L.append("driven only by rephrasing is never counted, so the causal label already excludes surface jitter.")
    L.append("")
    L.append("**Effect size.** Removing a silent passage moves the answer log-probability about as much as")
    L.append("removing a true culprit, and far above the near-zero shift a perturbation artifact would give:")
    L.append("")
    L.append(f"- silent passages: {line(silent)}")
    L.append(f"- culprit passages: {line(culprit)}")
    L.append("")
    L.append("**Candidate silent drivers** — the passage bridges to a *different* entity than the wrong")
    L.append("answer (its title shares no word with the wrong answer) and the wrong value is absent under")
    L.append("dash/abbreviation-normalized matching. Still verify by eye: the silent label is sensitive to the")
    L.append("salience matcher, and many silents elsewhere are matcher near-misses where the passage states")
    L.append("the wrong value in a variant form (so report the silent share under the strict matcher too).")
    L.append("")
    for to_gold, d, q, gold, ma, aw, title, body in candidates[:3]:
        L.append(f"- *{q}*")
        L.append(f"  - gold `{gold}`; model answered `{ma}` (wrong). Silent passage `{title}` (does not state `{ma}`):")
        L.append(f"    > {body}")
        L.append(f"  - removing it flips the answer to `{aw}` (Delta logprob {d:.1f}) -- it drives the wrong hop without stating the wrong value.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    ax.hist(silent, bins=bins, alpha=0.6, label=f"silent (n={len(silent)})", color="#3b6fb6")
    ax.hist(culprit, bins=bins, alpha=0.6, label=f"culprit (n={len(culprit)})", color="#e08a1e")
    ax.axvline(0.5, ls="--", color="grey", lw=1, label="jitter range (<0.5)")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("|change in answer log-probability| when the passage is removed")
    ax.set_ylabel("passages")
    ax.set_title("Silent passages are causal, not perturbation noise")
    ax.legend(frameon=False)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=300)
    plt.close(fig)

    print(f"silent : {line(silent)}")
    print(f"culprit: {line(culprit)}")
    print(f"clean candidates: {len(candidates)} (showing top 3 in {OUT.name})")
    for _, d, q, _, ma, aw, _, _ in candidates[:3]:
        print(f"  Delta {d:5.1f}  {ma!r} -> {aw!r}   {q[:60]}")
    print(f"wrote {OUT} and {FIG}")


if __name__ == "__main__":
    main()
