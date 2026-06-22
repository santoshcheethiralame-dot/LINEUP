"""Pull three archetypal cases into a readable markdown table for the paper's qualitative section:
a clean single culprit, an ill-posed coalition, and a silent mis-attribution onto a gold passage.
"""
from pathlib import Path

from lineup.data.serialization import read_predictions, read_roles, read_scenarios

ROOT = Path(__file__).resolve().parent.parent
CELL = ROOT / "paper" / "results_data" / "hotpotqa" / "hardtraps" / "qwen"
OUT = ROOT / "paper" / "examples.md"


def _short(text, limit=160):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render(title, caption, scenario, case, prediction):
    scores = {s.chunk_id: s.score for s in prediction.chunk_scores}
    role = {r.chunk_id: r for r in case.chunk_roles}
    letters = "ABCDEFGH"
    lines = [f"### {title}", ""]
    if caption:
        lines += [f"*{caption}*", ""]
    lines += [
        f"**Question:** {case.question}",
        "",
        f"**Gold answer:** {case.gold_answer}  |  **Model answered:** {case.original_answer}  (wrong)",
        "",
        "| # | provenance | oracle role | ContextCite | passage |",
        "|---|---|---|--:|---|",
    ]
    picked = prediction.predicted_culprit_id
    for index, chunk in enumerate(scenario.chunks):
        r = role.get(chunk.chunk_id)
        mark = " ◄ picked" if chunk.chunk_id == picked else ""
        lines.append(
            f"| {letters[index]} | {chunk.provenance} | {r.role if r else '?'} | "
            f"{scores.get(chunk.chunk_id, 0):.1f} | {_short(chunk.text)}{mark} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    scenarios = {s.qid: s for s in read_scenarios(CELL / "scenarios.jsonl")}
    cases = {c.qid: c for c in read_roles(CELL / "roles.jsonl") if not c.original_correct}
    preds = {p.qid: p for p in read_predictions(CELL / "predictions.jsonl") if p.method == "contextcite"}

    def role_of(case, chunk_id):
        for r in case.chunk_roles:
            if r.chunk_id == chunk_id:
                return r.role
        return None

    clean, coalition, trap = [], [], []
    cap = 3
    for qid, case in cases.items():
        pred = preds.get(qid)
        scenario = scenarios.get(qid)
        if pred is None or scenario is None:
            continue
        culprits = [r.chunk_id for r in case.chunk_roles if r.role == "culprit"]
        picked_role = role_of(case, pred.predicted_culprit_id)
        # require the model's wrong answer to actually be the planted value, so the case is a
        # clean illustration of the trap rather than an unrelated hallucination
        planted = (scenario.recipe.intended_wrong_answer or "").lower()
        answer = (case.original_answer or "").lower()
        adopted = bool(planted) and (planted in answer or answer in planted)
        if len(clean) < cap and len(culprits) == 1 and pred.predicted_culprit_id in culprits and adopted:
            clean.append((scenario, case, pred))
        if len(coalition) < cap and not culprits and sum(1 for r in case.chunk_roles if r.provenance in ("misleading", "decoy")) >= 2:
            coalition.append((scenario, case, pred))
        if len(trap) < cap and culprits and picked_role == "misleading" and adopted:
            trap.append((scenario, case, pred))

    blocks = [
        "# Qualitative examples — candidate pool",
        "",
        "Several candidates per archetype; **pick the clearest one of each for the paper.**",
        "",
    ]
    for label, caption, pool in [
        ("CLEAN CULPRIT — attribution succeeds",
         "One passage states the wrong value and removing it fixes the answer; ContextCite finds it.", clean),
        ("COALITION — no single culprit (ill-posed)",
         "Two passages assert the wrong value, so removing either leaves the error; no single chunk is causal, yet the method still names one.", coalition),
        ("SALIENCE TRAP — blames a salient-but-not-causal passage",
         "A real culprit is present, but the method blames a passage the oracle proves is salient but not causal.", trap),
    ]:
        blocks.append(f"## {label}\n\n*{caption}*\n")
        for index, candidate in enumerate(pool, 1):
            blocks.append(_render(f"Candidate {index}", "", *candidate))
        if not pool:
            blocks.append("_(no candidate found)_\n")
    OUT.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(f"wrote {OUT}  (clean {len(clean)}, coalition {len(coalition)}, trap {len(trap)})")


if __name__ == "__main__":
    main()
