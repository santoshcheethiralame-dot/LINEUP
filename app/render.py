from __future__ import annotations

from pathlib import Path

from lineup.data.serialization import (
    read_generations,
    read_predictions,
    read_roles,
    read_scenarios,
)
from lineup.downstream import evaluate_abstention
from lineup.scoring import score_predictions

ROLES = ("culprit", "misleading", "silent", "inert")
BADGE_BG = {"culprit": "#E1F5EE", "misleading": "#FCEBEB", "silent": "#FAEEDA", "inert": "#EDEBE3", "—": "#EDEBE3"}
BADGE_FG = {"culprit": "#085041", "misleading": "#791F1F", "silent": "#633806", "inert": "#444441", "—": "#444441"}
BAR = {"culprit": "#1D9E75", "misleading": "#E24B4A", "silent": "#BA7517", "inert": "#9B9A92"}


def load(directory):
    base = Path(directory)

    def read(name, reader):
        path = base / name
        return reader(path) if path.exists() else []

    scenarios = {s.qid: s for s in read("scenarios.jsonl", read_scenarios)}
    generations = {g.qid: g for g in read("generations.jsonl", read_generations)}
    roles = {c.qid: c for c in read("roles.jsonl", read_roles)}
    predictions = read("predictions.jsonl", read_predictions)
    predictions_by_qid: dict = {}
    for prediction in predictions:
        predictions_by_qid.setdefault(prediction.qid, {})[prediction.method] = prediction
    return scenarios, generations, roles, predictions, predictions_by_qid


def _rnd(value):
    return round(value, 2) if isinstance(value, float) else value


def scoring_reports(roles, predictions):
    wrong = [case for case in roles.values() if not case.original_correct]
    return score_predictions(wrong, predictions)


def scoring_table(reports):
    return [
        {
            "method": r.method, "wrong cases": r.n_cases, "with a culprit": r.n_with_culprit,
            "top-1 culprit": _rnd(r.top1_culprit_accuracy), "misleading-as-culprit": _rnd(r.misleading_as_culprit_rate),
            "culprit > misleading": _rnd(r.culprit_over_misleading_winrate), "culprit > rest": _rnd(r.culprit_over_rest_winrate),
        }
        for r in reports
    ]


def abstention_table(generations, roles, predictions):
    reports = evaluate_abstention(list(generations.values()), predictions, list(roles.values()))
    return [
        {"signal": a.signal, "n": a.n, "correct": a.n_correct, "AUROC": _rnd(a.auroc), "AURC": _rnd(a.aurc), "selective acc @50%": _rnd(a.selective_accuracy_at_50)}
        for a in reports
    ]


def role_distribution_html(reports):
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:16px;font-size:12px;opacity:0.8;">'
        f'<span style="width:11px;height:11px;border-radius:2px;background:{BAR[role]};"></span>{role}</span>'
        for role in ROLES
    )
    rows = ""
    for report in reports:
        segments = "".join(
            f'<div style="width:{report.predicted_role_rate.get(role, 0) * 100:.0f}%;background:{BAR[role]};"></div>'
            for role in ROLES
        )
        rows += (
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:9px;">'
            f'<span style="width:170px;font-size:13px;">{report.method}</span>'
            f'<div style="flex:1;display:flex;height:24px;border-radius:6px;overflow:hidden;">{segments}</div></div>'
        )
    return f'<div style="margin-bottom:14px;">{legend}</div>{rows}'


def _role_map(roles, qid):
    case = roles.get(qid)
    return {cr.chunk_id: cr.role for cr in case.chunk_roles} if case else {}


def case_is_fooled(qid, roles, predictions_by_qid):
    role_of = _role_map(roles, qid)
    return any(role_of.get(p.predicted_culprit_id) == "misleading" for p in predictions_by_qid.get(qid, {}).values())


def case_options(scenarios, roles, predictions_by_qid, only_fooled=False):
    return [qid for qid in scenarios if (not only_fooled or case_is_fooled(qid, roles, predictions_by_qid))]


def case_label(qid, scenarios, generations):
    scenario = scenarios[qid]
    generation = generations.get(qid)
    status = "—" if generation is None else ("correct" if generation.is_correct else "wrong")
    question = scenario.question if len(scenario.question) <= 48 else scenario.question[:46] + "…"
    return f"{qid} · {question} · {status}"


def case_detail(qid, scenarios, generations, roles, predictions_by_qid):
    scenario = scenarios[qid]
    generation = generations.get(qid)
    role_of = _role_map(roles, qid)
    predictions = predictions_by_qid.get(qid, {})

    answer = generation.model_answer if generation else "—"
    correct = generation.is_correct if generation else None
    status_bg, status_fg, status_text = (
        ("#E1F5EE", "#085041", "correct") if correct else ("#FCEBEB", "#791F1F", "wrong · model fooled")
    )

    head = (
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="opacity:0.6;font-size:12px;">{qid}</span>'
        f'<span style="font-size:12px;padding:3px 10px;border-radius:6px;background:{status_bg};color:{status_fg};">{status_text}</span></div>'
        f'<div style="margin-top:12px;font-size:16px;">{scenario.question}</div>'
        f'<div style="margin-top:6px;font-size:13px;opacity:0.85;">reference <b>{scenario.gold_answer}</b>'
        f' &nbsp;·&nbsp; model answered <span style="padding:1px 7px;border-radius:6px;background:{status_bg};color:{status_fg};">{answer}</span></div>'
    )

    chunk_rows = '<div style="margin-top:16px;font-size:11px;letter-spacing:0.4px;opacity:0.55;">RETRIEVED CONTEXT · COLORED BY TRUE ROLE</div>'
    numbers = {}
    for index, chunk in enumerate(scenario.chunks, 1):
        numbers[chunk.chunk_id] = index
        role = role_of.get(chunk.chunk_id, "—")
        chunk_rows += (
            f'<div style="display:flex;gap:10px;padding:9px 0;border-top:0.5px solid rgba(128,128,128,0.2);">'
            f'<span style="flex:0 0 78px;text-align:center;font-size:11px;padding:2px 0;height:fit-content;border-radius:6px;background:{BADGE_BG.get(role, "#EDEBE3")};color:{BADGE_FG.get(role, "#444441")};">{role}</span>'
            f'<div style="flex:1;"><div style="font-size:13px;"><span style="opacity:0.5;">[{index}]</span> {chunk.title}</div>'
            f'<div style="font-size:12px;opacity:0.75;line-height:1.5;">{chunk.text}</div></div></div>'
        )

    method_rows = '<div style="margin-top:16px;font-size:11px;letter-spacing:0.4px;opacity:0.55;">WHO EACH METHOD BLAMED</div>'
    fooled = 0
    for method in sorted(predictions):
        picked = predictions[method].predicted_culprit_id
        picked_role = role_of.get(picked, "—")
        if picked_role == "misleading":
            fooled += 1
            verdict = '<span style="color:#A32D2D;">✗ blamed a near-miss</span>'
        elif picked_role == "culprit":
            verdict = '<span style="color:#0F6E56;">✓ found the culprit</span>'
        else:
            verdict = f'<span style="opacity:0.7;">picked a {picked_role} chunk</span>'
        method_rows += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-top:0.5px solid rgba(128,128,128,0.2);">'
            f'<span style="font-size:13px;">{method}</span>'
            f'<span style="font-size:12px;opacity:0.85;">blamed <b>[{numbers.get(picked, "?")}]</b> · {verdict}</span></div>'
        )

    note = ""
    if fooled and not correct:
        note = (
            f'<div style="margin-top:16px;background:rgba(128,128,128,0.08);border-radius:6px;padding:11px 13px;font-size:12.5px;opacity:0.9;line-height:1.6;">'
            f'{fooled} of {len(predictions)} methods blamed a chunk that only <i>looks</i> responsible — the oracle marks it misleading, not the cause.</div>'
        )

    return f'<div style="border:0.5px solid rgba(128,128,128,0.25);border-radius:12px;padding:1rem 1.25rem;">{head}{chunk_rows}{method_rows}{note}</div>'
