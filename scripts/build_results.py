"""Assemble the paper's results tables from whatever run cells are present.

Drop each downloaded zip in with `--ingest`, then run with no arguments to (re)build
paper/results_tables.md. Cells that have not been run yet show as "pending", so the
tables fill themselves in as the experiment matrix completes.

Layout it reads/writes:
    paper/results_data/<dataset>/<condition>/<model>/{roles,predictions,...}.jsonl
"""
import argparse
import re
import zipfile
from pathlib import Path

from lineup.agreement import compare_models
from lineup.data.serialization import read_predictions, read_roles
from lineup.scoring import score_predictions
from lineup.setvalued import attribution_recovery

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "results_data"
OUT = ROOT / "paper" / "results_tables.md"

DATASETS = ["hotpotqa", "2wiki"]
CONDITIONS = ["baseline", "hardtraps"]
MODELS = ["qwen", "phi", "mistral"]
MODEL_LABEL = {"qwen": "Qwen2.5-7B", "phi": "Phi-3.5-mini", "mistral": "Mistral-7B"}
METHODS = ["contextcite", "single_chunk", "llm_judge", "lexical_similarity"]


def cell_dir(dataset, condition, model):
    return DATA / dataset / condition / model


def load_cell(dataset, condition, model):
    base = cell_dir(dataset, condition, model)
    roles_path, preds_path = base / "roles.jsonl", base / "predictions.jsonl"
    if not (roles_path.exists() and preds_path.exists()):
        return None
    roles = read_roles(roles_path)
    wrong = [case for case in roles if not case.original_correct]
    preds = read_predictions(preds_path)
    return {
        "roles": roles,
        "wrong": wrong,
        "scores": {r.method: r for r in score_predictions(wrong, preds)},
        "recovery": {r.method: r for r in attribution_recovery(wrong, preds)},
    }


def ingest(zips):
    for path in zips:
        name = Path(path).stem
        match = re.match(r"lineup_(hotpotqa|2wiki)_(baseline|hardtraps)_", name)
        if not match:
            print(f"  skip (name does not encode dataset/condition): {path}")
            continue
        dataset, condition = match.group(1), match.group(2)
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            models = sorted({m.split("/")[0] for m in members if "/" in m})
            for model in models:
                dest = cell_dir(dataset, condition, model)
                dest.mkdir(parents=True, exist_ok=True)
                for stem in ("scenarios", "generations", "roles", "predictions"):
                    member = f"{model}/{stem}.jsonl"
                    if member in members:
                        dest.joinpath(f"{stem}.jsonl").write_bytes(archive.read(member))
                print(f"  ingested {dataset}/{condition}/{model}")


def _fmt(value, places=2):
    return f"{value:.{places}f}" if value is not None else "—"


def _table_accuracy(cells):
    rows = [
        "## Table 1 — Attribution accuracy (top-1 culprit, wrong cases)",
        "",
        "| dataset | condition | model | n_wrong | " + " | ".join(METHODS) + " |",
        "|---|---|---|--:|" + "--:|" * len(METHODS),
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for model in MODELS:
                cell = cells[(dataset, condition, model)]
                if cell is None:
                    rows.append(f"| {dataset} | {condition} | {MODEL_LABEL[model]} | pending | " + " | ".join(["—"] * len(METHODS)) + " |")
                    continue
                tops = [_fmt(cell["scores"][m].top1_culprit_accuracy) if m in cell["scores"] else "—" for m in METHODS]
                rows.append(f"| {dataset} | {condition} | {MODEL_LABEL[model]} | {len(cell['wrong'])} | " + " | ".join(tops) + " |")
    return "\n".join(rows)


def _table_core(cells):
    rows = [
        "## Table 2 — Ill-posedness and set recovery (ContextCite)",
        "",
        "no-culprit% = errors with no single causal culprit. recall@1 vs recall@k = single pick vs top-|R| set.",
        "",
        "| dataset | condition | model | no-culprit% | recall@1 | recall@k | reliability AUROC | single-culprit AUROC |",
        "|---|---|---|--:|--:|--:|--:|--:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for model in MODELS:
                cell = cells[(dataset, condition, model)]
                if cell is None:
                    rows.append(f"| {dataset} | {condition} | {MODEL_LABEL[model]} | pending | — | — | — | — |")
                    continue
                wrong = cell["wrong"]
                no_culprit = sum(1 for c in wrong if not any(r.role == "culprit" for r in c.chunk_roles))
                pct = f"{100 * no_culprit / len(wrong):.0f}%" if wrong else "—"
                rec = cell["recovery"].get("contextcite")
                if rec is None:
                    rows.append(f"| {dataset} | {condition} | {MODEL_LABEL[model]} | {pct} | — | — | — | — |")
                    continue
                rows.append(
                    f"| {dataset} | {condition} | {MODEL_LABEL[model]} | {pct} | "
                    f"{_fmt(rec.recall_at_1)} | {_fmt(rec.recall_at_k)} | "
                    f"{_fmt(rec.reliability_auroc)} | {_fmt(rec.single_culprit_auroc)} |"
                )
    return "\n".join(rows)


def _table_agreement(cells):
    rows = [
        "## Table 3 — Cross-model agreement (per-passage role kappa, both-wrong cases)",
        "",
        "| dataset | condition | qwen-vs-phi | qwen-vs-mistral | phi-vs-mistral |",
        "|---|---|--:|--:|--:|",
    ]
    pairs = [("qwen", "phi"), ("qwen", "mistral"), ("phi", "mistral")]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            kappas = []
            for a, b in pairs:
                ca, cb = cells[(dataset, condition, a)], cells[(dataset, condition, b)]
                if ca is None or cb is None:
                    kappas.append("pending")
                else:
                    report = compare_models(ca["roles"], cb["roles"])
                    kappas.append(_fmt(report.role_kappa))
            rows.append(f"| {dataset} | {condition} | " + " | ".join(kappas) + " |")
    return "\n".join(rows)


def report():
    cells = {
        (dataset, condition, model): load_cell(dataset, condition, model)
        for dataset in DATASETS for condition in CONDITIONS for model in MODELS
    }
    done = sum(1 for cell in cells.values() if cell is not None)
    header = [
        "# LINEUP results",
        "",
        f"Matrix coverage: {done}/{len(cells)} cells. Regenerate with `python scripts/build_results.py`.",
        "",
    ]
    body = "\n\n".join([_table_accuracy(cells), _table_core(cells), _table_agreement(cells)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header) + "\n" + body + "\n", encoding="utf-8")
    print(f"wrote {OUT}  ({done}/{len(cells)} cells present)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest", nargs="+", default=None, help="run zip(s) named lineup_<dataset>_<condition>_<models>.zip")
    args = parser.parse_args()
    if args.ingest:
        ingest(args.ingest)
    report()


if __name__ == "__main__":
    main()
