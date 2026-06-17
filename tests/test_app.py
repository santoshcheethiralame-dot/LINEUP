import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import render

SAMPLE = str(APP / "sample")


def test_sample_loads():
    scenarios, generations, roles, predictions, by_qid = render.load(SAMPLE)
    assert len(scenarios) == 6
    assert predictions and roles


def test_scoring_table_has_a_row_per_method():
    _, _, roles, predictions, _ = render.load(SAMPLE)
    reports = render.scoring_reports(roles, predictions)
    methods = {row["method"] for row in render.scoring_table(reports)}
    assert {"contextcite", "lexical_similarity", "llm_judge"} <= methods


def test_case_detail_flags_a_fooled_case():
    scenarios, generations, roles, _, by_qid = render.load(SAMPLE)
    html = render.case_detail("c1", scenarios, generations, roles, by_qid)
    assert "near-miss" in html
    assert render.case_label("c1", scenarios, generations).startswith("c1")


def test_only_fooled_filter_excludes_correct_cases():
    scenarios, _, roles, _, by_qid = render.load(SAMPLE)
    fooled = render.case_options(scenarios, roles, by_qid, only_fooled=True)
    assert "c1" in fooled and "c5" not in fooled


def test_missing_directory_returns_empty_structures(tmp_path):
    scenarios, generations, roles, predictions, by_qid = render.load(str(tmp_path / "absent"))
    assert scenarios == {} and predictions == [] and by_qid == {}


def test_partial_run_renders_without_crashing(tmp_path):
    import shutil

    shutil.copy(Path(SAMPLE) / "scenarios.jsonl", tmp_path / "scenarios.jsonl")
    scenarios, generations, roles, predictions, by_qid = render.load(str(tmp_path))
    assert scenarios and not roles and not predictions
    assert render.scoring_reports(roles, predictions) == []
    qid = next(iter(scenarios))
    assert render.case_label(qid, scenarios, generations).startswith(qid)
    render.case_detail(qid, scenarios, generations, roles, by_qid)   # must not raise on partial data
