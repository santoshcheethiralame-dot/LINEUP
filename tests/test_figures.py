import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "app"))

import render  # noqa: E402
import run_abstention  # noqa: E402
import run_scoring  # noqa: E402
from lineup.scoring import bootstrap_intervals  # noqa: E402

SAMPLE = str(ROOT / "app" / "sample")


def test_scoring_figures_write_non_empty_files(tmp_path):
    _, _, roles, predictions, _ = render.load(SAMPLE)
    reports = render.scoring_reports(roles, predictions)
    wrong = [case for case in roles.values() if not case.original_correct]
    intervals = bootstrap_intervals(wrong, predictions, n_boot=50, seed=0)
    run_scoring._render_figure(reports, tmp_path / "dist.png")
    run_scoring._render_ci_figure(reports, intervals, tmp_path / "ci.png")
    run_scoring._render_confusion_figure(reports, tmp_path / "conf.png")
    for name in ("dist.png", "ci.png", "conf.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_abstention_figure_writes_a_non_empty_file(tmp_path):
    _, generations, roles, predictions, _ = render.load(SAMPLE)
    run_abstention._render_figure(list(generations.values()), predictions, list(roles.values()), tmp_path / "rc.png")
    assert (tmp_path / "rc.png").stat().st_size > 0
