from lineup.data.schema import Chunk, Recipe, Scenario
from lineup.methods import AttributionMethod, run_method
from lineup.stats import percentile_interval


class _AllTie(AttributionMethod):
    name = "all_tie"

    def score_chunks(self, model, scenario, answer):
        return [0.0] * len(scenario.chunks)


def _scenario(qid):
    chunks = [Chunk(f"{qid}::a", "A", "x"), Chunk(f"{qid}::b", "B", "y"), Chunk(f"{qid}::c", "C", "z")]
    recipe = Recipe(0, 3, "", "", "entity", "a", 0, ["a"], "b", ["c"], [f"{qid}::a", f"{qid}::b", f"{qid}::c"])
    return Scenario(qid, "q", "g", chunks, recipe)


def test_tied_scores_do_not_always_blame_the_first_chunk():
    picked = {run_method(_AllTie(), None, _scenario(f"q{i}"), "ans").predicted_culprit_id.split("::")[1] for i in range(20)}
    assert len(picked) > 1


def test_tie_break_is_deterministic_per_case():
    scenario = _scenario("q1")
    first = run_method(_AllTie(), None, scenario, "ans").predicted_culprit_id
    second = run_method(_AllTie(), None, scenario, "ans").predicted_culprit_id
    assert first == second


def test_percentile_interval_nearest_rank_is_inside_the_range():
    low, high = percentile_interval(list(range(100)), alpha=0.05)
    assert low is not None and 0 <= low < high <= 99
    assert percentile_interval([1.0] * 5, alpha=0.05) == (None, None)
