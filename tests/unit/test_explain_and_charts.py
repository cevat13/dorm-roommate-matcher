"""Tests for the explanation text and the chart builders."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from roommate_matcher.domain.enums import MatchVerdict, SleepSchedule, SmokingHabit
from roommate_matcher.matching.assignment import RoomAssigner
from roommate_matcher.matching.engine import CompatibilityEngine
from roommate_matcher.matching.explain import build_summary, compare, impact_table
from roommate_matcher.viz.charts import (
    STATUS_COLOURS,
    contribution_chart,
    contributions_table,
    metric_comparison,
    radar_chart,
    room_score_distribution,
    score_distribution,
    similarity_heatmap,
    strategy_comparison,
)
from tests.conftest import make_student


@pytest.fixture
def good_pair():
    """An explained pairing between two compatible students."""
    return CompatibilityEngine().explain_pair(make_student(1), make_student(2))


@pytest.fixture
def poor_pair():
    """An explained pairing between two deeply incompatible students."""
    first = make_student(1, smoking=SmokingHabit.NONE, sleep_schedule=SleepSchedule.EARLY)
    second = make_student(
        2, smoking=SmokingHabit.INDOOR, sleep_schedule=SleepSchedule.LATE, cleanliness=1
    )
    return CompatibilityEngine().explain_pair(first, second)


class TestSummary:
    """The summary names the factors behind the score, not just the number."""

    def test_summary_contains_the_percentage(self, good_pair) -> None:
        assert f"%{good_pair.percentage}" in good_pair.summary

    def test_good_pair_lists_strengths(self, good_pair) -> None:
        assert "Güçlü yanlar" in good_pair.summary

    def test_poor_pair_lists_concerns(self, poor_pair) -> None:
        assert "Dikkat edilmesi gerekenler" in poor_pair.summary
        assert "Sigara" in poor_pair.summary

    def test_poor_pair_reports_the_violation(self, poor_pair) -> None:
        assert "kısıt ihlali" in poor_pair.summary

    def test_summary_without_contributions_still_reports_the_score(self) -> None:
        assert "%42" in build_summary([], 0.42)

    def test_turkish_dotted_capital_is_not_mangled(self, poor_pair) -> None:
        """Lowercasing a dotted capital in Turkish leaves a combining dot behind."""
        assert "̇" not in poor_pair.summary


class TestImpactTable:
    """Rows are ordered by the points each dimension cost."""

    def test_rows_cover_every_contribution(self, poor_pair) -> None:
        assert len(impact_table(poor_pair.contributions)) == len(poor_pair.contributions)

    def test_worst_dimension_comes_first(self, poor_pair) -> None:
        rows = impact_table(poor_pair.contributions)
        assert rows[0].lost_points >= rows[-1].lost_points

    def test_shares_sum_to_the_score(self, poor_pair) -> None:
        rows = impact_table(poor_pair.contributions)
        assert sum(row.share_of_score for row in rows) == pytest.approx(poor_pair.score, abs=0.01)

    def test_max_share_is_at_least_the_actual_share(self, poor_pair) -> None:
        for row in impact_table(poor_pair.contributions):
            assert row.max_possible_share >= row.share_of_score


class TestCompare:
    """Comparing candidates names the dimensions that separate them."""

    def test_needs_two_candidates(self, good_pair) -> None:
        assert "en az iki" in compare([good_pair])

    def test_names_both_candidates(self, good_pair, poor_pair) -> None:
        text = compare([good_pair, poor_pair])
        assert good_pair.second_name in text
        assert poor_pair.second_name in text


class TestCharts:
    """Chart builders produce figures without needing a display."""

    def test_contribution_chart_has_one_bar_per_dimension(self, poor_pair) -> None:
        figure = contribution_chart(poor_pair)
        assert isinstance(figure, go.Figure)
        assert len(figure.data[0].x) == len(poor_pair.contributions)

    def test_contribution_bars_use_the_status_palette(self, poor_pair) -> None:
        colours = set(contribution_chart(poor_pair).data[0].marker.color)
        assert colours <= set(STATUS_COLOURS.values())

    def test_contribution_bars_carry_visible_labels(self, poor_pair) -> None:
        """Two status steps fall below 3:1 on light, so the labels carry the meaning."""
        figure = contribution_chart(poor_pair)
        assert figure.data[0].text is not None
        assert figure.data[0].textposition == "outside"

    def test_radar_has_two_named_series(self, poor_pair) -> None:
        figure = radar_chart(make_student(1), make_student(2), poor_pair.contributions)
        assert len(figure.data) == 2
        assert figure.layout.showlegend is True

    def test_radar_polygon_is_closed(self, poor_pair) -> None:
        trace = radar_chart(make_student(1), make_student(2), poor_pair.contributions).data[0]
        assert trace.r[0] == trace.r[-1]
        assert trace.theta[0] == trace.theta[-1]

    def test_distribution_marks_the_highlighted_value(self) -> None:
        assert len(score_distribution([0.2, 0.5, 0.8], highlight=0.8).layout.shapes) == 1

    def test_distribution_without_highlight_has_no_line(self) -> None:
        assert len(score_distribution([0.2, 0.5]).layout.shapes) == 0

    def test_room_distribution_marks_the_worst_room(self) -> None:
        from roommate_matcher.data.generator import generate_students

        plan = RoomAssigner("optimal").assign(generate_students(8, seed=51), explain=False)
        figure = room_score_distribution(plan)
        assert len(figure.layout.shapes) == 1

    def test_heatmap_is_square(self) -> None:
        students = [make_student(i) for i in range(1, 6)]
        matrix = CompatibilityEngine().score_matrix(students, with_violations=False)
        figure = similarity_heatmap(matrix.scores, [str(s.student_id) for s in students])
        assert len(figure.data[0].z) == 5

    def test_metric_comparison_sorts_ascending(self) -> None:
        rows: list[dict[str, float | str | int]] = [
            {"metric": "cosine", "nDCG@10": 0.45},
            {"metric": "weighted_gower", "nDCG@10": 0.76},
        ]
        assert list(metric_comparison(rows).data[0].y) == ["cosine", "weighted_gower"]

    def test_strategy_comparison_sorts_ascending(self) -> None:
        rows: list[dict[str, float | str | int]] = [
            {
                "strateji": "random",
                "toplam": 39.9,
                "fark %": 20.4,
                "en kötü oda": 0.43,
                "blocking": 2447,
            },
            {
                "strateji": "optimal",
                "toplam": 50.1,
                "fark %": 0.0,
                "en kötü oda": 0.70,
                "blocking": 25,
            },
        ]
        assert list(strategy_comparison(rows).data[0].y) == ["random", "optimal"]

    def test_table_view_exists_for_every_chart_row(self, poor_pair) -> None:
        frame = contributions_table(poor_pair.contributions)
        assert len(frame) == len(poor_pair.contributions)
        assert "Durum" in frame.columns
        assert set(frame["Durum"]) <= {v.label_tr for v in MatchVerdict}
