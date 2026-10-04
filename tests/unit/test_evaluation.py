"""Tests for the ranking measures, the oracle and the plan-quality measures."""

from __future__ import annotations

import pytest

from roommate_matcher.data.generator import generate_students
from roommate_matcher.domain.enums import Allergy, SleepSchedule, SmokingHabit, StudyLocation
from roommate_matcher.matching.assignment import RoomAssigner
from roommate_matcher.matching.evaluation import (
    compare_metrics,
    evaluate_metric,
    evaluate_plans,
    mean_reciprocal_rank,
    ndcg_at_k,
    oracle_relevant,
    oracle_share,
    precision_at_k,
    recall_at_k,
    relevant_set,
)
from tests.conftest import make_student


class TestRankingMeasures:
    """Known cases for the information-retrieval measures."""

    def test_precision_counts_hits_in_the_top_k(self) -> None:
        assert precision_at_k([1, 2, 3, 4], {1, 3}, 4) == 0.5
        assert precision_at_k([1, 2, 3, 4], {1, 2}, 2) == 1.0
        assert precision_at_k([5, 6], {1}, 2) == 0.0

    def test_precision_at_zero_is_zero(self) -> None:
        assert precision_at_k([1, 2], {1}, 0) == 0.0

    def test_recall_is_relative_to_the_relevant_set(self) -> None:
        assert recall_at_k([1, 2], {1, 2, 3, 4}, 2) == 0.5
        assert recall_at_k([1], set(), 1) == 0.0

    def test_mrr_uses_the_first_hit(self) -> None:
        assert mean_reciprocal_rank([9, 8, 1], {1}) == pytest.approx(1 / 3)
        assert mean_reciprocal_rank([1, 8, 9], {1}) == 1.0
        assert mean_reciprocal_rank([7, 8], {1}) == 0.0

    def test_ndcg_is_one_for_a_perfect_ranking(self) -> None:
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)

    def test_ndcg_rewards_putting_hits_first(self) -> None:
        assert ndcg_at_k([1, 9, 9], {1}, 3) > ndcg_at_k([9, 9, 1], {1}, 3)

    def test_ndcg_is_zero_without_relevant_items(self) -> None:
        assert ndcg_at_k([1, 2], set(), 2) == 0.0


class TestOracle:
    """The oracle encodes the dormitory rules and nothing from the scorer."""

    def test_a_student_is_not_relevant_to_themselves(self) -> None:
        student = make_student(1)
        assert not oracle_relevant(student, student)

    def test_indoor_smoker_with_non_smoker_is_rejected(self) -> None:
        clean = make_student(1, smoking=SmokingHabit.NONE)
        smoker = make_student(2, smoking=SmokingHabit.INDOOR)
        assert not oracle_relevant(clean, smoker)

    def test_smoke_allergy_rejects_any_smoker(self) -> None:
        allergic = make_student(1, allergies=frozenset({Allergy.SMOKE}))
        outdoor = make_student(2, smoking=SmokingHabit.OUTDOOR_ONLY)
        assert not oracle_relevant(allergic, outdoor)

    def test_large_cleanliness_gap_is_rejected(self) -> None:
        assert not oracle_relevant(make_student(1, cleanliness=5), make_student(2, cleanliness=2))

    def test_opposite_sleep_schedules_are_rejected(self) -> None:
        early = make_student(1, sleep_schedule=SleepSchedule.EARLY)
        late = make_student(2, sleep_schedule=SleepSchedule.LATE)
        assert not oracle_relevant(early, late)

    def test_two_room_studiers_are_rejected(self) -> None:
        a = make_student(1, study_location=StudyLocation.ROOM)
        b = make_student(2, study_location=StudyLocation.ROOM)
        assert not oracle_relevant(a, b)

    def test_large_year_gap_is_rejected(self) -> None:
        assert not oracle_relevant(make_student(1, study_year=1), make_student(2, study_year=6))

    def test_a_compatible_pair_is_accepted(self) -> None:
        a = make_student(1, study_location=StudyLocation.ROOM)
        b = make_student(2, study_location=StudyLocation.LIBRARY)
        assert oracle_relevant(a, b)

    def test_oracle_is_symmetric(self) -> None:
        students = generate_students(40, seed=17)
        for first in students[:10]:
            for second in students[10:20]:
                assert oracle_relevant(first, second) == oracle_relevant(second, first)

    def test_relevant_set_excludes_the_student(self) -> None:
        students = generate_students(30, seed=18)
        assert students[0].student_id not in relevant_set(students[0], students)


class TestMetricComparison:
    """The weighted metric outranks the geometric baselines."""

    @pytest.fixture(scope="class")
    @staticmethod
    def students() -> list:  # type: ignore[type-arg]
        """120 reproducible students, built once per class."""
        return generate_students(120, seed=21)

    def test_weighted_gower_beats_cosine(self, students) -> None:
        gower = evaluate_metric("weighted_gower", students, sample=40)
        cosine = evaluate_metric("cosine", students, sample=40)
        assert gower.ndcg_at_10 > cosine.ndcg_at_10

    def test_weighted_gower_beats_euclidean(self, students) -> None:
        gower = evaluate_metric("weighted_gower", students, sample=40)
        euclidean = evaluate_metric("euclidean", students, sample=40)
        assert gower.ndcg_at_10 > euclidean.ndcg_at_10

    def test_every_score_is_bounded(self, students) -> None:
        for scores in compare_metrics(students, sample=20).scores:
            assert 0.0 <= scores.precision_at_5 <= 1.0
            assert 0.0 <= scores.ndcg_at_10 <= 1.0
            assert 0.0 <= scores.mrr <= 1.0

    def test_report_picks_the_weighted_metric(self, students) -> None:
        report = compare_metrics(students, metrics=["weighted_gower", "cosine"], sample=20)
        assert report.winner is not None
        assert report.winner.metric == "weighted_gower"


class TestPlanQuality:
    """Plans are judged on what they deliver, by the independent oracle too."""

    @pytest.fixture(scope="class")
    @staticmethod
    def students() -> list:  # type: ignore[type-arg]
        """60 reproducible students, built once per class."""
        return generate_students(60, seed=31)

    def test_optimal_has_the_best_total(self, students) -> None:
        scored = evaluate_plans(students)
        assert scored[0].strategy == "optimal"

    def test_optimal_gap_is_zero(self, students) -> None:
        scored = evaluate_plans(students)
        assert scored[0].gap_percent == 0.0

    def test_random_is_clearly_worse(self, students) -> None:
        scored = {row.strategy: row for row in evaluate_plans(students)}
        assert scored["random"].gap_percent > scored["greedy"].gap_percent

    def test_every_strategy_places_everyone(self, students) -> None:
        assert all(row.unplaced == 0 for row in evaluate_plans(students))

    def test_oracle_share_is_a_fraction(self, students) -> None:
        for row in evaluate_plans(students):
            assert 0.0 <= row.oracle_share <= 1.0

    def test_optimal_beats_random_on_the_oracle_too(self, students) -> None:
        """A better total should also mean more rooms the oracle calls liveable."""
        scored = {row.strategy: row for row in evaluate_plans(students)}
        assert scored["optimal"].oracle_share > scored["random"].oracle_share

    def test_rows_are_renderable(self, students) -> None:
        row = evaluate_plans(students)[0].as_row()
        assert {"strateji", "toplam", "en kötü oda", "oracle %"} <= set(row)

    def test_oracle_share_of_an_empty_plan_is_zero(self) -> None:
        plan = RoomAssigner("optimal").assign([make_student(1)])
        assert oracle_share(plan, [make_student(1)]) == 0.0
