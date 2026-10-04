"""Tests for the room assignment layer.

The central correctness claims are here: everybody gets a bed, the optimal
strategy really is optimal, and constraint violations are avoided when avoidable
and minimised when not.
"""

from __future__ import annotations

import pytest

from roommate_matcher.data.generator import generate_students
from roommate_matcher.domain.enums import SleepSchedule, SmokingHabit
from roommate_matcher.domain.models import StudentProfile
from roommate_matcher.domain.preferences import Constraints, MatchPreferences
from roommate_matcher.exceptions import UnknownStrategyError
from roommate_matcher.matching.assignment import (
    VIOLATION_PENALTY,
    GreedyAssignment,
    OptimalAssignment,
    RoomAssigner,
    available_strategies,
    build_preference_lists,
    edge_weight,
    find_blocking_pairs,
    get_strategy,
    optimality_gap,
    stable_roommates,
)
from roommate_matcher.matching.engine import CompatibilityEngine
from tests.conftest import make_student


class TestStrategyRegistry:
    """Strategies are discovered the same way metrics are."""

    def test_every_expected_strategy_is_registered(self) -> None:
        assert set(available_strategies()) == {"optimal", "greedy", "random", "stable"}

    def test_only_the_matching_strategy_claims_optimality(self) -> None:
        optimal = [key for key in available_strategies() if get_strategy(key).optimal]
        assert optimal == ["optimal"]

    def test_unknown_strategy_lists_the_valid_options(self) -> None:
        with pytest.raises(UnknownStrategyError) as info:
            get_strategy("telepati")
        assert "optimal" in str(info.value)


class TestEveryoneIsPlaced:
    """A dormitory cannot leave a student without a bed."""

    @pytest.mark.parametrize("key", available_strategies())
    def test_even_population_leaves_nobody_unplaced(self, key: str) -> None:
        students = generate_students(20, seed=11)
        plan = RoomAssigner(key).assign(students, explain=False)
        assert plan.unplaced == ()
        assert plan.room_count == 10

    @pytest.mark.parametrize("key", available_strategies())
    def test_odd_population_leaves_exactly_one_unplaced(self, key: str) -> None:
        students = generate_students(21, seed=12)
        plan = RoomAssigner(key).assign(students, explain=False)
        assert len(plan.unplaced) == 1
        assert plan.room_count == 10

    @pytest.mark.parametrize("key", available_strategies())
    def test_each_student_appears_in_at_most_one_room(self, key: str) -> None:
        students = generate_students(18, seed=13)
        plan = RoomAssigner(key).assign(students, explain=False)
        seen = [sid for room in plan.rooms for sid in room.occupants]
        assert len(seen) == len(set(seen))

    def test_two_students_form_one_room(self) -> None:
        plan = RoomAssigner("optimal").assign([make_student(1), make_student(2)])
        assert plan.room_count == 1
        assert plan.unplaced == ()

    def test_single_student_cannot_be_placed(self) -> None:
        plan = RoomAssigner("optimal").assign([make_student(1)])
        assert plan.room_count == 0
        assert plan.unplaced == (1,)

    def test_empty_population_is_handled(self) -> None:
        plan = RoomAssigner("optimal").assign([])
        assert plan.room_count == 0
        assert plan.unplaced == ()


class TestOptimality:
    """The optimal strategy must not be beaten by any other."""

    @pytest.fixture(scope="class")
    @staticmethod
    def plans() -> dict[str, object]:
        """One plan per strategy over a shared population."""
        students = generate_students(30, seed=21)
        engine = CompatibilityEngine()
        matrix = engine.score_matrix(students)
        return {
            key: RoomAssigner(key, engine).assign(students, explain=False, matrix=matrix)
            for key in available_strategies()
        }

    @pytest.mark.parametrize("key", ["greedy", "random", "stable"])
    def test_optimal_is_at_least_as_good_as_every_alternative(self, plans, key: str) -> None:
        assert plans["optimal"].total_score >= plans[key].total_score - 1e-9

    def test_optimal_beats_random_clearly(self, plans) -> None:
        assert plans["optimal"].total_score > plans["random"].total_score

    def test_gap_is_zero_against_itself(self, plans) -> None:
        assert optimality_gap(plans["optimal"], plans["optimal"]) == 0.0

    def test_gap_is_positive_for_a_worse_plan(self, plans) -> None:
        assert optimality_gap(plans["random"], plans["optimal"]) > 0

    def test_hand_checked_example_is_solved_exactly(self) -> None:
        """Four students where the best total pairing differs from the greedy one.

        A-B is the single best pair, but taking it strands C with D. Pairing A-C and
        B-D scores less on the best pair and more in total, which is what the
        objective asks for.
        """
        a = make_student(1, cleanliness=5, sleep_schedule=SleepSchedule.EARLY)
        b = make_student(2, cleanliness=5, sleep_schedule=SleepSchedule.EARLY)
        c = make_student(3, cleanliness=4, sleep_schedule=SleepSchedule.EARLY)
        d = make_student(4, cleanliness=1, sleep_schedule=SleepSchedule.LATE)
        students = [a, b, c, d]

        engine = CompatibilityEngine()
        matrix = engine.score_matrix(students)
        optimal = OptimalAssignment().pair_up(matrix)
        total = sum(matrix.score_between(i, j) for i, j in optimal)

        # Enumerate all three perfect matchings of four students and confirm the
        # solver found the best one.
        candidates = [
            [(0, 1), (2, 3)],
            [(0, 2), (1, 3)],
            [(0, 3), (1, 2)],
        ]
        best = max(sum(matrix.score_between(i, j) for i, j in pairs) for pairs in candidates)
        assert total == pytest.approx(best)


class TestConstraintPenalties:
    """Violations are avoided when avoidable and minimised when not."""

    @staticmethod
    def _smokers(*habits: SmokingHabit) -> list[StudentProfile]:
        """Build students that differ only in smoking habit."""
        return [make_student(i, smoking=habit) for i, habit in enumerate(habits, start=1)]

    def test_avoidable_violation_is_avoided(self) -> None:
        students = self._smokers(
            SmokingHabit.INDOOR,
            SmokingHabit.INDOOR,
            SmokingHabit.NONE,
            SmokingHabit.NONE,
        )
        plan = RoomAssigner("optimal").assign(students, explain=False)
        assert plan.violations == {}
        assert plan.unplaced == ()

    def test_unavoidable_violation_still_places_everybody(self) -> None:
        students = self._smokers(
            SmokingHabit.INDOOR,
            SmokingHabit.NONE,
            SmokingHabit.NONE,
            SmokingHabit.NONE,
        )
        plan = RoomAssigner("optimal").assign(students, explain=False)
        assert plan.unplaced == ()
        assert plan.room_count == 2

    def test_unavoidable_violation_is_minimised(self) -> None:
        students = self._smokers(
            SmokingHabit.INDOOR,
            SmokingHabit.NONE,
            SmokingHabit.NONE,
            SmokingHabit.NONE,
        )
        plan = RoomAssigner("optimal").assign(students, explain=False)
        assert plan.violation_count == 1

    def test_penalty_dominates_any_compatibility_score(self) -> None:
        """A violation must outweigh a perfect score, or it could be traded away."""
        clean = make_student(1, smoking=SmokingHabit.NONE)
        smoker = make_student(2, smoking=SmokingHabit.INDOOR)
        engine = CompatibilityEngine()
        matrix = engine.score_matrix([clean, smoker])
        assert edge_weight(matrix, 0, 1) <= 1.0 - VIOLATION_PENALTY

    def test_permissive_constraints_flag_nothing(self) -> None:
        students = self._smokers(SmokingHabit.INDOOR, SmokingHabit.NONE)
        preferences = MatchPreferences(constraints=Constraints.permissive())
        plan = RoomAssigner("optimal", CompatibilityEngine(preferences=preferences)).assign(
            students, explain=False
        )
        assert plan.violations == {}


class TestStabilityDiagnostics:
    """Stability is reported, not optimised."""

    def test_blocking_pair_detector_finds_a_planted_one(self) -> None:
        prefs = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}
        assert (0, 1) in find_blocking_pairs([(0, 2), (1, 3)], prefs)

    def test_detector_accepts_a_correct_assignment(self) -> None:
        prefs = {0: [1, 2, 3], 1: [0, 2, 3], 2: [3, 0, 1], 3: [2, 0, 1]}
        assert find_blocking_pairs([(0, 1), (2, 3)], prefs) == []

    def test_stable_strategy_reports_no_blocking_pairs(self) -> None:
        students = generate_students(12, seed=31)
        plan = RoomAssigner("stable").assign(students, explain=False)
        assert plan.blocking_pairs == ()
        assert plan.is_stable

    def test_irving_returns_a_flag_rather_than_raising(self) -> None:
        pairs, ok = stable_roommates({0: [1], 1: [0]})
        assert ok
        assert pairs == [(0, 1)]

    def test_preference_lists_cover_everyone_else(self) -> None:
        students = generate_students(8, seed=32)
        matrix = CompatibilityEngine().score_matrix(students)
        prefs = build_preference_lists(matrix)
        assert len(prefs) == 8
        assert all(len(lst) == 7 for lst in prefs.values())


class TestPlanReporting:
    """A plan carries the figures staff act on."""

    @pytest.fixture(scope="class")
    @staticmethod
    def plan() -> object:
        """An explained plan over a small population."""
        return RoomAssigner("optimal").assign(generate_students(16, seed=41))

    def test_totals_are_consistent(self, plan) -> None:
        assert plan.total_score == pytest.approx(sum(r.score for r in plan.rooms))
        assert plan.mean_score == pytest.approx(plan.total_score / plan.room_count)

    def test_min_score_is_the_worst_room(self, plan) -> None:
        assert plan.min_score == pytest.approx(min(r.score for r in plan.rooms))

    def test_rooms_are_numbered_from_one_and_sorted_by_score(self, plan) -> None:
        assert [r.room_no for r in plan.rooms] == list(range(1, plan.room_count + 1))
        scores = [r.score for r in plan.rooms]
        assert scores == sorted(scores, reverse=True)

    def test_room_lookup_finds_a_placed_student(self, plan) -> None:
        first = plan.rooms[0].pair.first_id
        assert plan.room_of(first) is plan.rooms[0]

    def test_room_lookup_returns_none_for_an_unknown_student(self, plan) -> None:
        assert plan.room_of(999_999) is None

    def test_worst_rooms_are_the_lowest_scoring(self, plan) -> None:
        worst = plan.worst_rooms(3)
        assert [r.score for r in worst] == sorted(r.score for r in plan.rooms)[:3]

    def test_explained_rooms_carry_contributions(self, plan) -> None:
        assert all(room.pair.contributions for room in plan.rooms)

    def test_unexplained_rooms_skip_contributions(self) -> None:
        plan = RoomAssigner("optimal").assign(generate_students(8, seed=42), explain=False)
        assert all(not room.pair.contributions for room in plan.rooms)

    def test_timing_is_recorded(self, plan) -> None:
        assert plan.seconds >= 0.0


class TestGreedyBaseline:
    """The greedy baseline behaves as documented."""

    def test_greedy_takes_the_best_pair_first(self) -> None:
        a = make_student(1, cleanliness=5)
        b = make_student(2, cleanliness=5)
        c = make_student(3, cleanliness=1, sleep_schedule=SleepSchedule.LATE)
        d = make_student(4, cleanliness=2, sleep_schedule=SleepSchedule.LATE)
        matrix = CompatibilityEngine().score_matrix([a, b, c, d])
        pairs = GreedyAssignment().pair_up(matrix)
        assert (0, 1) in pairs
