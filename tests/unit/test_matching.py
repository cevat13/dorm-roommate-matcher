"""Tests for the dimension scorers, metrics, constraints and candidate ranking."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from roommate_matcher.domain.enums import (
    Allergy,
    Department,
    Interest,
    MatchVerdict,
    SleepSchedule,
    SmokingHabit,
    StudyLocation,
    StudyTime,
)
from roommate_matcher.domain.preferences import DEFAULT_WEIGHTS, Constraints, MatchPreferences
from roommate_matcher.exceptions import EmptyCandidatePoolError, UnknownMetricError
from roommate_matcher.matching.constraints import (
    apply_constraints,
    describe_violations,
    is_acceptable,
    violations_for,
)
from roommate_matcher.matching.dimensions import (
    DIMENSIONS,
    score_cleanliness,
    score_dimensions,
    score_sleep,
    score_smoking,
    score_study_location,
    score_study_time,
)
from roommate_matcher.matching.engine import CompatibilityEngine
from roommate_matcher.matching.metrics import available_metrics, get_metric
from tests.conftest import make_student


class TestDimensionCatalogue:
    """The dimension set matches the configured weights exactly."""

    def test_every_dimension_has_a_weight(self) -> None:
        assert {dim.key for dim in DIMENSIONS} == set(DEFAULT_WEIGHTS)

    def test_the_catalogue_is_the_documented_fifteen(self) -> None:
        """A dimension added without a weight, or vice versa, is a silent bug."""
        assert len(DIMENSIONS) == 15
        assert len({dim.key for dim in DIMENSIONS}) == 15

    def test_dormitory_specific_dimensions_are_present(self) -> None:
        keys = {dim.key for dim in DIMENSIONS}
        assert {"study_location", "study_time", "department", "study_year", "room_time"} <= keys

    def test_every_dimension_has_display_strings(self) -> None:
        for dim in DIMENSIONS:
            assert dim.label
            assert dim.positive
            assert dim.negative


class TestDimensionScores:
    """Each dimension must produce a bounded, symmetric compatibility score."""

    @pytest.mark.parametrize("dimension", DIMENSIONS, ids=lambda d: d.key)
    def test_score_is_bounded(self, dimension) -> None:
        a = make_student(1, cleanliness=1, smoking=SmokingHabit.INDOOR, study_year=1)
        b = make_student(2, cleanliness=5, smoking=SmokingHabit.NONE, study_year=6)
        assert 0.0 <= dimension.score(a, b) <= 1.0

    @pytest.mark.parametrize("dimension", DIMENSIONS, ids=lambda d: d.key)
    def test_score_is_symmetric(self, dimension) -> None:
        a = make_student(1, cleanliness=2, sleep_schedule=SleepSchedule.LATE, study_year=1)
        b = make_student(2, cleanliness=5, sleep_schedule=SleepSchedule.EARLY, study_year=5)
        assert dimension.score(a, b) == pytest.approx(dimension.score(b, a))

    def test_early_and_late_sleepers_clash(self) -> None:
        early = make_student(1, sleep_schedule=SleepSchedule.EARLY)
        late = make_student(2, sleep_schedule=SleepSchedule.LATE)
        assert score_sleep(early, late) < 0.2

    def test_flexible_sleeper_suits_everyone(self) -> None:
        flexible = make_student(1, sleep_schedule=SleepSchedule.FLEXIBLE)
        for schedule in SleepSchedule:
            assert score_sleep(flexible, make_student(2, sleep_schedule=schedule)) >= 0.8

    def test_non_smoker_and_indoor_smoker_score_zero(self) -> None:
        clean = make_student(1, smoking=SmokingHabit.NONE)
        smoker = make_student(2, smoking=SmokingHabit.INDOOR)
        assert score_smoking(clean, smoker) == 0.0

    def test_two_messy_students_are_penalised(self) -> None:
        """Agreeing on a low standard is not the same as being compatible."""
        messy = (make_student(1, cleanliness=1), make_student(2, cleanliness=1))
        tidy = (make_student(3, cleanliness=5), make_student(4, cleanliness=5))
        assert score_cleanliness(*messy) < score_cleanliness(*tidy)

    def test_both_studying_in_the_room_is_worse_than_one_in_the_library(self) -> None:
        """Two room-studiers need the same desk quiet at the same time."""
        room_a = make_student(1, study_location=StudyLocation.ROOM)
        room_b = make_student(2, study_location=StudyLocation.ROOM)
        library = make_student(3, study_location=StudyLocation.LIBRARY)
        assert score_study_location(room_a, room_b) < score_study_location(room_a, library)

    def test_room_and_library_is_the_ideal_arrangement(self) -> None:
        room = make_student(1, study_location=StudyLocation.ROOM)
        library = make_student(2, study_location=StudyLocation.LIBRARY)
        assert score_study_location(room, library) == 1.0

    def test_shared_study_hours_score_highest(self) -> None:
        night_a = make_student(1, study_time=StudyTime.NIGHT)
        night_b = make_student(2, study_time=StudyTime.NIGHT)
        day = make_student(3, study_time=StudyTime.DAYTIME)
        assert score_study_time(night_a, night_b) > score_study_time(night_a, day)

    def test_department_policy_comes_from_preferences(self) -> None:
        same_a = make_student(1, department=Department.LAW)
        same_b = make_student(2, department=Department.LAW)
        mixing = MatchPreferences(same_department_bonus=0.0)
        rewarding = MatchPreferences(same_department_bonus=1.0)

        def department_score(preferences: MatchPreferences) -> float:
            parts = score_dimensions(same_a, same_b, preferences)
            return next(c.similarity for c in parts if c.feature == "department")

        assert department_score(rewarding) == 1.0
        assert department_score(mixing) == 0.0


class TestMetrics:
    """Every registered metric must honour the same contract."""

    @pytest.mark.parametrize("key", available_metrics())
    def test_score_is_bounded(self, key: str) -> None:
        metric = get_metric(key)
        a, b = make_student(1), make_student(2, cleanliness=1, study_year=6)
        assert 0.0 <= metric.score(a, b, MatchPreferences()) <= 1.0

    @pytest.mark.parametrize("key", available_metrics())
    def test_score_is_symmetric(self, key: str) -> None:
        metric = get_metric(key)
        a, b = make_student(1), make_student(2, cleanliness=2, study_year=5)
        preferences = MatchPreferences()
        assert metric.score(a, b, preferences) == pytest.approx(metric.score(b, a, preferences))

    @pytest.mark.parametrize("key", available_metrics())
    def test_score_many_matches_score_one(self, key: str) -> None:
        metric = get_metric(key)
        student = make_student(1)
        pool = [make_student(i, cleanliness=(i % 5) + 1) for i in range(2, 8)]
        preferences = MatchPreferences()
        batch = metric.score_many(student, pool, preferences)
        for candidate, batched in zip(pool, batch, strict=True):
            assert batched == pytest.approx(metric.score(student, candidate, preferences), abs=1e-9)

    def test_identical_students_score_one_when_no_dimension_penalises_sameness(self) -> None:
        """Two dimensions penalise being alike, so neutralise them first."""
        metric = get_metric("weighted_gower")
        kwargs = {"study_location": StudyLocation.LIBRARY, "cleanliness": 5}
        first = make_student(1, **kwargs)
        second = make_student(2, **kwargs)
        assert metric.score(first, second, MatchPreferences()) == pytest.approx(1.0)

    def test_identical_room_studiers_do_not_score_one(self) -> None:
        """Sameness is not rewarded where it causes friction."""
        metric = get_metric("weighted_gower")
        first = make_student(1, study_location=StudyLocation.ROOM)
        second = make_student(2, study_location=StudyLocation.ROOM)
        assert metric.score(first, second, MatchPreferences()) < 1.0

    def test_unknown_metric_lists_the_valid_options(self) -> None:
        with pytest.raises(UnknownMetricError) as info:
            get_metric("kosinus")
        assert "weighted_gower" in str(info.value)

    def test_empty_pool_returns_empty_array(self) -> None:
        metric = get_metric("euclidean")
        assert len(metric.score_many(make_student(1), [], MatchPreferences())) == 0


class TestConstraints:
    """Constraints identify violations; they do not score."""

    def test_self_pairing_is_a_violation(self) -> None:
        student = make_student(1)
        assert violations_for(student, student, Constraints()) == ("self",)

    def test_smoker_mix_is_flagged(self) -> None:
        clean = make_student(1, smoking=SmokingHabit.NONE)
        smoker = make_student(2, smoking=SmokingHabit.INDOOR)
        assert "smoker_mix" in violations_for(clean, smoker, Constraints())

    def test_outdoor_smoker_is_not_a_mix_violation(self) -> None:
        clean = make_student(1, smoking=SmokingHabit.NONE)
        outdoor = make_student(2, smoking=SmokingHabit.OUTDOOR_ONLY)
        assert "smoker_mix" not in violations_for(clean, outdoor, Constraints())

    def test_smoke_allergy_rejects_any_smoker(self) -> None:
        allergic = make_student(1, allergies=frozenset({Allergy.SMOKE}))
        outdoor = make_student(2, smoking=SmokingHabit.OUTDOOR_ONLY)
        assert "smoke_allergy" in violations_for(allergic, outdoor, Constraints())

    def test_year_gap_is_enforced_when_configured(self) -> None:
        first = make_student(1, study_year=1)
        senior = make_student(2, study_year=5)
        assert violations_for(first, senior, Constraints()) == ()
        assert "year_gap" in violations_for(first, senior, Constraints(max_year_gap=2))

    def test_max_smoking_caps_severity(self) -> None:
        clean = make_student(1)
        outdoor = make_student(2, smoking=SmokingHabit.OUTDOOR_ONLY)
        constraints = Constraints(max_smoking=SmokingHabit.NONE)
        assert "max_smoking" in violations_for(clean, outdoor, constraints)

    def test_permissive_constraints_flag_nothing_but_self(self) -> None:
        clean = make_student(1, smoking=SmokingHabit.NONE)
        smoker = make_student(2, smoking=SmokingHabit.INDOOR)
        assert violations_for(clean, smoker, Constraints.permissive()) == ()

    def test_is_acceptable_mirrors_violations(self) -> None:
        clean = make_student(1, smoking=SmokingHabit.NONE)
        smoker = make_student(2, smoking=SmokingHabit.INDOOR)
        assert not is_acceptable(clean, smoker, Constraints())
        assert is_acceptable(clean, make_student(3), Constraints())

    def test_filtering_tallies_rejections(self) -> None:
        student = make_student(1, smoking=SmokingHabit.NONE)
        pool = [student, make_student(2, smoking=SmokingHabit.INDOOR), make_student(3)]
        outcome = apply_constraints(student, pool, Constraints())
        assert outcome.kept == 1
        assert outcome.rejections == {"self": 1, "smoker_mix": 1}

    def test_descriptions_are_translated_and_sorted(self) -> None:
        text = describe_violations({"smoker_mix": 10, "year_gap": 2})
        assert text.index("sigara") < text.index("sınıf")

    def test_empty_tally_is_empty_text(self) -> None:
        assert describe_violations({}) == ""


class TestCandidateRanking:
    """Candidate analysis still works, now as a staff tool."""

    def test_results_are_sorted_descending(self, population, engine) -> None:
        report = engine.candidates_for(population[0], population[1:], top_n=10)
        scores = [pair.score for pair in report.results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_is_respected(self, population, engine) -> None:
        assert len(engine.candidates_for(population[0], population[1:], top_n=3).results) <= 3

    def test_student_is_never_their_own_candidate(self, population, engine) -> None:
        student = population[0]
        report = engine.candidates_for(student, population, top_n=20)
        assert all(pair.second_id != student.student_id for pair in report.results)

    def test_counts_add_up(self, population, engine) -> None:
        report = engine.candidates_for(population[0], population[1:], top_n=5)
        assert 0 <= report.filtered_out <= report.considered

    def test_explanations_cover_every_weighted_dimension(self, population, engine) -> None:
        report = engine.candidates_for(population[0], population[1:], top_n=1)
        weighted = sum(1 for value in MatchPreferences().weights.values() if value > 0)
        assert len(report.results[0].contributions) == weighted

    def test_explain_false_skips_contributions(self, population, engine) -> None:
        report = engine.candidates_for(population[0], population[1:], top_n=3, explain=False)
        assert all(not pair.contributions for pair in report.results)

    def test_empty_pool_returns_empty_report(self, engine) -> None:
        report = engine.candidates_for(make_student(1), [], top_n=5)
        assert report.results == ()
        assert report.best is None

    def test_empty_pool_can_raise_with_reasons(self) -> None:
        clean = make_student(1, smoking=SmokingHabit.NONE)
        smoker = make_student(2, smoking=SmokingHabit.INDOOR)
        engine = CompatibilityEngine()
        with pytest.raises(EmptyCandidatePoolError) as info:
            engine.candidates_for(clean, [smoker], raise_on_empty=True)
        assert "smoker_mix" in str(info.value)


class TestScoreMatrix:
    """The matrix is symmetric and indexed by id."""

    def test_matrix_is_symmetric_with_unit_diagonal(self, population, engine) -> None:
        matrix = engine.score_matrix(population[:8])
        assert matrix.scores.shape == (8, 8)
        assert all(matrix.scores[i, i] == 1.0 for i in range(8))
        assert (abs(matrix.scores - matrix.scores.T) < 1e-9).all()

    def test_index_lookup_uses_ids_not_positions(self) -> None:
        matrix = CompatibilityEngine().score_matrix(
            [make_student(10), make_student(25), make_student(3)]
        )
        assert matrix.index_of(25) == 1
        assert matrix.index_of(3) == 2

    def test_missing_id_raises(self) -> None:
        matrix = CompatibilityEngine().score_matrix([make_student(10)])
        with pytest.raises(KeyError):
            matrix.index_of(9)

    def test_violations_can_be_skipped(self, population, engine) -> None:
        matrix = engine.score_matrix(population[:5], with_violations=False)
        assert matrix.violations == ()
        assert matrix.violations_between(0, 1) == ()


class TestWeights:
    """Weights change the ranking, and invalid weights are refused."""

    def test_zero_weight_removes_only_that_dimension(self) -> None:
        preferences = MatchPreferences(weights={"smoking": 0.0})
        features = {
            c.feature for c in score_dimensions(make_student(1), make_student(2), preferences)
        }
        assert "smoking" not in features
        assert "cleanliness" in features

    def test_partial_weights_are_completed_from_the_defaults(self) -> None:
        preferences = MatchPreferences(weights={"smoking": 4.0})
        assert preferences.weight_for("smoking") == 4.0
        assert preferences.weight_for("cleanliness") == DEFAULT_WEIGHTS["cleanliness"]

    def test_emphasising_a_dimension_changes_the_winner(self) -> None:
        student = make_student(1, smoking=SmokingHabit.NONE, cleanliness=5)
        tidy_smoker = make_student(2, smoking=SmokingHabit.INDOOR, cleanliness=5)
        messy_clean = make_student(3, smoking=SmokingHabit.NONE, cleanliness=1)

        by_smoking = CompatibilityEngine(
            preferences=MatchPreferences(weights={"smoking": 5.0, "cleanliness": 0.5})
        )
        by_cleanliness = CompatibilityEngine(
            preferences=MatchPreferences(weights={"smoking": 0.5, "cleanliness": 5.0})
        )
        assert by_smoking.score_pair(student, messy_clean) > by_smoking.score_pair(
            student, tidy_smoker
        )
        assert by_cleanliness.score_pair(student, tidy_smoker) > by_cleanliness.score_pair(
            student, messy_clean
        )

    def test_unknown_weight_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown weight keys"):
            MatchPreferences(weights={"telepati": 1.0})

    def test_negative_weight_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            MatchPreferences(weights={"smoking": -1.0})

    def test_all_zero_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than zero"):
            MatchPreferences(weights=dict.fromkeys(DEFAULT_WEIGHTS, 0.0))


class TestProperties:
    """Invariants that must hold for any input."""

    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(
        cleanliness=st.integers(1, 5),
        noise=st.integers(1, 5),
        study_year=st.integers(1, 6),
        room_time=st.integers(0, 5),
        smoking=st.sampled_from(list(SmokingHabit)),
        sleep=st.sampled_from(list(SleepSchedule)),
        location=st.sampled_from(list(StudyLocation)),
    )
    def test_score_is_always_in_unit_interval(
        self, cleanliness, noise, study_year, room_time, smoking, sleep, location
    ) -> None:
        candidate = make_student(
            2,
            cleanliness=cleanliness,
            noise_tolerance=noise,
            study_year=study_year,
            room_time_weekdays=room_time,
            smoking=smoking,
            sleep_schedule=sleep,
            study_location=location,
        )
        assert 0.0 <= CompatibilityEngine().score_pair(make_student(1), candidate) <= 1.0

    @settings(max_examples=40, deadline=None)
    @given(interests=st.sets(st.sampled_from(list(Interest)), max_size=6))
    def test_identical_students_reach_the_attainable_maximum(self, interests) -> None:
        """Whatever the interests, a clone is the best possible roommate for someone.

        The score is not always 1.0, because ``study_location`` and ``cleanliness``
        penalise sameness by design; it must still be unbeatable.
        """
        engine = CompatibilityEngine()
        base = make_student(1, interests=frozenset(interests))
        clone = make_student(2, interests=frozenset(interests))
        different = make_student(
            3,
            interests=frozenset(interests),
            cleanliness=1,
            sleep_schedule=SleepSchedule.LATE,
            smoking=SmokingHabit.INDOOR,
            study_year=6,
        )
        assert engine.score_pair(base, clone) >= engine.score_pair(base, different)

    @settings(max_examples=40, deadline=None)
    @given(cleanliness=st.integers(1, 5), study_year=st.integers(1, 6))
    def test_scoring_is_symmetric(self, cleanliness, study_year) -> None:
        engine = CompatibilityEngine()
        a = make_student(1)
        b = make_student(2, cleanliness=cleanliness, study_year=study_year)
        assert engine.score_pair(a, b) == pytest.approx(engine.score_pair(b, a))


class TestVerdicts:
    """Verdicts bucket consistently with the thresholds."""

    def test_identical_students_match_on_every_dimension(self) -> None:
        parts = score_dimensions(make_student(1), make_student(2), MatchPreferences())
        assert all(part.verdict is MatchVerdict.MATCH for part in parts)

    def test_clashing_students_report_concerns(self) -> None:
        student = make_student(1, smoking=SmokingHabit.NONE, sleep_schedule=SleepSchedule.EARLY)
        opposite = make_student(2, smoking=SmokingHabit.INDOOR, sleep_schedule=SleepSchedule.LATE)
        pair = CompatibilityEngine().explain_pair(student, opposite)
        clashes = {c.feature for c in pair.concerns(5)}
        assert {"smoking", "sleep_schedule"} <= clashes

    def test_violations_appear_in_the_summary(self) -> None:
        student = make_student(1, smoking=SmokingHabit.NONE)
        smoker = make_student(2, smoking=SmokingHabit.INDOOR)
        pair = CompatibilityEngine().explain_pair(student, smoker)
        assert "smoker_mix" in pair.violations
        assert "kısıt ihlali" in pair.summary
