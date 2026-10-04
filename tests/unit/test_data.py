"""Tests for the domain models, serialisation, repositories and the generator.

Also covers the data-quality invariants: whole-token interest matching, id lookup
that does not assume row order, column scaling and range validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roommate_matcher.data.generator import generate_students
from roommate_matcher.data.repository import (
    CsvStudentRepository,
    InMemoryStudentRepository,
    write_students,
)
from roommate_matcher.data.serialization import COLUMNS, from_record, to_record
from roommate_matcher.domain.enums import (
    Allergy,
    Interest,
    Language,
    SleepSchedule,
    StudyLocation,
)
from roommate_matcher.domain.models import StudentProfile
from roommate_matcher.exceptions import DuplicateStudentError, StudentNotFoundError
from roommate_matcher.features.interests import parse_interests, serialise_interests
from roommate_matcher.features.vectorizer import vectorize
from tests.conftest import make_student


class TestValidation:
    """Field ranges and required values are enforced on construction."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("age", 12),
            ("age", 150),
            ("cleanliness", 0),
            ("cleanliness", 6),
            ("noise_tolerance", 9),
            ("room_time_weekdays", -1),
            ("room_time_weekdays", 8),
            ("study_year", 0),
            ("study_year", 7),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field: str, value: int) -> None:
        with pytest.raises(ValueError, match=field):
            make_student(1, **{field: value})

    def test_student_id_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="student_id"):
            make_student(0)

    def test_blank_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="display_name"):
            make_student(1, display_name="")

    def test_blank_student_no_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="student_no"):
            make_student(1, student_no="")

    def test_whitespace_is_trimmed(self) -> None:
        assert make_student(1, display_name="  Ayşe  ").display_name == "Ayşe"

    def test_record_is_immutable(self, student: StudentProfile) -> None:
        with pytest.raises(ValueError, match="frozen"):
            student.cleanliness = 1  # type: ignore[misc]


class TestComputedFields:
    """Engineered features behave as documented."""

    def test_quiet_index_is_bounded(self) -> None:
        for cleanliness in range(1, 6):
            for noise in range(1, 6):
                record = make_student(1, cleanliness=cleanliness, noise_tolerance=noise)
                assert 0.0 <= record.quiet_index <= 1.0

    def test_tidy_quiet_homebody_scores_highest(self) -> None:
        calm = make_student(1, cleanliness=5, noise_tolerance=1, room_time_weekdays=5)
        chaotic = make_student(2, cleanliness=1, noise_tolerance=5, room_time_weekdays=0)
        assert calm.quiet_index == 1.0
        assert chaotic.quiet_index == 0.0

    def test_interest_diversity_is_a_fraction(self) -> None:
        assert make_student(1, interests=frozenset(Interest)).interest_diversity == 1.0
        assert make_student(2, interests=frozenset()).interest_diversity == 0.0

    def test_studies_in_room_excludes_library_users(self) -> None:
        assert make_student(1, study_location=StudyLocation.ROOM).studies_in_room
        assert make_student(2, study_location=StudyLocation.MIXED).studies_in_room
        assert not make_student(3, study_location=StudyLocation.LIBRARY).studies_in_room


class TestInterestParsing:
    """Interests are matched as whole tokens, never as substrings."""

    def test_book_does_not_leak_into_books(self) -> None:
        assert parse_interests("gaming,books") == {Interest.GAMING, Interest.BOOKS}

    def test_singular_spelling_normalises_to_the_canonical_member(self) -> None:
        assert parse_interests("book,cooking") == {Interest.BOOKS, Interest.COOKING}

    def test_whitespace_and_case_are_handled(self) -> None:
        assert parse_interests(" Books , MUSIC ,  ") == {Interest.BOOKS, Interest.MUSIC}

    def test_unknown_tokens_are_dropped(self) -> None:
        assert parse_interests("books,quidditch") == {Interest.BOOKS}

    def test_none_and_empty_are_safe(self) -> None:
        assert parse_interests(None) == frozenset()
        assert parse_interests("") == frozenset()

    def test_serialisation_is_stable(self) -> None:
        assert serialise_interests({Interest.MUSIC, Interest.BOOKS}) == "books,music"


class TestVectorSpace:
    """Row position is never assumed to match the student id, and columns are scaled."""

    def test_index_is_resolved_by_id_not_position(self) -> None:
        space = vectorize([make_student(10), make_student(25), make_student(3)])
        assert space.index_of(25) == 1
        assert space.index_of(3) == 2

    def test_missing_id_raises_instead_of_returning_a_neighbour(self) -> None:
        with pytest.raises(KeyError):
            vectorize([make_student(10)]).index_of(9)

    def test_every_encoded_column_is_within_unit_range(self) -> None:
        space = vectorize(generate_students(60, seed=5))
        assert space.matrix.min() >= 0.0
        assert space.matrix.max() <= 1.0

    def test_column_count_matches_declared_names(self) -> None:
        space = vectorize([make_student(1), make_student(2)])
        assert space.matrix.shape[1] == len(space.feature_names)

    def test_empty_input_yields_an_empty_matrix(self) -> None:
        space = vectorize([])
        assert space.n_students == 0


class TestSerialization:
    """Flat records round-trip without loss."""

    def test_round_trip_preserves_every_field(self) -> None:
        original = make_student(
            42,
            interests=frozenset({Interest.BOOKS, Interest.TRAVEL}),
            allergies=frozenset({Allergy.DUST}),
            languages=frozenset({Language.TR, Language.EN}),
            bio="Merhaba",
        )
        assert from_record(to_record(original)) == original

    def test_record_keys_match_the_declared_columns(self) -> None:
        assert set(to_record(make_student(1))) == set(COLUMNS)

    def test_empty_collections_survive_the_round_trip(self) -> None:
        original = make_student(1, interests=frozenset(), allergies=frozenset())
        restored = from_record(to_record(original))
        assert restored.interests == frozenset()
        assert restored.allergies == frozenset()


class TestRepository:
    """Repository behaviour that both backends share."""

    def test_get_missing_student_raises(self) -> None:
        with pytest.raises(StudentNotFoundError, match=r"404|not found"):
            InMemoryStudentRepository().get(404)

    def test_duplicate_insert_raises(self) -> None:
        repository = InMemoryStudentRepository([make_student(1)])
        with pytest.raises(DuplicateStudentError):
            repository.add(make_student(1))

    def test_next_id_starts_at_one_and_increments(self) -> None:
        repository = InMemoryStudentRepository()
        assert repository.next_id() == 1
        repository.add(make_student(7))
        assert repository.next_id() == 8

    def test_others_excludes_the_given_student(self) -> None:
        repository = InMemoryStudentRepository([make_student(1), make_student(2)])
        assert [s.student_id for s in repository.others(1)] == [2]

    def test_count_matches_length(self) -> None:
        repository = InMemoryStudentRepository([make_student(1), make_student(2)])
        assert repository.count() == len(repository) == 2

    def test_csv_repository_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "students.csv"
        repository = CsvStudentRepository(path)
        repository.add(make_student(1, display_name="Ayşe"))
        repository.add(make_student(2, display_name="Mehmet"))

        reloaded = CsvStudentRepository(path)
        assert len(reloaded.list_all()) == 2
        assert reloaded.get(1).display_name == "Ayşe"

    def test_csv_repository_skips_malformed_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "students.csv"
        write_students(path, [make_student(1)])
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write("bozuk,satır,burada\n")
        assert len(CsvStudentRepository(path).list_all()) == 1


class TestGenerator:
    """The synthetic population is valid, reproducible and correlated."""

    def test_generation_is_reproducible(self) -> None:
        assert generate_students(30, seed=7) == generate_students(30, seed=7)

    def test_different_seeds_produce_different_students(self) -> None:
        assert generate_students(30, seed=1) != generate_students(30, seed=2)

    def test_ids_are_unique_and_sequential(self) -> None:
        students = generate_students(50, seed=3, start_id=100)
        assert [s.student_id for s in students] == list(range(100, 150))

    def test_student_numbers_are_unique(self) -> None:
        students = generate_students(200, seed=4)
        assert len({s.student_no for s in students}) == len(students)

    def test_night_owls_tolerate_more_noise_on_average(self) -> None:
        """Evaluation needs a population with structure, so correlations are pinned."""
        students = generate_students(800, seed=11)
        late = [s.noise_tolerance for s in students if s.sleep_schedule is SleepSchedule.LATE]
        early = [s.noise_tolerance for s in students if s.sleep_schedule is SleepSchedule.EARLY]
        assert sum(late) / len(late) > sum(early) / len(early)

    def test_room_studiers_spend_more_evenings_in_the_room(self) -> None:
        students = generate_students(800, seed=12)
        in_room = [s.room_time_weekdays for s in students if s.study_location is StudyLocation.ROOM]
        library = [
            s.room_time_weekdays for s in students if s.study_location is StudyLocation.LIBRARY
        ]
        assert sum(in_room) / len(in_room) > sum(library) / len(library)

    def test_night_owls_study_later(self) -> None:
        from roommate_matcher.domain.enums import StudyTime

        students = generate_students(800, seed=13)
        late_night = sum(
            1
            for s in students
            if s.sleep_schedule is SleepSchedule.LATE and s.study_time is StudyTime.NIGHT
        )
        early_night = sum(
            1
            for s in students
            if s.sleep_schedule is SleepSchedule.EARLY and s.study_time is StudyTime.NIGHT
        )
        assert late_night > early_night
