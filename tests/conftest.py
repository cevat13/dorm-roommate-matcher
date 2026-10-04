"""Shared fixtures."""

from __future__ import annotations

import pytest

from roommate_matcher.data.generator import generate_students
from roommate_matcher.data.repository import InMemoryStudentRepository
from roommate_matcher.domain.enums import (
    AlcoholHabit,
    Department,
    DietType,
    GuestFrequency,
    Interest,
    Language,
    SleepSchedule,
    SmokingHabit,
    StudyLocation,
    StudyTime,
)
from roommate_matcher.domain.models import StudentProfile
from roommate_matcher.domain.preferences import MatchPreferences
from roommate_matcher.matching.engine import CompatibilityEngine


def make_student(student_id: int = 1, **overrides: object) -> StudentProfile:
    """Build a student, overriding only the fields a test cares about.

    Args:
        student_id: The id to assign.
        **overrides: Any field to override.

    Returns:
        A validated student record.
    """
    defaults: dict[str, object] = {
        "student_id": student_id,
        "student_no": f"2024{student_id:05d}",
        "display_name": f"Öğrenci {student_id}",
        "age": 20,
        "department": Department.COMPUTER_ENG,
        "study_year": 2,
        "languages": frozenset({Language.TR}),
        "study_location": StudyLocation.MIXED,
        "study_time": StudyTime.EVENING,
        "sleep_schedule": SleepSchedule.EARLY,
        "cleanliness": 4,
        "noise_tolerance": 3,
        "social_energy": 3,
        "room_time_weekdays": 3,
        "guest_frequency": GuestFrequency.SOMETIMES,
        "smoking": SmokingHabit.NONE,
        "alcohol": AlcoholHabit.NONE,
        "diet": DietType.OMNIVORE,
        "allergies": frozenset(),
        "interests": frozenset({Interest.BOOKS, Interest.MUSIC}),
        "bio": "",
    }
    defaults.update(overrides)
    return StudentProfile(**defaults)


@pytest.fixture
def student() -> StudentProfile:
    """A single baseline student.

    Returns:
        A validated record with id 1.
    """
    return make_student()


@pytest.fixture
def population() -> list[StudentProfile]:
    """A small reproducible synthetic population.

    Returns:
        120 generated students.
    """
    return generate_students(120, seed=99)


@pytest.fixture
def repository(population: list[StudentProfile]) -> InMemoryStudentRepository:
    """An in-memory repository seeded with the population.

    Args:
        population: The records to load.

    Returns:
        A ready repository.
    """
    return InMemoryStudentRepository(population)


@pytest.fixture
def engine() -> CompatibilityEngine:
    """A default engine with the standard preferences.

    Returns:
        A configured engine.
    """
    return CompatibilityEngine("weighted_gower", MatchPreferences())
