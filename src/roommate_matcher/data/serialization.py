"""Conversion between nested student records and flat rows.

A :class:`StudentProfile` nests set-valued fields, while CSV and SQL rows are flat.
Both directions live here so that every storage backend round-trips identically.
"""

from __future__ import annotations

from typing import Any

from roommate_matcher.domain.enums import (
    AlcoholHabit,
    Allergy,
    Department,
    DietType,
    GuestFrequency,
    Language,
    SleepSchedule,
    SmokingHabit,
    StudyLocation,
    StudyTime,
)
from roommate_matcher.domain.models import StudentProfile
from roommate_matcher.features.interests import parse_interests, serialise_interests

COLUMNS: tuple[str, ...] = (
    "student_id",
    "student_no",
    "display_name",
    "age",
    "department",
    "study_year",
    "languages",
    "study_location",
    "study_time",
    "sleep_schedule",
    "cleanliness",
    "noise_tolerance",
    "social_energy",
    "room_time_weekdays",
    "guest_frequency",
    "smoking",
    "alcohol",
    "diet",
    "allergies",
    "interests",
    "bio",
)
"""Canonical column order, shared by CSV files and database rows."""


def to_record(student: StudentProfile) -> dict[str, Any]:
    """Flatten a student into a plain record.

    Args:
        student: The student to flatten.

    Returns:
        A dict keyed by :data:`COLUMNS`.
    """
    return {
        "student_id": student.student_id,
        "student_no": student.student_no,
        "display_name": student.display_name,
        "age": student.age,
        "department": student.department.value,
        "study_year": student.study_year,
        "languages": ",".join(sorted(lang.value for lang in student.languages)),
        "study_location": student.study_location.value,
        "study_time": student.study_time.value,
        "sleep_schedule": student.sleep_schedule.value,
        "cleanliness": student.cleanliness,
        "noise_tolerance": student.noise_tolerance,
        "social_energy": student.social_energy,
        "room_time_weekdays": student.room_time_weekdays,
        "guest_frequency": student.guest_frequency.value,
        "smoking": student.smoking.value,
        "alcohol": student.alcohol.value,
        "diet": student.diet.value,
        "allergies": ",".join(sorted(a.value for a in student.allergies)),
        "interests": serialise_interests(student.interests),
        "bio": student.bio,
    }


def _split(raw: Any) -> list[str]:
    """Split a delimited cell into tokens, tolerating blanks and NaN.

    Args:
        raw: The raw cell value.

    Returns:
        The non-empty tokens.
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def from_record(record: dict[str, Any]) -> StudentProfile:
    """Rebuild a validated student from a flat record.

    Args:
        record: A record shaped like :data:`COLUMNS`.

    Returns:
        A validated :class:`StudentProfile`.

    Raises:
        pydantic.ValidationError: If the record violates the domain constraints.
    """
    languages = {Language(code) for code in _split(record.get("languages"))} or {Language.TR}
    allergies = {Allergy(code) for code in _split(record.get("allergies"))}
    return StudentProfile(
        student_id=int(record["student_id"]),
        student_no=str(record["student_no"]),
        display_name=str(record["display_name"]),
        age=int(record["age"]),
        department=Department(str(record["department"])),
        study_year=int(record["study_year"]),
        languages=frozenset(languages),
        study_location=StudyLocation(str(record["study_location"])),
        study_time=StudyTime(str(record["study_time"])),
        sleep_schedule=SleepSchedule(str(record["sleep_schedule"])),
        cleanliness=int(record["cleanliness"]),
        noise_tolerance=int(record["noise_tolerance"]),
        social_energy=int(record["social_energy"]),
        room_time_weekdays=int(record["room_time_weekdays"]),
        guest_frequency=GuestFrequency(str(record["guest_frequency"])),
        smoking=SmokingHabit(str(record["smoking"])),
        alcohol=AlcoholHabit(str(record["alcohol"])),
        diet=DietType(str(record["diet"])),
        allergies=frozenset(allergies),
        interests=parse_interests(record.get("interests")),
        bio=str(record.get("bio") or "")[:1000],
    )
