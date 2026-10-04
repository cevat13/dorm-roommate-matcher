"""Student encoding into a scaled numeric matrix.

Every column is declared with its type and scaled into ``[0, 1]`` before any
geometric metric sees it, so no single wide-ranged attribute can dominate a
distance computation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from roommate_matcher.domain.enums import (
    AlcoholHabit,
    Department,
    DietType,
    GuestFrequency,
    Interest,
    SleepSchedule,
    SmokingHabit,
    StudyLocation,
    StudyTime,
)
from roommate_matcher.domain.models import MAX_WEEKDAY_EVENINGS, StudentProfile

FloatArray = npt.NDArray[np.float64]

_SLEEP_CODE = {SleepSchedule.EARLY: 0.0, SleepSchedule.FLEXIBLE: 0.5, SleepSchedule.LATE: 1.0}
_SMOKING_CODE = {SmokingHabit.NONE: 0.0, SmokingHabit.OUTDOOR_ONLY: 0.5, SmokingHabit.INDOOR: 1.0}
_ALCOHOL_CODE = {AlcoholHabit.NONE: 0.0, AlcoholHabit.OCCASIONAL: 0.5, AlcoholHabit.REGULAR: 1.0}
_GUEST_CODE = {GuestFrequency.RARELY: 0.0, GuestFrequency.SOMETIMES: 0.5, GuestFrequency.OFTEN: 1.0}
_STUDY_TIME_CODE = {StudyTime.DAYTIME: 0.0, StudyTime.EVENING: 0.5, StudyTime.NIGHT: 1.0}
_STUDY_LOCATION_CODE = {
    StudyLocation.LIBRARY: 0.0,
    StudyLocation.MIXED: 0.5,
    StudyLocation.ROOM: 1.0,
}

_CATEGORICAL_SETS: tuple[tuple[str, type[DietType] | type[Department]], ...] = (
    ("diet", DietType),
    ("department", Department),
)

INTEREST_ORDER: tuple[Interest, ...] = tuple(Interest)

MAX_STUDY_YEAR = 6


@dataclass(frozen=True)
class VectorSpace:
    """A student matrix together with the names of its columns."""

    matrix: FloatArray
    feature_names: tuple[str, ...]
    student_ids: tuple[int, ...] = field(default=())

    def index_of(self, student_id: int) -> int:
        """Row index of a student id.

        Ids are not assumed to be contiguous or to match row order.

        Args:
            student_id: The id to locate.

        Returns:
            The row index of that student.

        Raises:
            KeyError: If the id is not present in this vector space.
        """
        try:
            return self.student_ids.index(student_id)
        except ValueError as exc:
            msg = f"student_id {student_id} is not present in this vector space"
            raise KeyError(msg) from exc

    @property
    def n_students(self) -> int:
        """Number of students in the matrix."""
        return int(self.matrix.shape[0])


def feature_names() -> tuple[str, ...]:
    """Names of every column produced by :func:`vectorize`, in order.

    Returns:
        Column names matching the matrix layout.
    """
    names: list[str] = [
        "age",
        "study_year",
        "cleanliness",
        "noise_tolerance",
        "social_energy",
        "room_time_weekdays",
        "sleep_schedule",
        "smoking",
        "alcohol",
        "guest_frequency",
        "study_time",
        "study_location",
        "quiet_index",
        "interest_diversity",
    ]
    for prefix, enum_cls in _CATEGORICAL_SETS:
        names.extend(f"{prefix}__{member.value}" for member in enum_cls)
    names.extend(f"interest__{interest.value}" for interest in INTEREST_ORDER)
    return tuple(names)


def _numeric_block(student: StudentProfile) -> list[float]:
    """Scaled continuous and ordinal features for one student.

    Normalisation happens here rather than downstream, so callers can rely on the
    matrix already being scaled.

    Args:
        student: The student to encode.

    Returns:
        The scaled numeric feature values.
    """
    return [
        (student.age - 17) / (40 - 17),
        (student.study_year - 1) / (MAX_STUDY_YEAR - 1),
        (student.cleanliness - 1) / 4,
        (student.noise_tolerance - 1) / 4,
        (student.social_energy - 1) / 4,
        student.room_time_weekdays / MAX_WEEKDAY_EVENINGS,
        _SLEEP_CODE[student.sleep_schedule],
        _SMOKING_CODE[student.smoking],
        _ALCOHOL_CODE[student.alcohol],
        _GUEST_CODE[student.guest_frequency],
        _STUDY_TIME_CODE[student.study_time],
        _STUDY_LOCATION_CODE[student.study_location],
        student.quiet_index,
        student.interest_diversity,
    ]


def encode(student: StudentProfile) -> FloatArray:
    """Encode a single student into a scaled feature vector.

    Args:
        student: The student to encode.

    Returns:
        A 1-D float array aligned with :func:`feature_names`.
    """
    values = _numeric_block(student)
    for prefix, enum_cls in _CATEGORICAL_SETS:
        current = getattr(student, prefix)
        values.extend(float(current is member) for member in enum_cls)
    values.extend(float(interest in student.interests) for interest in INTEREST_ORDER)
    return np.asarray(values, dtype=np.float64)


def vectorize(students: Sequence[StudentProfile]) -> VectorSpace:
    """Encode many students into one matrix.

    Args:
        students: Students to encode.

    Returns:
        A :class:`VectorSpace` holding the matrix, the column names and the ids.
    """
    names = feature_names()
    if not students:
        return VectorSpace(np.empty((0, len(names)), dtype=np.float64), names, ())
    matrix = np.vstack([encode(student) for student in students])
    return VectorSpace(matrix, names, tuple(student.student_id for student in students))
