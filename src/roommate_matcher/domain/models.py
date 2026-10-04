"""Core domain models.

:class:`StudentProfile` is the single definition of what a student record is; the
API schemas, the CSV loader, the database layer and the generator all construct it,
so validation happens once and a malformed record cannot reach the matching engine.

:class:`Room` and :class:`AssignmentPlan` are the output of an assignment run. A
room is not pre-existing inventory: it is a pair produced by the matching, carrying
the score and the reasoning that placed those two students together.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from roommate_matcher.domain.enums import (
    AlcoholHabit,
    Allergy,
    Department,
    DietType,
    GuestFrequency,
    Interest,
    Language,
    MatchVerdict,
    SleepSchedule,
    SmokingHabit,
    StudyLocation,
    StudyTime,
)

Rating = Annotated[int, Field(ge=1, le=5)]
"""A 1-5 Likert rating used for cleanliness, noise tolerance and social energy."""

MAX_WEEKDAY_EVENINGS = 5
"""Weekday evenings in a week; the upper bound for ``room_time_weekdays``."""


class StudentProfile(BaseModel):
    """A dormitory applicant's record.

    Every field is validated on construction, so downstream code can rely on ranges
    and vocabularies without defensive checks.
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity ----------------------------------------------------------
    student_id: int = Field(ge=1)
    student_no: str = Field(min_length=1, max_length=20)
    display_name: str = Field(min_length=1, max_length=80)
    age: int = Field(ge=17, le=40)
    department: Department
    study_year: int = Field(ge=1, le=6)
    languages: frozenset[Language] = frozenset({Language.TR})

    # --- Study habits ------------------------------------------------------
    study_location: StudyLocation
    study_time: StudyTime

    # --- Living habits -----------------------------------------------------
    sleep_schedule: SleepSchedule
    cleanliness: Rating
    noise_tolerance: Rating
    social_energy: Rating
    room_time_weekdays: int = Field(
        ge=0,
        le=MAX_WEEKDAY_EVENINGS,
        description="Weekday evenings per week spent in the room (0-5).",
    )
    guest_frequency: GuestFrequency = GuestFrequency.SOMETIMES

    smoking: SmokingHabit = SmokingHabit.NONE
    alcohol: AlcoholHabit = AlcoholHabit.NONE
    diet: DietType = DietType.OMNIVORE
    allergies: frozenset[Allergy] = frozenset()

    # --- Soft data ---------------------------------------------------------
    interests: frozenset[Interest] = frozenset()
    bio: str = Field(default="", max_length=1000)

    @field_validator("display_name", "student_no")
    @classmethod
    def _strip(cls, value: str) -> str:
        """Trim surrounding whitespace on free-text identity fields.

        Args:
            value: Raw input.

        Returns:
            The trimmed value.
        """
        return value.strip()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def smokes(self) -> bool:
        """Whether this student smokes at all."""
        return self.smoking is not SmokingHabit.NONE

    @computed_field  # type: ignore[prop-decorator]
    @property
    def studies_in_room(self) -> bool:
        """Whether the room desk is this student's main study spot."""
        return self.study_location is not StudyLocation.LIBRARY

    @computed_field  # type: ignore[prop-decorator]
    @property
    def interest_diversity(self) -> float:
        """Engineered feature: share of the taxonomy this student is interested in."""
        return round(len(self.interests) / len(Interest), 4)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def quiet_index(self) -> float:
        """Engineered feature in ``[0, 1]`` summarising how quiet a roommate this is.

        High values mean tidy, noise-averse and often in the room.
        """
        tidy = (self.cleanliness - 1) / 4
        quiet = 1 - (self.noise_tolerance - 1) / 4
        present = self.room_time_weekdays / MAX_WEEKDAY_EVENINGS
        return round((tidy + quiet + present) / 3, 4)

    def summary(self) -> str:
        """One-line description used in CLI tables and tooltips.

        Returns:
            A compact Turkish summary of the profile.
        """
        return (
            f"{self.display_name} ({self.student_no}) - {self.department.label_tr} "
            f"{self.study_year}. sınıf - {self.sleep_schedule.label_tr} - "
            f"temizlik {self.cleanliness}/5 - {self.smoking.label_tr}"
        )


class FeatureContribution(BaseModel):
    """How one feature contributed to an overall compatibility score.

    Every score carries the per-feature breakdown that produced it.
    """

    model_config = ConfigDict(frozen=True)

    feature: str
    label: str
    similarity: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    verdict: MatchVerdict
    first_value: str
    second_value: str
    explanation: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def weighted_contribution(self) -> float:
        """The share of the final score this feature is responsible for."""
        return round(self.similarity * self.weight, 6)


class PairScore(BaseModel):
    """A scored student pair together with the reasoning behind the score."""

    model_config = ConfigDict(frozen=True)

    first_id: int
    second_id: int
    first_name: str = ""
    second_name: str = ""
    score: float = Field(ge=0.0, le=1.0)
    metric: str = ""
    contributions: tuple[FeatureContribution, ...] = ()
    violations: tuple[str, ...] = ()
    summary: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def percentage(self) -> int:
        """Score rendered as an integer percentage for display."""
        return round(self.score * 100)

    def strengths(self, limit: int = 3) -> tuple[FeatureContribution, ...]:
        """Strongest positive factors behind this pairing.

        Args:
            limit: How many factors to return.

        Returns:
            Contributions sorted by weighted impact, best first.
        """
        ranked = sorted(
            (c for c in self.contributions if c.verdict is MatchVerdict.MATCH),
            key=lambda c: c.weighted_contribution,
            reverse=True,
        )
        return tuple(ranked[:limit])

    def concerns(self, limit: int = 3) -> tuple[FeatureContribution, ...]:
        """Biggest incompatibilities behind this pairing.

        Args:
            limit: How many factors to return.

        Returns:
            Clashing contributions sorted by the weight they carry.
        """
        ranked = sorted(
            (c for c in self.contributions if c.verdict is MatchVerdict.CLASH),
            key=lambda c: c.weight * (1 - c.similarity),
            reverse=True,
        )
        return tuple(ranked[:limit])


class Room(BaseModel):
    """Two students placed together, with the score and reasoning behind it."""

    model_config = ConfigDict(frozen=True)

    room_no: int = Field(ge=1)
    pair: PairScore

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score(self) -> float:
        """Compatibility of the two occupants."""
        return self.pair.score

    @computed_field  # type: ignore[prop-decorator]
    @property
    def occupants(self) -> tuple[int, int]:
        """Student ids of the two occupants."""
        return (self.pair.first_id, self.pair.second_id)

    def label(self) -> str:
        """Room heading for tables and reports.

        Returns:
            A line naming the room and both occupants.
        """
        return (
            f"Oda {self.room_no}: {self.pair.first_name} (#{self.pair.first_id}) + "
            f"{self.pair.second_name} (#{self.pair.second_id}) - %{self.pair.percentage}"
        )


class AssignmentPlan(BaseModel):
    """The result of an assignment run over the whole student population."""

    model_config = ConfigDict(frozen=True)

    strategy: str
    rooms: tuple[Room, ...] = ()
    unplaced: tuple[int, ...] = ()
    violations: dict[str, int] = Field(default_factory=dict)
    blocking_pairs: tuple[tuple[int, int], ...] = ()
    seconds: float = 0.0
    optimal: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def room_count(self) -> int:
        """How many rooms were formed."""
        return len(self.rooms)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_score(self) -> float:
        """Sum of every room's compatibility; the objective being maximised."""
        return round(sum(room.score for room in self.rooms), 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_score(self) -> float:
        """Average room compatibility."""
        if not self.rooms:
            return 0.0
        return round(self.total_score / len(self.rooms), 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def min_score(self) -> float:
        """Compatibility of the worst room, which bounds the worst student outcome."""
        if not self.rooms:
            return 0.0
        return round(min(room.score for room in self.rooms), 6)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def violation_count(self) -> int:
        """Total number of constraint violations that could not be avoided."""
        return sum(self.violations.values())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_stable(self) -> bool:
        """Whether the plan contains no blocking pair."""
        return not self.blocking_pairs

    def room_of(self, student_id: int) -> Room | None:
        """Find the room a student was placed in.

        Args:
            student_id: The student to look up.

        Returns:
            The room, or ``None`` when the student was left unplaced.
        """
        for room in self.rooms:
            if student_id in room.occupants:
                return room
        return None

    def worst_rooms(self, limit: int = 5) -> tuple[Room, ...]:
        """Rooms most likely to need attention from dormitory staff.

        Args:
            limit: How many rooms to return.

        Returns:
            Rooms sorted by score, lowest first.
        """
        return tuple(sorted(self.rooms, key=lambda room: room.score)[:limit])
