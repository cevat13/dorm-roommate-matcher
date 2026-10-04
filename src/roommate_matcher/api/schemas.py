"""Request and response models for the HTTP API.

Kept separate from the domain models so that the wire format can change without
affecting the domain, and so internal fields are never exposed by accident.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, Field

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
from roommate_matcher.domain.models import (
    AssignmentPlan,
    FeatureContribution,
    PairScore,
    Room,
    StudentProfile,
)
from roommate_matcher.domain.preferences import Constraints


class StudentCreate(BaseModel):
    """Payload for registering a student record."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_no": "202400042",
                "display_name": "Ahmet Deniz",
                "age": 20,
                "department": "computer_eng",
                "study_year": 2,
                "study_location": "room",
                "study_time": "night",
                "sleep_schedule": "late",
                "cleanliness": 4,
                "noise_tolerance": 3,
                "social_energy": 3,
                "room_time_weekdays": 4,
                "smoking": "none",
                "interests": ["books", "gaming", "technology"],
            }
        }
    )

    student_no: str = Field(min_length=1, max_length=20)
    display_name: str = Field(min_length=1, max_length=80)
    age: int = Field(ge=17, le=40)
    department: Department
    study_year: int = Field(ge=1, le=6)
    languages: list[Language] = Field(default_factory=lambda: [Language.TR])

    study_location: StudyLocation
    study_time: StudyTime

    sleep_schedule: SleepSchedule
    cleanliness: int = Field(ge=1, le=5)
    noise_tolerance: int = Field(ge=1, le=5)
    social_energy: int = Field(ge=1, le=5)
    room_time_weekdays: int = Field(ge=0, le=5)
    guest_frequency: GuestFrequency = GuestFrequency.SOMETIMES

    smoking: SmokingHabit = SmokingHabit.NONE
    alcohol: AlcoholHabit = AlcoholHabit.NONE
    diet: DietType = DietType.OMNIVORE
    allergies: list[Allergy] = Field(default_factory=list)

    interests: list[Interest] = Field(default_factory=list)
    bio: str = Field(default="", max_length=1000)

    def to_profile(self, student_id: int) -> StudentProfile:
        """Convert the payload into a domain model.

        Args:
            student_id: The id to assign.

        Returns:
            A validated :class:`StudentProfile`.
        """
        return StudentProfile(
            student_id=student_id,
            student_no=self.student_no,
            display_name=self.display_name,
            age=self.age,
            department=self.department,
            study_year=self.study_year,
            languages=frozenset(self.languages) or frozenset({Language.TR}),
            study_location=self.study_location,
            study_time=self.study_time,
            sleep_schedule=self.sleep_schedule,
            cleanliness=self.cleanliness,
            noise_tolerance=self.noise_tolerance,
            social_energy=self.social_energy,
            room_time_weekdays=self.room_time_weekdays,
            guest_frequency=self.guest_frequency,
            smoking=self.smoking,
            alcohol=self.alcohol,
            diet=self.diet,
            allergies=frozenset(self.allergies),
            interests=frozenset(self.interests),
            bio=self.bio,
        )


class StudentOut(BaseModel):
    """A student as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    student_id: int
    student_no: str
    display_name: str
    age: int
    department: Department
    study_year: int
    study_location: StudyLocation
    study_time: StudyTime
    sleep_schedule: SleepSchedule
    cleanliness: int
    noise_tolerance: int
    smoking: SmokingHabit
    interests: list[Interest]
    bio: str
    quiet_index: float
    summary: str

    @classmethod
    def from_profile(cls, student: StudentProfile) -> StudentOut:
        """Build the response model from a domain model.

        Args:
            student: The record to expose.

        Returns:
            The API representation.
        """
        return cls(
            student_id=student.student_id,
            student_no=student.student_no,
            display_name=student.display_name,
            age=student.age,
            department=student.department,
            study_year=student.study_year,
            study_location=student.study_location,
            study_time=student.study_time,
            sleep_schedule=student.sleep_schedule,
            cleanliness=student.cleanliness,
            noise_tolerance=student.noise_tolerance,
            smoking=student.smoking,
            interests=sorted(student.interests, key=lambda i: i.value),
            bio=student.bio,
            quiet_index=student.quiet_index,
            summary=student.summary(),
        )


class ContributionOut(BaseModel):
    """One dimension's contribution to a score."""

    feature: str
    label: str
    similarity: float
    weight: float
    verdict: MatchVerdict
    first_value: str
    second_value: str
    explanation: str

    @classmethod
    def from_contribution(cls, contribution: FeatureContribution) -> ContributionOut:
        """Build the response model from a domain model.

        Args:
            contribution: The contribution to expose.

        Returns:
            The API representation.
        """
        return cls(**contribution.model_dump(exclude={"weighted_contribution"}))


class PairOut(BaseModel):
    """A scored student pair with its explanation."""

    first_id: int
    second_id: int
    first_name: str
    second_name: str
    score: float
    percentage: int
    metric: str
    violations: list[str]
    summary: str
    contributions: list[ContributionOut]

    @classmethod
    def from_pair(cls, pair: PairScore) -> PairOut:
        """Build the response model from a domain model.

        Args:
            pair: The scored pair to expose.

        Returns:
            The API representation.
        """
        return cls(
            first_id=pair.first_id,
            second_id=pair.second_id,
            first_name=pair.first_name,
            second_name=pair.second_name,
            score=pair.score,
            percentage=pair.percentage,
            metric=pair.metric,
            violations=list(pair.violations),
            summary=pair.summary,
            contributions=[ContributionOut.from_contribution(c) for c in pair.contributions],
        )


class CandidateResponse(BaseModel):
    """Ranked candidates for one student, with diagnostics."""

    student_id: int
    metric: str
    considered: int
    filtered_out: int
    rejections: dict[str, int]
    candidates: list[PairOut]


class CandidateQuery(BaseModel):
    """Optional body for a candidate request: per-run weights and constraints."""

    weights: dict[str, float] | None = None
    constraints: Constraints | None = None
    metric: str | None = None
    top_n: int = Field(default=10, ge=1, le=100)
    explain: bool = True


class RoomOut(BaseModel):
    """One room of an assignment plan."""

    room_no: int
    first_id: int
    second_id: int
    first_name: str
    second_name: str
    score: float
    percentage: int
    violations: list[str]
    summary: str

    @classmethod
    def from_room(cls, room: Room) -> RoomOut:
        """Build the response model from a domain model.

        Args:
            room: The room to expose.

        Returns:
            The API representation.
        """
        return cls(
            room_no=room.room_no,
            first_id=room.pair.first_id,
            second_id=room.pair.second_id,
            first_name=room.pair.first_name,
            second_name=room.pair.second_name,
            score=room.score,
            percentage=room.pair.percentage,
            violations=list(room.pair.violations),
            summary=room.pair.summary,
        )


class PlanResponse(BaseModel):
    """An assignment plan and its headline figures."""

    strategy: str
    optimal: bool
    room_count: int
    total_score: float
    mean_score: float
    min_score: float
    unplaced: list[int]
    violations: dict[str, int]
    blocking_pair_count: int
    is_stable: bool
    seconds: float
    rooms: list[RoomOut]

    @classmethod
    def from_plan(cls, plan: AssignmentPlan, *, include_rooms: bool = True) -> PlanResponse:
        """Build the response model from a domain model.

        Args:
            plan: The plan to expose.
            include_rooms: Whether to embed every room.

        Returns:
            The API representation.
        """
        return cls(
            strategy=plan.strategy,
            optimal=plan.optimal,
            room_count=plan.room_count,
            total_score=plan.total_score,
            mean_score=plan.mean_score,
            min_score=plan.min_score,
            unplaced=list(plan.unplaced),
            violations=plan.violations,
            blocking_pair_count=len(plan.blocking_pairs),
            is_stable=plan.is_stable,
            seconds=plan.seconds,
            rooms=[RoomOut.from_room(room) for room in plan.rooms] if include_rooms else [],
        )


class AssignRequest(BaseModel):
    """Options for an assignment run."""

    strategy: str | None = None
    metric: str | None = None
    limit: int = Field(default=0, ge=0, le=5000, description="0 means the whole population.")
    weights: dict[str, float] | None = None
    constraints: Constraints | None = None
    save: bool = True


class LoginRequest(BaseModel):
    """Staff credentials."""

    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """An issued access token."""

    access_token: str
    token_type: str = "bearer"
    admin_id: int
    expires_in_minutes: int


class AdminOut(BaseModel):
    """The authenticated staff account."""

    admin_id: int
    email: str


class HealthResponse(BaseModel):
    """Service health and configuration summary."""

    status: str
    version: str
    students: int
    admins: int
    metrics: list[str]
    strategies: list[str]
    default_metric: str
    default_strategy: str
    room_capacity: int
    insecure_secret: bool


class NamedInfo(BaseModel):
    """Description of one registered metric or strategy."""

    key: str
    label: str
    description: str
    flag: bool = Field(
        default=False,
        description="Explainable for a metric; proven optimal for a strategy.",
    )


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    detail: str
    error_type: str
