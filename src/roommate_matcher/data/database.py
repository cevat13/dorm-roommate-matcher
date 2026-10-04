"""SQLAlchemy persistence.

Implements the same ``StudentRepository`` protocol as the CSV store, so the two are
interchangeable at the composition root. Assignment plans are persisted too, so a
plan computed once can be reviewed, explained and re-read without recomputing it.

SQLite is the default so the project runs with no external services; pointing
``RM_DATABASE_URL`` at PostgreSQL is the only change needed otherwise.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    delete,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from roommate_matcher.config import get_settings
from roommate_matcher.data.serialization import from_record, to_record
from roommate_matcher.domain.models import StudentProfile
from roommate_matcher.exceptions import DuplicateStudentError, StudentNotFoundError
from roommate_matcher.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


class StudentRow(Base):
    """A student row, mirroring the flat CSV column layout."""

    __tablename__ = "students"

    student_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_no: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(80))
    age: Mapped[int] = mapped_column(Integer)
    department: Mapped[str] = mapped_column(String(30), index=True)
    study_year: Mapped[int] = mapped_column(Integer, index=True)
    languages: Mapped[str] = mapped_column(String(100), default="tr")
    study_location: Mapped[str] = mapped_column(String(20), index=True)
    study_time: Mapped[str] = mapped_column(String(20))
    sleep_schedule: Mapped[str] = mapped_column(String(20), index=True)
    cleanliness: Mapped[int] = mapped_column(Integer)
    noise_tolerance: Mapped[int] = mapped_column(Integer)
    social_energy: Mapped[int] = mapped_column(Integer)
    room_time_weekdays: Mapped[int] = mapped_column(Integer)
    guest_frequency: Mapped[str] = mapped_column(String(20))
    smoking: Mapped[str] = mapped_column(String(20), index=True)
    alcohol: Mapped[str] = mapped_column(String(20))
    diet: Mapped[str] = mapped_column(String(20))
    allergies: Mapped[str] = mapped_column(String(200), default="")
    interests: Mapped[str] = mapped_column(String(400), default="")
    bio: Mapped[str] = mapped_column(String(1000), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    def to_profile(self) -> StudentProfile:
        """Convert this row into a validated domain model.

        Returns:
            The corresponding :class:`StudentProfile`.
        """
        record: dict[str, Any] = {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }
        return from_record(record)

    @classmethod
    def from_profile(cls, student: StudentProfile) -> StudentRow:
        """Build a row from a domain model.

        Args:
            student: The record to persist.

        Returns:
            An unsaved :class:`StudentRow`.
        """
        return cls(**to_record(student))


class PlanRow(Base):
    """One assignment run, so a computed plan can be reviewed later."""

    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(30))
    metric: Mapped[str] = mapped_column(String(30))
    room_count: Mapped[int] = mapped_column(Integer)
    total_score: Mapped[float] = mapped_column(Float)
    min_score: Mapped[float] = mapped_column(Float)
    violation_count: Mapped[int] = mapped_column(Integer, default=0)
    blocking_pair_count: Mapped[int] = mapped_column(Integer, default=0)
    unplaced: Mapped[str] = mapped_column(String(500), default="")
    seconds: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )


class RoomRow(Base):
    """One room belonging to a stored plan."""

    __tablename__ = "rooms"
    __table_args__ = (UniqueConstraint("plan_id", "room_no", name="uq_room_per_plan"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[int] = mapped_column(Integer, index=True)
    room_no: Mapped[int] = mapped_column(Integer)
    first_id: Mapped[int] = mapped_column(Integer, index=True)
    second_id: Mapped[int] = mapped_column(Integer, index=True)
    score: Mapped[float] = mapped_column(Float)
    violations: Mapped[str] = mapped_column(String(200), default="")


class AdminRow(Base):
    """A dormitory staff account. Students are data, not API users."""

    __tablename__ = "admins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


def create_db_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine.

    Args:
        url: Database URL; defaults to the configured one.
        echo: Whether to log every statement.

    Returns:
        A configured :class:`~sqlalchemy.engine.Engine`.
    """
    settings = get_settings()
    resolved = url or settings.database_url
    connect_args = {"check_same_thread": False} if resolved.startswith("sqlite") else {}
    if resolved.startswith("sqlite:///") and ":memory:" not in resolved:
        settings.ensure_directories()
    return create_engine(resolved, echo=echo, future=True, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    """Create every table if it does not already exist.

    Args:
        engine: The engine to create tables on.
    """
    Base.metadata.create_all(engine)
    logger.info("Database schema ready")


class SqlStudentRepository:
    """A repository backed by SQLAlchemy.

    Args:
        engine: The engine to use; one is created from settings when omitted.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        """Initialise the repository and ensure the schema exists."""
        self.engine = engine or create_db_engine()
        init_db(self.engine)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Provide a transactional scope.

        Yields:
            An open session that is committed on success and rolled back on error.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Students
    # ------------------------------------------------------------------ #

    def get(self, student_id: int) -> StudentProfile:
        """Fetch one student.

        Args:
            student_id: The id to fetch.

        Returns:
            The stored student.

        Raises:
            StudentNotFoundError: If the id does not exist.
        """
        with self.session() as session:
            row = session.get(StudentRow, student_id)
            if row is None:
                raise StudentNotFoundError(student_id)
            return row.to_profile()

    def list_all(self) -> list[StudentProfile]:
        """Every stored student.

        Returns:
            All students, ordered by id.
        """
        with self.session() as session:
            rows = session.scalars(select(StudentRow).order_by(StudentRow.student_id)).all()
            return [row.to_profile() for row in rows]

    def others(self, student_id: int) -> list[StudentProfile]:
        """Every student except the given one.

        Args:
            student_id: The student to exclude.

        Returns:
            The candidate pool.
        """
        with self.session() as session:
            rows = session.scalars(
                select(StudentRow)
                .where(StudentRow.student_id != student_id)
                .order_by(StudentRow.student_id)
            ).all()
            return [row.to_profile() for row in rows]

    def add(self, student: StudentProfile) -> StudentProfile:
        """Insert a student.

        Args:
            student: The record to store.

        Returns:
            The stored record.

        Raises:
            DuplicateStudentError: If the id already exists.
        """
        with self.session() as session:
            if session.get(StudentRow, student.student_id) is not None:
                raise DuplicateStudentError(f"student_id={student.student_id}")
            session.add(StudentRow.from_profile(student))
        return student

    def bulk_add(self, students: list[StudentProfile]) -> int:
        """Insert many students in one transaction.

        Args:
            students: The records to store.

        Returns:
            How many rows were inserted.
        """
        with self.session() as session:
            session.add_all([StudentRow.from_profile(student) for student in students])
        logger.info("Inserted %s students", len(students))
        return len(students)

    def next_id(self) -> int:
        """The id to assign to the next inserted student.

        Returns:
            One past the current maximum id.
        """
        with self.session() as session:
            rows = session.scalars(select(StudentRow.student_id)).all()
            return max(rows, default=0) + 1

    def count(self) -> int:
        """How many students are stored.

        Returns:
            The row count.
        """
        with self.session() as session:
            return len(session.scalars(select(StudentRow.student_id)).all())

    def reset(self) -> int:
        """Delete every student and every stored plan.

        Staff accounts are kept, so resetting the data does not lock the operator
        out. Plans are dropped along with the students because their rooms refer to
        student ids that would otherwise dangle.

        Returns:
            How many students were removed.
        """
        with self.session() as session:
            removed = len(session.scalars(select(StudentRow.student_id)).all())
            session.execute(delete(RoomRow))
            session.execute(delete(PlanRow))
            session.execute(delete(StudentRow))
        logger.info("Reset removed %s students and every stored plan", removed)
        return removed

    # ------------------------------------------------------------------ #
    # Plans
    # ------------------------------------------------------------------ #

    def save_plan(self, plan: Any, metric: str) -> int:
        """Persist an assignment plan and its rooms.

        Args:
            plan: The :class:`~roommate_matcher.domain.models.AssignmentPlan` to store.
            metric: The compatibility metric the plan was built with.

        Returns:
            The stored plan's id.
        """
        with self.session() as session:
            row = PlanRow(
                strategy=plan.strategy,
                metric=metric,
                room_count=plan.room_count,
                total_score=plan.total_score,
                min_score=plan.min_score,
                violation_count=plan.violation_count,
                blocking_pair_count=len(plan.blocking_pairs),
                unplaced=",".join(str(sid) for sid in plan.unplaced),
                seconds=plan.seconds,
            )
            session.add(row)
            session.flush()
            plan_id = int(row.id)
            session.add_all(
                [
                    RoomRow(
                        plan_id=plan_id,
                        room_no=room.room_no,
                        first_id=room.pair.first_id,
                        second_id=room.pair.second_id,
                        score=room.score,
                        violations=",".join(room.pair.violations),
                    )
                    for room in plan.rooms
                ]
            )
        logger.info("Saved plan %s (%s rooms)", plan_id, plan.room_count)
        return plan_id

    def latest_plan_row(self) -> PlanRow | None:
        """The most recently stored plan's summary row.

        Returns:
            The plan row, or ``None`` when no plan has been stored.
        """
        with self.session() as session:
            return session.scalar(select(PlanRow).order_by(PlanRow.created_at.desc()).limit(1))

    def plan_rooms(self, plan_id: int) -> list[RoomRow]:
        """Rooms belonging to a stored plan.

        Args:
            plan_id: The plan to read.

        Returns:
            Its rooms, ordered by room number.
        """
        with self.session() as session:
            return list(
                session.scalars(
                    select(RoomRow).where(RoomRow.plan_id == plan_id).order_by(RoomRow.room_no)
                ).all()
            )

    # ------------------------------------------------------------------ #
    # Admins
    # ------------------------------------------------------------------ #

    def add_admin(self, email: str, password_hash: str) -> None:
        """Create a staff account.

        Args:
            email: The login address.
            password_hash: The bcrypt hash of the password.

        Raises:
            DuplicateStudentError: If the email is already registered.
        """
        with self.session() as session:
            if session.scalar(select(AdminRow).where(AdminRow.email == email)) is not None:
                raise DuplicateStudentError(f"email={email}")
            session.add(AdminRow(email=email, password_hash=password_hash))

    def find_admin(self, email: str) -> AdminRow | None:
        """Look a staff account up by email.

        Args:
            email: The address to search for.

        Returns:
            The matching row, or ``None``.
        """
        with self.session() as session:
            return session.scalar(select(AdminRow).where(AdminRow.email == email))

    def admin_by_id(self, admin_id: int) -> AdminRow | None:
        """Look a staff account up by id, for token resolution.

        Args:
            admin_id: The id to search for.

        Returns:
            The matching row, or ``None``.
        """
        with self.session() as session:
            return session.get(AdminRow, admin_id)

    def admin_count(self) -> int:
        """How many staff accounts exist.

        Returns:
            The row count.
        """
        with self.session() as session:
            return len(session.scalars(select(AdminRow.id)).all())
