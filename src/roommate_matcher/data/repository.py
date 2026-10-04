"""Storage abstraction.

Callers talk to :class:`StudentRepository` rather than to a backend, so the storage
layer can change without touching the matching logic. Three implementations are
provided: in-memory, CSV and (in :mod:`roommate_matcher.data.database`) SQLAlchemy.
"""

from __future__ import annotations

import csv
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from roommate_matcher.data.serialization import COLUMNS, from_record, to_record
from roommate_matcher.domain.models import StudentProfile
from roommate_matcher.exceptions import DuplicateStudentError, StudentNotFoundError
from roommate_matcher.logging_config import get_logger

logger = get_logger(__name__)


@runtime_checkable
class StudentRepository(Protocol):
    """Read/write access to student records."""

    def get(self, student_id: int) -> StudentProfile:
        """Fetch one student.

        Args:
            student_id: The id to fetch.

        Returns:
            The stored student.

        Raises:
            StudentNotFoundError: If the id does not exist.
        """
        ...

    def list_all(self) -> list[StudentProfile]:
        """Every stored student.

        Returns:
            All students, in insertion order.
        """
        ...

    def others(self, student_id: int) -> list[StudentProfile]:
        """Every student except the given one.

        Args:
            student_id: The student to exclude.

        Returns:
            The candidate pool.
        """
        ...

    def add(self, student: StudentProfile) -> StudentProfile:
        """Insert a student.

        Args:
            student: The record to store.

        Returns:
            The stored record.

        Raises:
            DuplicateStudentError: If the id already exists.
        """
        ...

    def next_id(self) -> int:
        """The id to assign to the next inserted student.

        Returns:
            One past the current maximum id.
        """
        ...


class InMemoryStudentRepository:
    """A repository backed by a plain dict. The base for tests and for the CSV store."""

    def __init__(self, students: Iterable[StudentProfile] = ()) -> None:
        """Seed the repository.

        Args:
            students: Records to load at construction time.
        """
        self._students: dict[int, StudentProfile] = {s.student_id: s for s in students}
        self._lock = threading.RLock()

    def get(self, student_id: int) -> StudentProfile:
        """Fetch one student.

        Args:
            student_id: The id to fetch.

        Returns:
            The stored student.

        Raises:
            StudentNotFoundError: If the id does not exist.
        """
        try:
            return self._students[student_id]
        except KeyError:
            raise StudentNotFoundError(student_id) from None

    def list_all(self) -> list[StudentProfile]:
        """Every stored student.

        Returns:
            All students.
        """
        return list(self._students.values())

    def others(self, student_id: int) -> list[StudentProfile]:
        """Every student except the given one.

        Args:
            student_id: The student to exclude.

        Returns:
            The candidate pool.
        """
        return [s for s in self._students.values() if s.student_id != student_id]

    def add(self, student: StudentProfile) -> StudentProfile:
        """Insert a student.

        Args:
            student: The record to store.

        Returns:
            The stored record.

        Raises:
            DuplicateStudentError: If the id already exists.
        """
        with self._lock:
            if student.student_id in self._students:
                raise DuplicateStudentError(f"student_id={student.student_id}")
            self._students[student.student_id] = student
            return student

    def next_id(self) -> int:
        """The id to assign to the next inserted student.

        Returns:
            One past the current maximum id, or 1 when empty.
        """
        return max(self._students, default=0) + 1

    def count(self) -> int:
        """How many students are stored.

        Returns:
            The record count.
        """
        return len(self._students)

    def __len__(self) -> int:
        """Number of stored students."""
        return len(self._students)


class CsvStudentRepository(InMemoryStudentRepository):
    """A CSV-backed repository.

    Reads and writes one file, skipping rows that fail validation rather than
    letting a malformed record reach the matching engine.
    """

    def __init__(self, path: Path, *, autoload: bool = True) -> None:
        """Open (and optionally load) a CSV-backed repository.

        Args:
            path: The CSV file to read from and write to.
            autoload: Whether to load existing rows immediately.
        """
        super().__init__()
        self.path = Path(path)
        if autoload and self.path.exists():
            self.load()

    def load(self) -> int:
        """Read every row from disk, skipping invalid ones with a warning.

        Returns:
            How many students were loaded.
        """
        loaded = 0
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    student = from_record(dict(row))
                except (ValueError, KeyError, TypeError) as exc:
                    logger.warning("Skipping malformed row %s: %s", row.get("student_id"), exc)
                    continue
                self._students[student.student_id] = student
                loaded += 1
        logger.info("Loaded %s students from %s", loaded, self.path)
        return loaded

    def save(self) -> None:
        """Write every student back to disk atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for student in self._students.values():
                writer.writerow(to_record(student))
        temporary.replace(self.path)
        logger.info("Saved %s students to %s", len(self._students), self.path)

    def add(self, student: StudentProfile) -> StudentProfile:
        """Insert a student and flush to disk.

        Args:
            student: The record to store.

        Returns:
            The stored record.
        """
        stored = super().add(student)
        self.save()
        return stored


def write_students(path: Path, students: Sequence[StudentProfile]) -> Path:
    """Write students to a CSV file.

    Args:
        path: Destination path; parent directories are created as needed.
        students: The records to write.

    Returns:
        The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(to_record(student) for student in students)
    return path
