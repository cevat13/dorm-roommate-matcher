"""Exception hierarchy.

Every error raised by this package derives from :class:`RoommateMatcherError`, so
callers can catch one base class and map it to a user-facing message.
"""

from __future__ import annotations


class RoommateMatcherError(Exception):
    """Base class for every error raised by the matcher."""


class StudentNotFoundError(RoommateMatcherError):
    """Raised when a requested student id does not exist in the repository."""

    def __init__(self, student_id: int) -> None:
        """Store the missing id and build a readable message.

        Args:
            student_id: The identifier that could not be resolved.
        """
        self.student_id = student_id
        super().__init__(f"Student with id={student_id} was not found.")


class DuplicateStudentError(RoommateMatcherError):
    """Raised when inserting a student whose id or number already exists."""

    def __init__(self, detail: str) -> None:
        """Build the error.

        Args:
            detail: Which attribute collided.
        """
        super().__init__(f"Student already exists: {detail}")


class InvalidProfileError(RoommateMatcherError):
    """Raised when input fails validation before it reaches the domain model."""


class UnknownMetricError(RoommateMatcherError):
    """Raised when an unregistered similarity metric is requested."""

    def __init__(self, name: str, available: list[str]) -> None:
        """Build the error.

        Args:
            name: The metric key that was requested.
            available: Registered metric keys, used to build a helpful hint.
        """
        self.name = name
        self.available = available
        super().__init__(
            f"Unknown metric {name!r}. Available metrics: {', '.join(sorted(available))}."
        )


class UnknownStrategyError(RoommateMatcherError):
    """Raised when an unregistered assignment strategy is requested."""

    def __init__(self, name: str, available: list[str]) -> None:
        """Build the error.

        Args:
            name: The strategy key that was requested.
            available: Registered strategy keys, used to build a helpful hint.
        """
        self.name = name
        self.available = available
        super().__init__(
            f"Unknown strategy {name!r}. Available strategies: {', '.join(sorted(available))}."
        )


class EmptyCandidatePoolError(RoommateMatcherError):
    """Raised when the constraints eliminate every candidate."""

    def __init__(self, reasons: dict[str, int] | None = None) -> None:
        """Build the error.

        Args:
            reasons: Optional mapping of constraint name to how many candidates it removed.
        """
        self.reasons = reasons or {}
        hint = ""
        if self.reasons:
            worst = max(self.reasons.items(), key=lambda kv: kv[1])
            hint = f" The most restrictive constraint was {worst[0]!r} ({worst[1]} removed)."
        super().__init__(f"No candidate satisfied the constraints.{hint}")


class NoPlanError(RoommateMatcherError):
    """Raised when an assignment plan is requested before one has been computed."""

    def __init__(self) -> None:
        """Build the error."""
        super().__init__("No assignment plan has been computed yet. Run an assignment first.")


class AuthenticationError(RoommateMatcherError):
    """Raised on bad credentials or an invalid/expired token."""
