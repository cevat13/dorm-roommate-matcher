"""Constraint checking, used two different ways.

Scores answer how compatible two students are; constraints answer whether the
pairing should be allowed at all.

* When **ranking** candidates for analysis, a violated constraint removes the
  candidate from the list, exactly like a filter.
* When **assigning** rooms, every student must get a bed. Removing pairings there
  can make a complete assignment impossible, so a violation becomes a large score
  penalty instead: the solver avoids it when it can, accepts it when it must, and
  the plan reports what had to be accepted.

Both paths share :func:`violations_for`, so the two views can never disagree about
what counts as a violation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from roommate_matcher.domain.enums import Allergy, SmokingHabit
from roommate_matcher.domain.models import StudentProfile
from roommate_matcher.domain.preferences import Constraints

_SMOKING_SEVERITY: dict[SmokingHabit, int] = {
    SmokingHabit.NONE: 0,
    SmokingHabit.OUTDOOR_ONLY: 1,
    SmokingHabit.INDOOR: 2,
}

VIOLATION_LABELS: dict[str, str] = {
    "self": "aynı öğrenci",
    "smoker_mix": "sigara uyuşmazlığı",
    "max_smoking": "sigara sınırı",
    "smoke_allergy": "sigara alerjisi",
    "year_gap": "sınıf farkı",
}
"""Turkish labels for each violation key, used in reports."""


def violations_for(
    first: StudentProfile,
    second: StudentProfile,
    constraints: Constraints,
) -> tuple[str, ...]:
    """List every constraint this pairing violates.

    Args:
        first: One student.
        second: The other student.
        constraints: The requirements to check against.

    Returns:
        Violation keys, empty when the pairing is acceptable.
    """
    if first.student_id == second.student_id:
        return ("self",)

    found: list[str] = []

    if constraints.separate_smokers and _smoker_clash(first, second):
        found.append("smoker_mix")

    if constraints.max_smoking is not None:
        allowed = _SMOKING_SEVERITY[constraints.max_smoking]
        if max(_SMOKING_SEVERITY[first.smoking], _SMOKING_SEVERITY[second.smoking]) > allowed:
            found.append("max_smoking")

    if constraints.enforce_allergies and _smoke_allergy_clash(first, second):
        found.append("smoke_allergy")

    if (
        constraints.max_year_gap is not None
        and abs(first.study_year - second.study_year) > constraints.max_year_gap
    ):
        found.append("year_gap")

    return tuple(found)


def _smoker_clash(first: StudentProfile, second: StudentProfile) -> bool:
    """Check whether a non-smoker would share a room with an indoor smoker.

    Args:
        first: One student.
        second: The other student.

    Returns:
        ``True`` when the pairing mixes a non-smoker with an indoor smoker.
    """
    habits = {first.smoking, second.smoking}
    return SmokingHabit.NONE in habits and SmokingHabit.INDOOR in habits


def _smoke_allergy_clash(first: StudentProfile, second: StudentProfile) -> bool:
    """Check whether either student's smoke allergy is triggered by the other.

    Args:
        first: One student.
        second: The other student.

    Returns:
        ``True`` when a declared smoke allergy would be triggered.
    """
    if Allergy.SMOKE in first.allergies and second.smokes:
        return True
    return Allergy.SMOKE in second.allergies and first.smokes


def is_acceptable(
    first: StudentProfile,
    second: StudentProfile,
    constraints: Constraints,
) -> bool:
    """Whether a pairing violates nothing.

    Args:
        first: One student.
        second: The other student.
        constraints: The requirements to check against.

    Returns:
        ``True`` when no constraint is violated.
    """
    return not violations_for(first, second, constraints)


@dataclass(frozen=True)
class FilterOutcome:
    """The surviving candidates plus a tally of why the others were removed."""

    candidates: tuple[StudentProfile, ...]
    rejections: dict[str, int]

    @property
    def kept(self) -> int:
        """How many candidates survived every constraint."""
        return len(self.candidates)


def apply_constraints(
    student: StudentProfile,
    candidates: Iterable[StudentProfile],
    constraints: Constraints,
) -> FilterOutcome:
    """Filter a candidate pool and record why candidates were dropped.

    Args:
        student: The student being analysed.
        candidates: The pool to filter.
        constraints: The requirements to enforce.

    Returns:
        A :class:`FilterOutcome` with survivors and a rejection tally.
    """
    survivors: list[StudentProfile] = []
    rejections: Counter[str] = Counter()
    for candidate in candidates:
        found = violations_for(student, candidate, constraints)
        if not found:
            survivors.append(candidate)
            continue
        # A candidate rejected for several reasons is counted under the first, so
        # the tally sums to the number of rejected candidates.
        rejections[found[0]] += 1
    return FilterOutcome(tuple(survivors), dict(rejections))


def describe_violations(violations: dict[str, int]) -> str:
    """Render a violation tally as Turkish text for the CLI and UI.

    Args:
        violations: Mapping of violation key to how many times it occurred.

    Returns:
        A human readable summary, or an empty string when there is nothing to report.
    """
    if not violations:
        return ""
    parts = [
        f"{VIOLATION_LABELS.get(key, key)}: {count}"
        for key, count in sorted(violations.items(), key=lambda kv: -kv[1])
    ]
    return ", ".join(parts)
