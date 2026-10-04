"""The compatibility engine: scores, explains, and builds the score matrix.

One in-memory object shared by the CLI, the API and the UI, so all three produce
identical results. The engine answers "how compatible are these two?"; turning the
answers into room assignments is :mod:`roommate_matcher.matching.assignment`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from roommate_matcher.config import get_settings
from roommate_matcher.domain.models import PairScore, StudentProfile
from roommate_matcher.domain.preferences import MatchPreferences
from roommate_matcher.exceptions import EmptyCandidatePoolError
from roommate_matcher.logging_config import get_logger
from roommate_matcher.matching.constraints import FilterOutcome, apply_constraints, violations_for
from roommate_matcher.matching.explain import build_summary
from roommate_matcher.matching.metrics import CompatibilityMetric, get_metric

logger = get_logger(__name__)

FloatMatrix = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ScoreMatrix:
    """Pairwise compatibility for a whole population.

    The matrix is symmetric with 1.0 on the diagonal. Building it is ``O(n^2)`` by
    definition, which is inherent to assigning every student rather than ranking
    for one.
    """

    students: tuple[StudentProfile, ...]
    scores: FloatMatrix
    violations: tuple[tuple[tuple[str, ...], ...], ...] = ()

    @property
    def size(self) -> int:
        """Number of students covered."""
        return len(self.students)

    def index_of(self, student_id: int) -> int:
        """Row index of a student id.

        Args:
            student_id: The id to locate.

        Returns:
            The row index.

        Raises:
            KeyError: If the id is not present.
        """
        for index, student in enumerate(self.students):
            if student.student_id == student_id:
                return index
        msg = f"student_id {student_id} is not in this score matrix"
        raise KeyError(msg)

    def score_between(self, first_index: int, second_index: int) -> float:
        """Compatibility of two students by row index.

        Args:
            first_index: Row index of one student.
            second_index: Row index of the other.

        Returns:
            Their compatibility score.
        """
        return float(self.scores[first_index, second_index])

    def violations_between(self, first_index: int, second_index: int) -> tuple[str, ...]:
        """Constraint violations of a pairing by row index.

        Args:
            first_index: Row index of one student.
            second_index: Row index of the other.

        Returns:
            Violation keys, empty when the pairing is acceptable.
        """
        if not self.violations:
            return ()
        return self.violations[first_index][second_index]


@dataclass(frozen=True)
class CandidateReport:
    """Ranked candidates for one student, with diagnostics."""

    results: tuple[PairScore, ...]
    considered: int
    filtered_out: int
    rejections: dict[str, int] = field(default_factory=dict)
    metric: str = ""

    @property
    def best(self) -> PairScore | None:
        """The single best candidate, or ``None`` when nothing survived."""
        return self.results[0] if self.results else None


class CompatibilityEngine:
    """Scores and explains student pairings.

    Args:
        metric: Metric key or instance. Defaults to the configured metric.
        preferences: Weights and constraints. Defaults to the standard preset.
    """

    def __init__(
        self,
        metric: str | CompatibilityMetric | None = None,
        preferences: MatchPreferences | None = None,
    ) -> None:
        """Initialise the engine."""
        settings = get_settings()
        if metric is None:
            metric = settings.default_metric
        self.metric: CompatibilityMetric = get_metric(metric) if isinstance(metric, str) else metric
        self.preferences = preferences or MatchPreferences()
        self._settings = settings

    # ------------------------------------------------------------------ #
    # Pair scoring
    # ------------------------------------------------------------------ #

    def score_pair(self, first: StudentProfile, second: StudentProfile) -> float:
        """Score a single pairing.

        Args:
            first: One student.
            second: The other student.

        Returns:
            A score in ``[0, 1]``.
        """
        raw = self.metric.score(first, second, self.preferences)
        return float(np.clip(raw, 0.0, 1.0))

    def explain_pair(self, first: StudentProfile, second: StudentProfile) -> PairScore:
        """Score one pairing and attach the full explanation.

        Args:
            first: One student.
            second: The other student.

        Returns:
            A fully populated :class:`PairScore`.
        """
        score = self.score_pair(first, second)
        contributions = self.metric.contributions(first, second, self.preferences)
        violations = violations_for(first, second, self.preferences.constraints)
        return PairScore(
            first_id=first.student_id,
            second_id=second.student_id,
            first_name=first.display_name,
            second_name=second.display_name,
            score=round(score, self._settings.score_precision),
            metric=self.metric.key,
            contributions=contributions,
            violations=violations,
            summary=build_summary(contributions, score, violations=violations),
        )

    # ------------------------------------------------------------------ #
    # Population-level scoring
    # ------------------------------------------------------------------ #

    def score_matrix(
        self,
        students: Sequence[StudentProfile],
        *,
        with_violations: bool = True,
    ) -> ScoreMatrix:
        """Compute every pairwise score for a population.

        Only the upper triangle is evaluated and mirrored, since every metric here
        is symmetric. That halves the work without changing the result.

        Args:
            students: The population to score.
            with_violations: Whether to record constraint violations per pair.

        Returns:
            A :class:`ScoreMatrix` with scores and optional violation keys.
        """
        size = len(students)
        scores = np.ones((size, size), dtype=np.float64)
        constraints = self.preferences.constraints
        violations: list[list[tuple[str, ...]]] = (
            [[() for _ in range(size)] for _ in range(size)] if with_violations else []
        )

        for i in range(size):
            for j in range(i + 1, size):
                value = self.score_pair(students[i], students[j])
                scores[i, j] = scores[j, i] = value
                if with_violations:
                    found = violations_for(students[i], students[j], constraints)
                    violations[i][j] = violations[j][i] = found

        logger.debug("Built %sx%s score matrix", size, size)
        return ScoreMatrix(
            students=tuple(students),
            scores=scores,
            violations=tuple(tuple(row) for row in violations) if with_violations else (),
        )

    # ------------------------------------------------------------------ #
    # Candidate analysis
    # ------------------------------------------------------------------ #

    def candidates_for(
        self,
        student: StudentProfile,
        pool: Sequence[StudentProfile],
        *,
        top_n: int | None = None,
        explain: bool = True,
        apply_constraint_filter: bool = True,
        raise_on_empty: bool = False,
    ) -> CandidateReport:
        """Rank the most compatible roommates for one student.

        This is an analysis tool for dormitory staff: it answers "who would suit
        this student best?" without committing to a placement.

        Args:
            student: The student being analysed.
            pool: The candidates to rank.
            top_n: How many results to return. Defaults to the configured value.
            explain: Whether to attach per-dimension breakdowns.
            apply_constraint_filter: Whether to drop candidates that violate a
                constraint. Assignment does not drop them, it penalises them.
            raise_on_empty: Raise instead of returning an empty report when the
                constraints eliminate everybody.

        Returns:
            A :class:`CandidateReport` with ranked results and diagnostics.

        Raises:
            EmptyCandidatePoolError: When ``raise_on_empty`` is set and no candidate
                survived the constraints.
        """
        limit = min(top_n or self._settings.default_top_n, self._settings.max_top_n)

        candidates = list(pool)
        considered = len(candidates)
        if apply_constraint_filter:
            outcome = apply_constraints(student, candidates, self.preferences.constraints)
        else:
            survivors = tuple(c for c in candidates if c.student_id != student.student_id)
            outcome = FilterOutcome(survivors, {"self": considered - len(survivors)})

        if not outcome.candidates:
            if raise_on_empty:
                raise EmptyCandidatePoolError(outcome.rejections)
            return CandidateReport((), considered, considered, outcome.rejections, self.metric.key)

        scores = self.metric.score_many(student, outcome.candidates, self.preferences)
        order = np.argsort(-scores, kind="stable")[:limit]

        results: list[PairScore] = []
        for index in order:
            candidate = outcome.candidates[int(index)]
            if explain:
                results.append(self.explain_pair(student, candidate))
            else:
                results.append(
                    PairScore(
                        first_id=student.student_id,
                        second_id=candidate.student_id,
                        first_name=student.display_name,
                        second_name=candidate.display_name,
                        score=round(float(scores[index]), self._settings.score_precision),
                        metric=self.metric.key,
                    )
                )

        return CandidateReport(
            results=tuple(results),
            considered=considered,
            filtered_out=considered - outcome.kept,
            rejections=outcome.rejections,
            metric=self.metric.key,
        )
