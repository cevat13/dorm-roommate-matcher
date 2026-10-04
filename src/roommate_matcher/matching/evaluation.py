"""Offline evaluation, on two axes.

**Metric quality.** Ground truth comes from a rule-based oracle: a simple
compatibility judgement built from constraints a dormitory officer would state out
loud ("no indoor smoker with a non-smoker", "cleanliness within one point", "not
both studying at the room desk"). A metric scores well when the candidates it ranks
highest are the ones the oracle considers liveable.

The oracle is not the Gower scorer; using the scorer as its own ground truth would
grade the model against itself.

**Assignment quality.** A metric that ranks well still has to produce a good plan,
so the strategies are also compared on what they actually deliver: total and worst
room compatibility, unavoidable violations, and stability.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from roommate_matcher.domain.enums import Allergy, SleepSchedule, SmokingHabit, StudyLocation
from roommate_matcher.domain.models import AssignmentPlan, StudentProfile
from roommate_matcher.domain.preferences import MatchPreferences
from roommate_matcher.matching.assignment import RoomAssigner, available_strategies, optimality_gap
from roommate_matcher.matching.engine import CompatibilityEngine
from roommate_matcher.matching.metrics import available_metrics, get_metric

CLEANLINESS_TOLERANCE = 1
YEAR_TOLERANCE = 2


def oracle_relevant(first: StudentProfile, second: StudentProfile) -> bool:
    """Whether two students could genuinely share a room, independent of any metric.

    Args:
        first: One student.
        second: The other student.

    Returns:
        ``True`` when every common-sense requirement is satisfied.
    """
    if first.student_id == second.student_id:
        return False

    habits = {first.smoking, second.smoking}
    if SmokingHabit.NONE in habits and SmokingHabit.INDOOR in habits:
        return False
    if Allergy.SMOKE in first.allergies and second.smokes:
        return False
    if Allergy.SMOKE in second.allergies and first.smokes:
        return False

    if abs(first.cleanliness - second.cleanliness) > CLEANLINESS_TOLERANCE:
        return False

    if {first.sleep_schedule, second.sleep_schedule} == {SleepSchedule.EARLY, SleepSchedule.LATE}:
        return False

    # Two students who both work at the room desk compete for the same quiet hours.
    if first.study_location is StudyLocation.ROOM and second.study_location is StudyLocation.ROOM:
        return False

    return abs(first.study_year - second.study_year) <= YEAR_TOLERANCE


def relevant_set(student: StudentProfile, pool: Sequence[StudentProfile]) -> set[int]:
    """Ids of every candidate the oracle considers relevant.

    Args:
        student: The student being analysed.
        pool: The candidate pool.

    Returns:
        The relevant candidate ids.
    """
    return {c.student_id for c in pool if oracle_relevant(student, c)}


# --------------------------------------------------------------------------- #
# Ranking measures
# --------------------------------------------------------------------------- #


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the top-K that is relevant.

    Args:
        ranked: Candidate ids in ranked order.
        relevant: The relevant ids.
        k: Cut-off.

    Returns:
        A value in ``[0, 1]``.
    """
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for candidate_id in top if candidate_id in relevant) / len(top)


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of all relevant candidates that made the top-K.

    Args:
        ranked: Candidate ids in ranked order.
        relevant: The relevant ids.
        k: Cut-off.

    Returns:
        A value in ``[0, 1]``.
    """
    if not relevant:
        return 0.0
    return sum(1 for candidate_id in ranked[:k] if candidate_id in relevant) / len(relevant)


def mean_reciprocal_rank(ranked: Sequence[int], relevant: set[int]) -> float:
    """Reciprocal rank of the first relevant hit.

    Args:
        ranked: Candidate ids in ranked order.
        relevant: The relevant ids.

    Returns:
        ``1 / rank`` of the first hit, or 0 when there is none.
    """
    for position, candidate_id in enumerate(ranked, start=1):
        if candidate_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Args:
        ranked: Candidate ids in ranked order.
        relevant: The relevant ids.
        k: Cut-off.

    Returns:
        A value in ``[0, 1]``.
    """
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, candidate_id in enumerate(ranked[:k], start=1)
        if candidate_id in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return gain / ideal if ideal > 0 else 0.0


@dataclass(frozen=True)
class MetricScores:
    """Averaged ranking quality for one metric."""

    metric: str
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    queries: int
    seconds: float = 0.0

    def as_row(self) -> dict[str, float | str | int]:
        """Flatten into a table row.

        Returns:
            A dict ready for a DataFrame or a Rich table.
        """
        return {
            "metric": self.metric,
            "P@5": round(self.precision_at_5, 4),
            "P@10": round(self.precision_at_10, 4),
            "R@10": round(self.recall_at_10, 4),
            "MRR": round(self.mrr, 4),
            "nDCG@10": round(self.ndcg_at_10, 4),
            "queries": self.queries,
            "sec": round(self.seconds, 3),
        }


@dataclass
class EvaluationReport:
    """Comparison of every evaluated metric."""

    scores: list[MetricScores] = field(default_factory=list)

    @property
    def winner(self) -> MetricScores | None:
        """The metric with the best nDCG@10."""
        return max(self.scores, key=lambda s: s.ndcg_at_10) if self.scores else None

    def rows(self) -> list[dict[str, float | str | int]]:
        """Table rows sorted best-first by nDCG@10.

        Returns:
            One row per metric.
        """
        return [s.as_row() for s in sorted(self.scores, key=lambda s: -s.ndcg_at_10)]


def evaluate_metric(
    metric_key: str,
    students: Sequence[StudentProfile],
    *,
    sample: int = 100,
    preferences: MatchPreferences | None = None,
) -> MetricScores:
    """Evaluate one metric over a sample of queries.

    The constraint filter is switched off during evaluation, otherwise the filter
    would do the work and every metric would look identical.

    Args:
        metric_key: The metric to evaluate.
        students: The population to evaluate against.
        sample: How many students to use as queries.
        preferences: Optional weight override.

    Returns:
        The averaged :class:`MetricScores`.
    """
    engine = CompatibilityEngine(get_metric(metric_key), preferences)
    queries = list(students[:sample])
    pool = list(students)

    totals = {"p5": 0.0, "p10": 0.0, "r10": 0.0, "mrr": 0.0, "ndcg": 0.0}
    counted = 0
    started = time.perf_counter()

    for student in queries:
        relevant = relevant_set(student, pool)
        if not relevant:
            continue
        report = engine.candidates_for(
            student, pool, top_n=10, explain=False, apply_constraint_filter=False
        )
        ranked = [result.second_id for result in report.results]
        totals["p5"] += precision_at_k(ranked, relevant, 5)
        totals["p10"] += precision_at_k(ranked, relevant, 10)
        totals["r10"] += recall_at_k(ranked, relevant, 10)
        totals["mrr"] += mean_reciprocal_rank(ranked, relevant)
        totals["ndcg"] += ndcg_at_k(ranked, relevant, 10)
        counted += 1

    elapsed = time.perf_counter() - started
    divisor = max(counted, 1)
    return MetricScores(
        metric=metric_key,
        precision_at_5=totals["p5"] / divisor,
        precision_at_10=totals["p10"] / divisor,
        recall_at_10=totals["r10"] / divisor,
        mrr=totals["mrr"] / divisor,
        ndcg_at_10=totals["ndcg"] / divisor,
        queries=counted,
        seconds=elapsed,
    )


def compare_metrics(
    students: Sequence[StudentProfile],
    *,
    metrics: Sequence[str] | None = None,
    sample: int = 100,
) -> EvaluationReport:
    """Evaluate several metrics side by side.

    Args:
        students: The population to evaluate against.
        metrics: Metric keys; defaults to every registered metric.
        sample: How many students to use as queries.

    Returns:
        A populated :class:`EvaluationReport`.
    """
    keys = list(metrics) if metrics else available_metrics()
    return EvaluationReport([evaluate_metric(key, students, sample=sample) for key in keys])


# --------------------------------------------------------------------------- #
# Assignment quality
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlanScores:
    """What one strategy actually delivered for a population."""

    strategy: str
    total_score: float
    mean_score: float
    min_score: float
    rooms: int
    unplaced: int
    violations: int
    blocking_pairs: int
    oracle_share: float
    gap_percent: float
    seconds: float
    optimal: bool

    def as_row(self) -> dict[str, float | str | int]:
        """Flatten into a table row.

        Returns:
            A dict ready for a DataFrame or a Rich table.
        """
        return {
            "strateji": self.strategy,
            "toplam": round(self.total_score, 4),
            "ortalama": round(self.mean_score, 4),
            "en kötü oda": round(self.min_score, 4),
            "oracle %": round(self.oracle_share * 100, 1),
            "fark %": self.gap_percent,
            "blocking": self.blocking_pairs,
            "ihlal": self.violations,
            "yerleşmeyen": self.unplaced,
            "sn": round(self.seconds, 3),
        }


def oracle_share(plan: AssignmentPlan, students: Sequence[StudentProfile]) -> float:
    """Share of rooms the independent oracle considers liveable.

    This is the assignment-side counterpart of precision: it judges the plan with
    the same yardstick used to judge the rankings, and it does not come from the
    scorer being evaluated.

    Args:
        plan: The plan to judge.
        students: The population it was built from.

    Returns:
        A value in ``[0, 1]``; ``0.0`` for an empty plan.
    """
    if not plan.rooms:
        return 0.0
    by_id = {s.student_id: s for s in students}
    liveable = sum(
        1
        for room in plan.rooms
        if oracle_relevant(by_id[room.pair.first_id], by_id[room.pair.second_id])
    )
    return liveable / len(plan.rooms)


def evaluate_plans(
    students: list[StudentProfile],
    *,
    engine: CompatibilityEngine | None = None,
    strategies: Sequence[str] | None = None,
) -> list[PlanScores]:
    """Run every strategy over one population and measure what it delivered.

    The score matrix is built once and shared, so the comparison measures the
    strategies rather than the scoring.

    Args:
        students: The population to assign.
        engine: Optional engine override.
        strategies: Strategy keys; defaults to every registered strategy.

    Returns:
        One :class:`PlanScores` per strategy, best total first.
    """
    engine = engine or CompatibilityEngine()
    matrix = engine.score_matrix(students)
    keys = list(strategies) if strategies else available_strategies()

    plans = [
        RoomAssigner(key, engine).assign(students, explain=False, matrix=matrix) for key in keys
    ]
    best = max(plans, key=lambda plan: plan.total_score)

    scored = [
        PlanScores(
            strategy=plan.strategy,
            total_score=plan.total_score,
            mean_score=plan.mean_score,
            min_score=plan.min_score,
            rooms=plan.room_count,
            unplaced=len(plan.unplaced),
            violations=plan.violation_count,
            blocking_pairs=len(plan.blocking_pairs),
            oracle_share=oracle_share(plan, students),
            gap_percent=optimality_gap(plan, best),
            seconds=plan.seconds,
            optimal=plan.optimal,
        )
        for plan in plans
    ]
    return sorted(scored, key=lambda row: -row.total_score)
