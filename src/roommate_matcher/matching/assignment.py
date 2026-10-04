"""Turning pairwise compatibility into a dormitory room plan.

Every student must get a bed, so this is not a ranking problem: the output is a
partition of the whole population into two-person rooms. With a fixed capacity of
two and "maximise total compatibility" as the objective, that is exactly a
**maximum weight perfect matching** on a general graph, which Edmonds' blossom
algorithm solves *exactly* in polynomial time.

Constraints are modelled as large negative edge weights rather than removed edges.
Removing edges can make a complete assignment impossible (an odd number of smokers
is enough), and leaving a student without a bed is not an acceptable output. With a
penalty the solver avoids a violation whenever it can, accepts it when it must, and
the plan reports every violation it had to accept.

Three further strategies are registered for comparison: Irving's stable-roommates
algorithm (a *different* objective, stability rather than total weight), a greedy
baseline, and a seeded random baseline.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import ClassVar

import networkx as nx

from roommate_matcher.config import get_settings
from roommate_matcher.domain.models import AssignmentPlan, PairScore, Room, StudentProfile
from roommate_matcher.exceptions import UnknownStrategyError
from roommate_matcher.logging_config import get_logger
from roommate_matcher.matching.engine import CompatibilityEngine, ScoreMatrix
from roommate_matcher.matching.explain import build_summary

logger = get_logger(__name__)

VIOLATION_PENALTY = 1000.0
"""Weight subtracted per violated constraint.

Large enough to dominate any compatibility score (which lives in ``[0, 1]``), so a
violation is only ever accepted when no violation-free assignment exists.
"""

Pairing = list[tuple[int, int]]
"""Row-index pairs produced by a strategy, before they become rooms."""

_REGISTRY: dict[str, type[AssignmentStrategy]] = {}


def register(cls: type[AssignmentStrategy]) -> type[AssignmentStrategy]:
    """Register a strategy class under its ``key``.

    Args:
        cls: The strategy class to register.

    Returns:
        The same class, so this can be used as a decorator.
    """
    _REGISTRY[cls.key] = cls
    return cls


def available_strategies() -> list[str]:
    """Keys of every registered strategy.

    Returns:
        Sorted strategy keys.
    """
    return sorted(_REGISTRY)


def get_strategy(key: str) -> AssignmentStrategy:
    """Instantiate a strategy by key.

    Args:
        key: The strategy key, e.g. ``"optimal"``.

    Returns:
        A ready-to-use strategy instance.

    Raises:
        UnknownStrategyError: If no strategy is registered under that key.
    """
    try:
        return _REGISTRY[key]()
    except KeyError:
        raise UnknownStrategyError(key, available_strategies()) from None


def edge_weight(matrix: ScoreMatrix, first: int, second: int) -> float:
    """Objective weight of one candidate pairing.

    Args:
        matrix: The population's score matrix.
        first: Row index of one student.
        second: Row index of the other.

    Returns:
        The compatibility score minus a penalty per violated constraint.
    """
    score = matrix.score_between(first, second)
    penalty = VIOLATION_PENALTY * len(matrix.violations_between(first, second))
    return score - penalty


class AssignmentStrategy(ABC):
    """Strategy interface: partition a population into two-person rooms."""

    key: ClassVar[str] = "base"
    label: ClassVar[str] = "Base"
    description: ClassVar[str] = ""
    optimal: ClassVar[bool] = False

    @abstractmethod
    def pair_up(self, matrix: ScoreMatrix) -> Pairing:
        """Choose the pairings.

        Args:
            matrix: The population's score matrix.

        Returns:
            Row-index pairs; students left over are simply absent.
        """


@register
class OptimalAssignment(AssignmentStrategy):
    """Maximum weight perfect matching via the blossom algorithm.

    ``maxcardinality=True`` maximises the number of pairs first and the total weight
    second, which is what a dormitory needs: everybody gets a bed, and among all the
    ways of doing that the most compatible one is chosen. Negative penalty weights
    do not prevent a complete matching, they only make the solver avoid those edges.
    """

    key = "optimal"
    label = "Optimal (maksimum ağırlıklı eşleştirme)"
    description = "Blossom algoritması ile toplam uyumu maksimize eden kesin çözüm."
    optimal = True

    def pair_up(self, matrix: ScoreMatrix) -> Pairing:
        """Solve the matching exactly.

        Args:
            matrix: The population's score matrix.

        Returns:
            Row-index pairs covering every student bar at most one.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(matrix.size))
        for i in range(matrix.size):
            for j in range(i + 1, matrix.size):
                graph.add_edge(i, j, weight=edge_weight(matrix, i, j))
        matched = nx.max_weight_matching(graph, maxcardinality=True)
        return [(min(a, b), max(a, b)) for a, b in matched]


@register
class GreedyAssignment(AssignmentStrategy):
    """Take the best remaining pair until nobody is left.

    A natural first instinct, and a useful baseline: it locks in strong pairs early
    and can strand the leftovers with each other.
    """

    key = "greedy"
    label = "Açgözlü"
    description = "En yüksek skorlu çifti sırayla seç; kalanlar birbirine kalır."

    def pair_up(self, matrix: ScoreMatrix) -> Pairing:
        """Pair greedily by descending weight.

        Args:
            matrix: The population's score matrix.

        Returns:
            Row-index pairs.
        """
        edges = [
            (edge_weight(matrix, i, j), i, j)
            for i in range(matrix.size)
            for j in range(i + 1, matrix.size)
        ]
        edges.sort(key=lambda item: (-item[0], item[1], item[2]))
        taken: set[int] = set()
        pairs: Pairing = []
        for _, i, j in edges:
            if i in taken or j in taken:
                continue
            taken.update((i, j))
            pairs.append((i, j))
        return pairs


@register
class RandomAssignment(AssignmentStrategy):
    """Shuffle and pair neighbours. The floor any real method must beat."""

    key = "random"
    label = "Rastgele"
    description = "Tohumlanmış rastgele eşleştirme; taban çizgisi."

    seed: ClassVar[int] = 42

    def pair_up(self, matrix: ScoreMatrix) -> Pairing:
        """Pair students at random.

        Args:
            matrix: The population's score matrix.

        Returns:
            Row-index pairs.
        """
        order = list(range(matrix.size))
        random.Random(self.seed).shuffle(order)
        pairs: Pairing = []
        for index in range(0, len(order) - 1, 2):
            a, b = order[index], order[index + 1]
            pairs.append((min(a, b), max(a, b)))
        return pairs


@register
class StableAssignment(AssignmentStrategy):
    """Irving's stable-roommates algorithm.

    Registered for comparison, not as the default: it optimises **stability** (no
    two students in different rooms both prefer each other) rather than total
    compatibility, so it answers a different question. A stable matching is also not
    guaranteed to exist; when none does, this falls back to the greedy baseline and
    the plan says so.
    """

    key = "stable"
    label = "Kararlı (Irving)"
    description = "Kararlılığı hedefleyen klasik algoritma; toplam uyumu maksimize etmez."

    def pair_up(self, matrix: ScoreMatrix) -> Pairing:
        """Run Irving's algorithm, falling back to greedy when no solution exists.

        Args:
            matrix: The population's score matrix.

        Returns:
            Row-index pairs.
        """
        prefs = build_preference_lists(matrix)
        pairs, ok = stable_roommates(prefs)
        if not ok:
            logger.info("No stable matching exists for this population; using greedy instead")
            return GreedyAssignment().pair_up(matrix)
        return pairs


# --------------------------------------------------------------------------- #
# Irving's stable-roommates algorithm and its diagnostics
# --------------------------------------------------------------------------- #

Preferences = dict[int, list[int]]


def build_preference_lists(matrix: ScoreMatrix) -> Preferences:
    """Rank every student against every other student.

    Args:
        matrix: The population's score matrix.

    Returns:
        Mapping of row index to preference list, most preferred first.
    """
    ranking: Preferences = {}
    for i in range(matrix.size):
        scored = [(edge_weight(matrix, i, j), j) for j in range(matrix.size) if j != i]
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranking[i] = [index for _, index in scored]
    return ranking


def _phase_one(prefs: Preferences) -> tuple[dict[int, int], bool]:
    """Proposal phase: everyone proposes down their list until all hold an offer.

    Args:
        prefs: Preference lists, mutated in place as entries are removed.

    Returns:
        A tuple of the holder mapping and whether the phase succeeded.
    """
    holder: dict[int, int] = {}
    for proposer in list(prefs):
        current: int | None = proposer
        while current is not None:
            if not prefs[current]:
                return holder, False
            target = prefs[current][0]
            existing = holder.get(target)
            if existing is None:
                holder[target] = current
                current = None
            elif prefs[target].index(current) < prefs[target].index(existing):
                holder[target] = current
                current = existing
                prefs[current].pop(0)
            else:
                prefs[current].pop(0)
                if not prefs[current]:
                    return holder, False
    return holder, True


def _trim(prefs: Preferences, holder: dict[int, int]) -> None:
    """Remove pairs that can never be part of a stable matching.

    Anyone a student likes less than their current proposer is unreachable, so those
    entries are struck from both preference lists.

    Args:
        prefs: Preference lists, mutated in place.
        holder: Mapping of receiver to the proposer they currently hold.
    """
    for receiver, proposer in holder.items():
        keep = prefs[receiver][: prefs[receiver].index(proposer) + 1]
        for dropped in prefs[receiver][prefs[receiver].index(proposer) + 1 :]:
            if receiver in prefs[dropped]:
                prefs[dropped].remove(receiver)
        prefs[receiver] = keep


def _find_rotation(prefs: Preferences) -> list[tuple[int, int]] | None:
    """Locate an all-or-nothing cycle of preferences that must be eliminated.

    Args:
        prefs: The reduced preference lists.

    Returns:
        The rotation as ``(student, second_choice)`` pairs, or ``None`` if none exists.
    """
    start = next((p for p, lst in prefs.items() if len(lst) > 1), None)
    if start is None:
        return None
    sequence: list[int] = []
    seen: dict[int, int] = {}
    current = start
    while current not in seen:
        seen[current] = len(sequence)
        sequence.append(current)
        second = prefs[current][1]
        if not prefs[second]:
            return None
        current = prefs[second][-1]
    cycle = sequence[seen[current] :]
    return [(person, prefs[person][1]) for person in cycle]


def _eliminate(prefs: Preferences, rotation: list[tuple[int, int]]) -> bool:
    """Remove a rotation from the preference lists.

    Args:
        prefs: Preference lists, mutated in place.
        rotation: The rotation returned by :func:`_find_rotation`.

    Returns:
        ``False`` when elimination empties someone's list (no stable matching).
    """
    size = len(rotation)
    for index, (_, second) in enumerate(rotation):
        predecessor = rotation[(index - 1) % size][0]
        if predecessor in prefs[second]:
            cut = prefs[second].index(predecessor)
            for dropped in prefs[second][cut + 1 :]:
                if second in prefs[dropped]:
                    prefs[dropped].remove(second)
            prefs[second] = prefs[second][: cut + 1]
        if not prefs[second]:
            return False
    return all(prefs[person] for person in prefs)


def stable_roommates(prefs: Preferences) -> tuple[Pairing, bool]:
    """Run Irving's algorithm over complete preference lists.

    Args:
        prefs: Mapping of row index to preference list, most preferred first.

    Returns:
        A tuple of the pairs found and whether a stable matching exists. A ``False``
        flag is a legitimate outcome: unlike stable marriage, the roommates problem
        has no guaranteed solution.
    """
    people = list(prefs)
    if len(people) < 2:
        return [], True
    if len(people) % 2 == 1:
        # An odd student out cannot be paired; drop them and solve the rest.
        odd = people[-1]
        reduced = {p: [q for q in lst if q != odd] for p, lst in prefs.items() if p != odd}
        return stable_roommates(reduced)

    working: Preferences = {person: list(lst) for person, lst in prefs.items()}
    holder, ok = _phase_one(working)
    if not ok:
        return [], False
    _trim(working, holder)

    while True:
        if all(len(lst) == 1 for lst in working.values()):
            break
        if any(not lst for lst in working.values()):
            return [], False
        rotation = _find_rotation(working)
        if rotation is None:
            break
        if not _eliminate(working, rotation):
            return [], False

    pairs: set[tuple[int, int]] = set()
    for person, lst in working.items():
        if lst:
            pairs.add((min(person, lst[0]), max(person, lst[0])))
    return sorted(pairs), True


def find_blocking_pairs(pairs: Pairing, prefs: Preferences) -> list[tuple[int, int]]:
    """Verify an assignment by searching for pairs that would defect.

    A blocking pair is two students in different rooms who would both rather share
    with each other. An empty result proves the assignment is stable; a non-empty
    one is a diagnostic, not a failure, because the optimal-total plan is allowed to
    trade stability for a higher total.

    Args:
        pairs: The proposed assignment, as row-index pairs.
        prefs: The preference lists used to judge it.

    Returns:
        Every blocking pair found.
    """
    partner: dict[int, int] = {}
    for left, right in pairs:
        partner[left] = right
        partner[right] = left

    def prefers(person: int, other: int) -> bool:
        """Whether ``person`` would rather be with ``other`` than their partner."""
        current = partner.get(person)
        if current is None:
            return other in prefs.get(person, [])
        ranking = prefs.get(person, [])
        if other not in ranking or current not in ranking:
            return False
        return ranking.index(other) < ranking.index(current)

    people = sorted(prefs)
    return [
        (a, b)
        for i, a in enumerate(people)
        for b in people[i + 1 :]
        if partner.get(a) != b and prefers(a, b) and prefers(b, a)
    ]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class RoomAssigner:
    """Builds a room plan from a population.

    Args:
        strategy: Strategy key or instance. Defaults to the configured strategy.
        engine: The compatibility engine used to score pairings.
    """

    def __init__(
        self,
        strategy: str | AssignmentStrategy | None = None,
        engine: CompatibilityEngine | None = None,
    ) -> None:
        """Initialise the assigner."""
        settings = get_settings()
        if strategy is None:
            strategy = settings.default_strategy
        self.strategy: AssignmentStrategy = (
            get_strategy(strategy) if isinstance(strategy, str) else strategy
        )
        self.engine = engine or CompatibilityEngine()

    def assign(
        self,
        students: list[StudentProfile],
        *,
        explain: bool = True,
        check_stability: bool = True,
        matrix: ScoreMatrix | None = None,
    ) -> AssignmentPlan:
        """Partition the population into rooms.

        Args:
            students: Everyone who needs a bed.
            explain: Whether to attach per-dimension breakdowns to each room.
            check_stability: Whether to search the result for blocking pairs.
            matrix: A pre-computed score matrix, to avoid rebuilding it when
                several strategies are compared on the same population.

        Returns:
            The resulting :class:`AssignmentPlan`.
        """
        started = time.perf_counter()
        scores = matrix if matrix is not None else self.engine.score_matrix(students)

        if scores.size < 2:
            return AssignmentPlan(
                strategy=self.strategy.key,
                unplaced=tuple(s.student_id for s in scores.students),
                optimal=self.strategy.optimal,
                seconds=round(time.perf_counter() - started, 4),
            )

        pairs = self.strategy.pair_up(scores)
        rooms = self._build_rooms(scores, pairs, explain=explain)

        placed = {index for pair in pairs for index in pair}
        unplaced = tuple(
            scores.students[index].student_id for index in range(scores.size) if index not in placed
        )

        violations: Counter[str] = Counter()
        for first, second in pairs:
            for key in scores.violations_between(first, second):
                violations[key] += 1

        blocking: tuple[tuple[int, int], ...] = ()
        if check_stability:
            prefs = build_preference_lists(scores)
            blocking = tuple(
                (scores.students[a].student_id, scores.students[b].student_id)
                for a, b in find_blocking_pairs(pairs, prefs)
            )

        plan = AssignmentPlan(
            strategy=self.strategy.key,
            rooms=rooms,
            unplaced=unplaced,
            violations=dict(violations),
            blocking_pairs=blocking,
            seconds=round(time.perf_counter() - started, 4),
            optimal=self.strategy.optimal,
        )
        logger.info(
            "%s: %s rooms, total %.4f, %s unplaced, %s violations",
            self.strategy.key,
            plan.room_count,
            plan.total_score,
            len(plan.unplaced),
            plan.violation_count,
        )
        return plan

    def _build_rooms(
        self,
        scores: ScoreMatrix,
        pairs: Pairing,
        *,
        explain: bool,
    ) -> tuple[Room, ...]:
        """Turn index pairs into numbered rooms, best room first.

        Args:
            scores: The population's score matrix.
            pairs: The chosen pairings.
            explain: Whether to compute per-dimension breakdowns.

        Returns:
            Rooms numbered from 1, ordered by descending compatibility.
        """
        ordered = sorted(pairs, key=lambda pair: -scores.score_between(*pair))
        rooms: list[Room] = []
        for number, (first, second) in enumerate(ordered, start=1):
            left, right = scores.students[first], scores.students[second]
            score = scores.score_between(first, second)
            violations = scores.violations_between(first, second)
            if explain:
                pair_score = self.engine.explain_pair(left, right)
            else:
                pair_score = PairScore(
                    first_id=left.student_id,
                    second_id=right.student_id,
                    first_name=left.display_name,
                    second_name=right.display_name,
                    score=round(score, 6),
                    metric=self.engine.metric.key,
                    violations=violations,
                    summary=build_summary((), score, violations=violations),
                )
            rooms.append(Room(room_no=number, pair=pair_score))
        return tuple(rooms)


def compare_strategies(
    students: list[StudentProfile],
    *,
    engine: CompatibilityEngine | None = None,
    strategies: list[str] | None = None,
) -> list[AssignmentPlan]:
    """Run several strategies over one population and return every plan.

    The score matrix is built once and shared, so the comparison measures the
    strategies rather than the scoring.

    Args:
        students: The population to assign.
        engine: Optional engine override.
        strategies: Strategy keys; defaults to every registered strategy.

    Returns:
        One plan per strategy, best total score first.
    """
    engine = engine or CompatibilityEngine()
    matrix = engine.score_matrix(students)
    keys = strategies or available_strategies()
    plans = [
        RoomAssigner(key, engine).assign(students, explain=False, matrix=matrix) for key in keys
    ]
    return sorted(plans, key=lambda plan: -plan.total_score)


def optimality_gap(plan: AssignmentPlan, best: AssignmentPlan) -> float:
    """How far a plan falls short of the best total score, as a percentage.

    Args:
        plan: The plan being measured.
        best: The plan with the highest total score.

    Returns:
        The shortfall in percent; ``0.0`` when the plan matches the best.
    """
    if best.total_score <= 0:
        return 0.0
    return round((1 - plan.total_score / best.total_score) * 100, 2)
