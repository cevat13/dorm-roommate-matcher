"""Compatibility metrics behind a common interface.

Adding a metric means adding one class and one ``@register`` decorator. The purely
geometric metrics are kept alongside the weighted one so that
:mod:`roommate_matcher.matching.evaluation` can compare them.

Every metric here is symmetric: ``score(a, b) == score(b, a)``. Sharing a room is
not a one-sided arrangement, and the dimension scorers reflect that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from roommate_matcher.domain.models import FeatureContribution, StudentProfile
from roommate_matcher.domain.preferences import MatchPreferences
from roommate_matcher.exceptions import UnknownMetricError
from roommate_matcher.features.interests import jaccard
from roommate_matcher.features.vectorizer import FloatArray, encode
from roommate_matcher.matching.dimensions import score_dimensions

_REGISTRY: dict[str, type[CompatibilityMetric]] = {}


def register(cls: type[CompatibilityMetric]) -> type[CompatibilityMetric]:
    """Register a metric class under its ``key``.

    Args:
        cls: The metric class to register.

    Returns:
        The same class, so this can be used as a decorator.
    """
    _REGISTRY[cls.key] = cls
    return cls


def available_metrics() -> list[str]:
    """Keys of every registered metric.

    Returns:
        Sorted metric keys.
    """
    return sorted(_REGISTRY)


def get_metric(key: str) -> CompatibilityMetric:
    """Instantiate a metric by key.

    Args:
        key: The metric key, e.g. ``"weighted_gower"``.

    Returns:
        A ready-to-use metric instance.

    Raises:
        UnknownMetricError: If no metric is registered under that key.
    """
    try:
        return _REGISTRY[key]()
    except KeyError:
        raise UnknownMetricError(key, available_metrics()) from None


class CompatibilityMetric(ABC):
    """Strategy interface: turn two students into a score in ``[0, 1]``."""

    key: ClassVar[str] = "base"
    label: ClassVar[str] = "Base"
    description: ClassVar[str] = ""
    explainable: ClassVar[bool] = False

    @abstractmethod
    def score(
        self,
        first: StudentProfile,
        second: StudentProfile,
        preferences: MatchPreferences,
    ) -> float:
        """Score one pair.

        Args:
            first: One student.
            second: The other student.
            preferences: Weights and policies for the run.

        Returns:
            A compatibility in ``[0, 1]``, where 1 means perfectly compatible.
        """

    def contributions(
        self,
        first: StudentProfile,
        second: StudentProfile,
        preferences: MatchPreferences,
    ) -> tuple[FeatureContribution, ...]:
        """Per-dimension breakdown behind the score.

        Args:
            first: One student.
            second: The other student.
            preferences: Weights for the run.

        Returns:
            The contributions, or an empty tuple for metrics that cannot explain
            themselves (the purely geometric ones).
        """
        del first, second, preferences
        return ()

    def score_many(
        self,
        student: StudentProfile,
        others: Sequence[StudentProfile],
        preferences: MatchPreferences,
    ) -> FloatArray:
        """Score a whole candidate pool.

        Vectorised metrics override this to avoid a Python-level loop.

        Args:
            student: The student being compared.
            others: The candidate pool.
            preferences: Weights for the run.

        Returns:
            One score per candidate, aligned with the input order.
        """
        return np.asarray(
            [self.score(student, other, preferences) for other in others],
            dtype=np.float64,
        )


@register
class WeightedGowerMetric(CompatibilityMetric):
    """Weighted Gower compatibility over mixed-type attributes; the default metric.

    Each dimension contributes its own ``[0, 1]`` compatibility score and the result
    is their weighted mean. Numeric, ordinal, categorical and set-valued attributes
    are all handled in the same model, and the per-dimension scores double as the
    explanation.
    """

    key = "weighted_gower"
    label = "Ağırlıklı Gower"
    description = "Karışık tipli veriler için ağırlıklı uyumluluk skoru (varsayılan)."
    explainable = True

    def score(
        self,
        first: StudentProfile,
        second: StudentProfile,
        preferences: MatchPreferences,
    ) -> float:
        """Weighted mean of the per-dimension compatibility scores.

        Args:
            first: One student.
            second: The other student.
            preferences: Weights for the run.

        Returns:
            A score in ``[0, 1]``.
        """
        parts = self.contributions(first, second, preferences)
        if not parts:
            return 0.0
        total_weight = sum(part.weight for part in parts)
        if total_weight <= 0:
            return 0.0
        weighted = sum(part.similarity * part.weight for part in parts)
        return float(weighted / total_weight)

    def contributions(
        self,
        first: StudentProfile,
        second: StudentProfile,
        preferences: MatchPreferences,
    ) -> tuple[FeatureContribution, ...]:
        """Per-dimension breakdown.

        Args:
            first: One student.
            second: The other student.
            preferences: Weights for the run.

        Returns:
            One contribution per weighted dimension.
        """
        return score_dimensions(first, second, preferences)


class _VectorMetric(CompatibilityMetric):
    """Base class for metrics that operate on the scaled numeric vector space."""

    def _pairwise(self, left: FloatArray, right: FloatArray) -> float:
        """Compatibility between two encoded vectors.

        Args:
            left: First encoded student.
            right: Second encoded student.

        Returns:
            A score in ``[0, 1]``.
        """
        raise NotImplementedError

    def score(
        self,
        first: StudentProfile,
        second: StudentProfile,
        preferences: MatchPreferences,
    ) -> float:
        """Encode both students and compare them geometrically.

        Args:
            first: One student.
            second: The other student.
            preferences: Unused; geometric metrics ignore weights.

        Returns:
            A score in ``[0, 1]``.
        """
        del preferences
        return self._pairwise(encode(first), encode(second))

    def score_many(
        self,
        student: StudentProfile,
        others: Sequence[StudentProfile],
        preferences: MatchPreferences,
    ) -> FloatArray:
        """Score the pool with a single vectorised pass.

        Args:
            student: The student being compared.
            others: The candidate pool.
            preferences: Unused.

        Returns:
            One score per candidate.
        """
        del preferences
        if not others:
            return np.empty(0, dtype=np.float64)
        target = encode(student)
        matrix = np.vstack([encode(other) for other in others])
        return self._batch(target, matrix)

    def _batch(self, target: FloatArray, matrix: FloatArray) -> FloatArray:
        """Vectorised scoring against a matrix.

        Args:
            target: The encoded vector of the student being compared.
            matrix: Encoded candidate matrix.

        Returns:
            One score per row of ``matrix``.
        """
        return np.asarray([self._pairwise(target, row) for row in matrix], dtype=np.float64)


@register
class CosineMetric(_VectorMetric):
    """Cosine similarity on the scaled vector space.

    A poor fit for this data: cosine measures angle, so a binary attribute both
    students answered "no" to adds nothing to their similarity. Kept as a baseline.
    """

    key = "cosine"
    label = "Kosinüs"
    description = "Ölçeklenmiş vektör uzayında kosinüs benzerliği (referans metrik)."

    def _pairwise(self, left: FloatArray, right: FloatArray) -> float:
        """Cosine similarity mapped from ``[-1, 1]`` into ``[0, 1]``.

        Args:
            left: First encoded student.
            right: Second encoded student.

        Returns:
            A score in ``[0, 1]``.
        """
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0.0:
            return 0.0
        return float(np.clip((left @ right) / denominator, -1.0, 1.0) * 0.5 + 0.5)

    def _batch(self, target: FloatArray, matrix: FloatArray) -> FloatArray:
        """Vectorised cosine similarity.

        Args:
            target: The encoded vector of the student being compared.
            matrix: Encoded candidate matrix.

        Returns:
            One score per row.
        """
        norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(target))
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(norms > 0, (matrix @ target) / norms, 0.0)
        return np.clip(raw, -1.0, 1.0) * 0.5 + 0.5


@register
class EuclideanMetric(_VectorMetric):
    """Euclidean distance on the scaled vector space, converted to a similarity.

    Every column is pre-scaled to ``[0, 1]``, so no single wide-ranged attribute
    dominates the distance.
    """

    key = "euclidean"
    label = "Öklid"
    description = "Ölçeklenmiş uzayda Öklid mesafesinin benzerliğe çevrilmiş hali."

    def _pairwise(self, left: FloatArray, right: FloatArray) -> float:
        """Convert distance to similarity with ``1 / (1 + d)``.

        Args:
            left: First encoded student.
            right: Second encoded student.

        Returns:
            A score in ``(0, 1]``.
        """
        return float(1.0 / (1.0 + np.linalg.norm(left - right)))

    def _batch(self, target: FloatArray, matrix: FloatArray) -> FloatArray:
        """Vectorised distance-to-similarity conversion.

        Args:
            target: The encoded vector of the student being compared.
            matrix: Encoded candidate matrix.

        Returns:
            One score per row.
        """
        return 1.0 / (1.0 + np.linalg.norm(matrix - target, axis=1))


@register
class ManhattanMetric(_VectorMetric):
    """Manhattan (L1) distance converted to a similarity.

    More robust than L2 on the many one-hot columns this feature space contains.
    """

    key = "manhattan"
    label = "Manhattan"
    description = "L1 mesafesinin benzerliğe çevrilmiş hali; one-hot sütunlara dayanıklı."

    def _pairwise(self, left: FloatArray, right: FloatArray) -> float:
        """Convert L1 distance to similarity.

        Args:
            left: First encoded student.
            right: Second encoded student.

        Returns:
            A score in ``(0, 1]``.
        """
        return float(1.0 / (1.0 + np.abs(left - right).sum()))

    def _batch(self, target: FloatArray, matrix: FloatArray) -> FloatArray:
        """Vectorised L1 scoring.

        Args:
            target: The encoded vector of the student being compared.
            matrix: Encoded candidate matrix.

        Returns:
            One score per row.
        """
        return 1.0 / (1.0 + np.abs(matrix - target).sum(axis=1))


@register
class JaccardInterestsMetric(CompatibilityMetric):
    """Interest-set overlap only; a narrow baseline."""

    key = "jaccard_interests"
    label = "Jaccard (ilgi alanları)"
    description = "Yalnızca ortak ilgi alanlarına bakan sade referans metrik."

    def score(
        self,
        first: StudentProfile,
        second: StudentProfile,
        preferences: MatchPreferences,
    ) -> float:
        """Jaccard similarity of the two interest sets.

        Args:
            first: One student.
            second: The other student.
            preferences: Unused.

        Returns:
            A score in ``[0, 1]``.
        """
        del preferences
        return jaccard(first.interests, second.interests)


@register
class HybridMetric(CompatibilityMetric):
    """Blend of weighted Gower and scaled Euclidean.

    Gower captures rule-based compatibility, Euclidean overall profile proximity;
    blending them softens the step changes in the lookup matrices.
    """

    key = "hybrid"
    label = "Hibrit"
    description = "Ağırlıklı Gower ve ölçeklenmiş Öklid skorlarının ağırlıklı ortalaması."
    explainable = True

    gower_share: ClassVar[float] = 0.7

    def __init__(self) -> None:
        """Compose the two underlying strategies."""
        self._gower = WeightedGowerMetric()
        self._euclidean = EuclideanMetric()

    def score(
        self,
        first: StudentProfile,
        second: StudentProfile,
        preferences: MatchPreferences,
    ) -> float:
        """Weighted blend of the two component scores.

        Args:
            first: One student.
            second: The other student.
            preferences: Weights, passed through to the Gower component.

        Returns:
            A score in ``[0, 1]``.
        """
        gower = self._gower.score(first, second, preferences)
        euclid = self._euclidean.score(first, second, preferences)
        return self.gower_share * gower + (1 - self.gower_share) * euclid

    def contributions(
        self,
        first: StudentProfile,
        second: StudentProfile,
        preferences: MatchPreferences,
    ) -> tuple[FeatureContribution, ...]:
        """Reuse the Gower breakdown for explanations.

        Args:
            first: One student.
            second: The other student.
            preferences: Weights for the run.

        Returns:
            The Gower component's contributions.
        """
        return self._gower.contributions(first, second, preferences)
