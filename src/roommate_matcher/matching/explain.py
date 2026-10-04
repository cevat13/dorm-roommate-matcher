"""Rendering scores as explanations.

Converts the per-dimension contributions behind a score into readable Turkish text
and into rows suitable for a table or a bar chart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from roommate_matcher.domain.enums import MatchVerdict
from roommate_matcher.domain.models import FeatureContribution, PairScore


def build_summary(
    contributions: Sequence[FeatureContribution],
    score: float,
    *,
    violations: Sequence[str] = (),
    max_positives: int = 3,
    max_concerns: int = 2,
) -> str:
    """Compose a Turkish explanation of a pairing.

    Args:
        contributions: The per-dimension breakdown behind the score.
        score: The final score in ``[0, 1]``.
        violations: Constraint violations that had to be accepted, if any.
        max_positives: How many strengths to mention.
        max_concerns: How many concerns to mention.

    Returns:
        A one-paragraph explanation.
    """
    headline = f"Uyum skoru %{round(score * 100)}."
    if not contributions:
        return headline

    positives = sorted(
        (c for c in contributions if c.verdict is MatchVerdict.MATCH),
        key=lambda c: c.weighted_contribution,
        reverse=True,
    )[:max_positives]
    concerns = sorted(
        (c for c in contributions if c.verdict is MatchVerdict.CLASH),
        key=lambda c: c.weight * (1 - c.similarity),
        reverse=True,
    )[:max_concerns]

    sentences = [headline]
    if positives:
        strengths = "; ".join(f"{c.label} ({c.first_value} / {c.second_value})" for c in positives)
        sentences.append(f"Güçlü yanlar: {strengths}.")
    if concerns:
        issues = "; ".join(f"{c.label} ({c.first_value} / {c.second_value})" for c in concerns)
        sentences.append(f"Dikkat edilmesi gerekenler: {issues}.")
    if not positives and not concerns:
        sentences.append("Belirgin bir uyum ya da çatışma noktası öne çıkmıyor.")
    if violations:
        from roommate_matcher.matching.constraints import VIOLATION_LABELS

        named = ", ".join(VIOLATION_LABELS.get(key, key) for key in violations)
        sentences.append(f"Kaçınılamayan kısıt ihlali: {named}.")

    return " ".join(sentences)


@dataclass(frozen=True)
class ImpactRow:
    """One dimension's impact on a score, ready for a table or a bar chart.

    Typed rather than a loose dict so the CLI table, the charts and the DataFrame
    all read the same attributes.
    """

    feature: str
    label: str
    similarity: float
    weight: float
    share_of_score: float
    max_possible_share: float
    lost_points: float
    verdict: MatchVerdict
    first_value: str
    second_value: str

    def as_dict(self) -> dict[str, float | str]:
        """Flatten for a DataFrame.

        Returns:
            A mapping with the verdict rendered as its string value.
        """
        return {
            "feature": self.feature,
            "label": self.label,
            "similarity": self.similarity,
            "weight": self.weight,
            "share_of_score": self.share_of_score,
            "max_possible_share": self.max_possible_share,
            "lost_points": self.lost_points,
            "verdict": self.verdict.value,
            "first_value": self.first_value,
            "second_value": self.second_value,
        }


def impact_table(contributions: Sequence[FeatureContribution]) -> list[ImpactRow]:
    """Per-dimension impact rows, sorted worst-first.

    Args:
        contributions: The per-dimension breakdown.

    Returns:
        Rows sorted by how many points each dimension cost, biggest loss first.
    """
    total_weight = sum(c.weight for c in contributions) or 1.0
    rows = [
        ImpactRow(
            feature=c.feature,
            label=c.label,
            similarity=c.similarity,
            weight=c.weight,
            share_of_score=round(c.similarity * c.weight / total_weight, 4),
            max_possible_share=round(c.weight / total_weight, 4),
            lost_points=round((1 - c.similarity) * c.weight / total_weight, 4),
            verdict=c.verdict,
            first_value=c.first_value,
            second_value=c.second_value,
        )
        for c in contributions
    ]
    return sorted(rows, key=lambda row: row.lost_points, reverse=True)


def compare(pairs: Sequence[PairScore]) -> str:
    """Explain how the top candidates differ from one another.

    Args:
        pairs: Ranked pair scores, best first.

    Returns:
        A short comparison, or a fallback message when there is nothing to compare.
    """
    if len(pairs) < 2:
        return "Karşılaştırma için en az iki aday gerekiyor."
    best, runner_up = pairs[0], pairs[1]
    by_feature = {c.feature: c for c in runner_up.contributions}
    gaps = [
        (c.label, c.similarity - by_feature[c.feature].similarity)
        for c in best.contributions
        if c.feature in by_feature
    ]
    gaps.sort(key=lambda item: abs(item[1]), reverse=True)
    biggest = ", ".join(f"{label} ({delta:+.2f})" for label, delta in gaps[:3])
    return (
        f"{best.second_name} (%{best.percentage}) ile {runner_up.second_name} "
        f"(%{runner_up.percentage}) arasındaki en belirgin farklar: {biggest}."
    )
