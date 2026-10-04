"""How much each dimension matters, and what counts as a constraint violation.

Smoking and sleep schedule break a shared room while a shared taste in films does
not, so per-dimension importance is modelled explicitly as weights, and the
absolute requirements as separate constraints.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from roommate_matcher.domain.enums import SmokingHabit

DEFAULT_WEIGHTS: dict[str, float] = {
    # Habits that make a shared room unliveable carry the most weight.
    "smoking": 3.0,
    "sleep_schedule": 2.5,
    "cleanliness": 2.5,
    "study_location": 2.0,
    "noise_tolerance": 2.0,
    "study_time": 1.5,
    "room_time": 1.2,
    "guest_frequency": 1.2,
    "interests": 1.2,
    "social_energy": 1.0,
    "alcohol": 1.0,
    "department": 0.8,
    "study_year": 0.8,
    "languages": 0.5,
    "diet": 0.5,
}
"""Default importance of each scoring dimension. Relative magnitudes are what matter."""


class Constraints(BaseModel):
    """Requirements that a pairing should not violate.

    When ranking candidates these act as filters and remove candidates outright.
    When assigning rooms every student must get a bed, so a filter could make the
    problem infeasible; there the violations become large score penalties instead
    and are reported. See :mod:`roommate_matcher.matching.assignment`.
    """

    model_config = ConfigDict(frozen=True)

    separate_smokers: bool = Field(
        default=True,
        description="Treat a non-smoker paired with an indoor smoker as a violation.",
    )
    max_smoking: SmokingHabit | None = Field(
        default=None,
        description="Most intrusive smoking habit tolerated at all; None means no limit.",
    )
    enforce_allergies: bool = Field(
        default=True,
        description="Treat a smoker paired with a smoke-allergic student as a violation.",
    )
    max_year_gap: int | None = Field(
        default=None,
        ge=0,
        le=5,
        description="Maximum acceptable gap in study year; None disables the check.",
    )

    @classmethod
    def permissive(cls) -> Self:
        """Constraints that flag nothing, useful for evaluation and benchmarking.

        Returns:
            A :class:`Constraints` instance with every check disabled.
        """
        return cls(
            separate_smokers=False,
            max_smoking=None,
            enforce_allergies=False,
            max_year_gap=None,
        )


class MatchPreferences(BaseModel):
    """Weights plus constraints: the full description of a matching run."""

    model_config = ConfigDict(frozen=True)

    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    constraints: Constraints = Field(default_factory=Constraints)
    same_department_bonus: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Score given to two students from the same department. 1.0 treats it as a"
            " perfect match; 0.5 as neutral; a low value when the institution"
            " mixes departments by policy."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _complete_weights(cls, data: Any) -> Any:
        """Validate the supplied weights and fill in the missing dimensions.

        A partial dict overrides the defaults and keeps the rest, so the stored
        mapping always covers every dimension and a dimension is excluded only by
        an explicit zero.

        Args:
            data: Raw input, normally a mapping of field names to values.

        Returns:
            The input with a complete ``weights`` mapping.

        Raises:
            ValueError: If a key is unknown, a value is negative, or every weight
                is zero.
        """
        if not isinstance(data, dict):
            return data
        supplied = data.get("weights")
        if supplied is None:
            return data
        unknown = set(supplied) - set(DEFAULT_WEIGHTS)
        if unknown:
            msg = f"Unknown weight keys: {sorted(unknown)}. Valid keys: {sorted(DEFAULT_WEIGHTS)}."
            raise ValueError(msg)
        negative = [key for key, value in supplied.items() if value < 0]
        if negative:
            msg = f"Weights must be non-negative; got negative values for {sorted(negative)}."
            raise ValueError(msg)
        completed = {**DEFAULT_WEIGHTS, **supplied}
        if sum(completed.values()) <= 0:
            msg = "At least one weight must be greater than zero."
            raise ValueError(msg)
        return {**data, "weights": completed}

    def weight_for(self, feature: str) -> float:
        """Weight of a single scoring dimension.

        Args:
            feature: Dimension key, e.g. ``"smoking"``.

        Returns:
            The configured weight; ``0.0`` for an unknown dimension.
        """
        return self.weights.get(feature, 0.0)

    def normalised_weights(self) -> dict[str, float]:
        """Weights rescaled to sum to 1, which keeps final scores inside ``[0, 1]``.

        Returns:
            A mapping of dimension to normalised weight.
        """
        total = sum(self.weights.values())
        return {key: value / total for key, value in self.weights.items()}

    def emphasise(self, feature: str, factor: float) -> MatchPreferences:
        """Return a copy with one dimension scaled up or down.

        Args:
            feature: Dimension key to adjust.
            factor: Multiplier applied to the current weight.

        Returns:
            A new :class:`MatchPreferences` instance.
        """
        updated = dict(self.weights)
        updated[feature] = self.weight_for(feature) * factor
        return self.model_copy(update={"weights": updated})
