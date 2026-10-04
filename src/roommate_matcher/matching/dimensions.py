"""Per-dimension compatibility scoring (Gower formulation for mixed data).

Every dimension defines its own compatibility function returning ``[0, 1]``, which
handles numeric, ordinal, categorical and set-valued attributes in one model and
lets each dimension report its own score for the explanation layer.

These measure compatibility rather than similarity: see ``score_cleanliness`` and
``score_study_location`` for the cases where the two diverge.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from roommate_matcher.domain.enums import (
    AlcoholHabit,
    Allergy,
    DietType,
    GuestFrequency,
    MatchVerdict,
    SleepSchedule,
    SmokingHabit,
    StudyLocation,
    StudyTime,
)
from roommate_matcher.domain.models import MAX_WEEKDAY_EVENINGS, FeatureContribution, StudentProfile
from roommate_matcher.domain.preferences import MatchPreferences
from roommate_matcher.features.interests import jaccard, labels

ScoreFn = Callable[[StudentProfile, StudentProfile], float]
DescribeFn = Callable[[StudentProfile], str]

# Study-year gap at which the score decays to roughly 37%.
YEAR_DECAY = 2.5


def _symmetric_lookup[T](
    matrix: dict[tuple[T, T], float],
    left: T,
    right: T,
    *,
    default: float,
) -> float:
    """Look a pair up in a half-filled matrix, trying both orderings.

    Membership is tested rather than truthiness so that a stored 0.0 is returned
    instead of falling through to the default.

    Args:
        matrix: Pair-keyed scores, populated in one direction only.
        left: First value.
        right: Second value.
        default: Returned when neither ordering is present.

    Returns:
        The stored score, or ``default``.
    """
    if (left, right) in matrix:
        return matrix[(left, right)]
    if (right, left) in matrix:
        return matrix[(right, left)]
    return default


def _ordinal_similarity(left: int, right: int, span: int) -> float:
    """Linear similarity between two ordinal levels.

    Args:
        left: First level.
        right: Second level.
        span: Maximum possible gap between levels.

    Returns:
        ``1 - |left - right| / span`` clamped to ``[0, 1]``.
    """
    if span <= 0:
        return 1.0
    return max(0.0, 1.0 - abs(left - right) / span)


# --------------------------------------------------------------------------- #
# Individual dimension scorers
# --------------------------------------------------------------------------- #

_SLEEP_MATRIX: dict[tuple[SleepSchedule, SleepSchedule], float] = {
    (SleepSchedule.EARLY, SleepSchedule.EARLY): 1.0,
    (SleepSchedule.LATE, SleepSchedule.LATE): 1.0,
    (SleepSchedule.FLEXIBLE, SleepSchedule.FLEXIBLE): 1.0,
    (SleepSchedule.EARLY, SleepSchedule.FLEXIBLE): 0.8,
    (SleepSchedule.LATE, SleepSchedule.FLEXIBLE): 0.8,
    (SleepSchedule.EARLY, SleepSchedule.LATE): 0.1,
}


def score_sleep(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of sleep schedules.

    Sharing one room means sharing one light switch, so an early riser paired with
    a night owl is the single most common complaint in a dormitory.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``; early paired with late scores 0.1.
    """
    return _symmetric_lookup(
        _SLEEP_MATRIX, first.sleep_schedule, second.sleep_schedule, default=0.1
    )


def score_cleanliness(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of cleanliness standards.

    Close standards score highly, but two students who are both very untidy are
    penalised: they agree with each other and the room is still unliveable.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    base = _ordinal_similarity(first.cleanliness, second.cleanliness, span=4)
    if first.cleanliness <= 2 and second.cleanliness <= 2:
        base *= 0.7
    return base


_SMOKING_MATRIX: dict[tuple[SmokingHabit, SmokingHabit], float] = {
    (SmokingHabit.NONE, SmokingHabit.NONE): 1.0,
    (SmokingHabit.NONE, SmokingHabit.OUTDOOR_ONLY): 0.65,
    (SmokingHabit.NONE, SmokingHabit.INDOOR): 0.0,
    (SmokingHabit.OUTDOOR_ONLY, SmokingHabit.OUTDOOR_ONLY): 1.0,
    (SmokingHabit.OUTDOOR_ONLY, SmokingHabit.INDOOR): 0.4,
    (SmokingHabit.INDOOR, SmokingHabit.INDOOR): 1.0,
}


def score_smoking(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of smoking habits.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``. A non-smoker paired with an indoor smoker scores 0.
    """
    return _symmetric_lookup(_SMOKING_MATRIX, first.smoking, second.smoking, default=0.0)


_STUDY_LOCATION_MATRIX: dict[tuple[StudyLocation, StudyLocation], float] = {
    # Two students who both work at the room desk compete for the same quiet hours.
    (StudyLocation.ROOM, StudyLocation.ROOM): 0.45,
    (StudyLocation.ROOM, StudyLocation.MIXED): 0.7,
    (StudyLocation.ROOM, StudyLocation.LIBRARY): 1.0,
    (StudyLocation.MIXED, StudyLocation.MIXED): 0.8,
    (StudyLocation.MIXED, StudyLocation.LIBRARY): 0.95,
    (StudyLocation.LIBRARY, StudyLocation.LIBRARY): 1.0,
}


def score_study_location(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of where each student studies.

    Unlike most dimensions, agreement is not rewarded here: two students who both
    study at the room desk need the room quiet at the same time and share one
    table, so that pairing scores worst. One room-studier and one library-goer is
    the ideal arrangement.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    return _symmetric_lookup(
        _STUDY_LOCATION_MATRIX, first.study_location, second.study_location, default=0.7
    )


_STUDY_TIME_ORDER: dict[StudyTime, int] = {
    StudyTime.DAYTIME: 0,
    StudyTime.EVENING: 1,
    StudyTime.NIGHT: 2,
}


def score_study_time(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of study hours.

    Shared hours mean the room is quiet at the same time, which both students need.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    return _ordinal_similarity(
        _STUDY_TIME_ORDER[first.study_time], _STUDY_TIME_ORDER[second.study_time], span=2
    )


def score_noise(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of noise tolerance.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    return _ordinal_similarity(first.noise_tolerance, second.noise_tolerance, span=4)


_GUEST_ORDER: dict[GuestFrequency, int] = {
    GuestFrequency.RARELY: 0,
    GuestFrequency.SOMETIMES: 1,
    GuestFrequency.OFTEN: 2,
}


def score_guests(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of how often guests are hosted.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    return _ordinal_similarity(
        _GUEST_ORDER[first.guest_frequency], _GUEST_ORDER[second.guest_frequency], span=2
    )


def score_room_time(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of how much time each student spends in the room.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    return _ordinal_similarity(
        first.room_time_weekdays, second.room_time_weekdays, span=MAX_WEEKDAY_EVENINGS
    )


def score_social_energy(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of how socially active each student is in the room.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    return _ordinal_similarity(first.social_energy, second.social_energy, span=4)


_ALCOHOL_ORDER: dict[AlcoholHabit, int] = {
    AlcoholHabit.NONE: 0,
    AlcoholHabit.OCCASIONAL: 1,
    AlcoholHabit.REGULAR: 2,
}


def score_alcohol(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of drinking habits.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    return _ordinal_similarity(
        _ALCOHOL_ORDER[first.alcohol], _ALCOHOL_ORDER[second.alcohol], span=2
    )


_DIET_AFFINITY: dict[tuple[DietType, DietType], float] = {
    (DietType.VEGAN, DietType.VEGETARIAN): 0.85,
    (DietType.VEGAN, DietType.OMNIVORE): 0.45,
    (DietType.VEGETARIAN, DietType.OMNIVORE): 0.7,
    (DietType.HALAL, DietType.OMNIVORE): 0.6,
    (DietType.HALAL, DietType.VEGETARIAN): 0.8,
    (DietType.HALAL, DietType.VEGAN): 0.8,
    (DietType.GLUTEN_FREE, DietType.OMNIVORE): 0.8,
}


def score_diet(first: StudentProfile, second: StudentProfile) -> float:
    """Compatibility of dietary patterns, which matters for the shared fridge.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    if first.diet is second.diet:
        return 1.0
    return _symmetric_lookup(_DIET_AFFINITY, first.diet, second.diet, default=0.7)


def score_study_year(first: StudentProfile, second: StudentProfile) -> float:
    """Closeness in study year, decaying exponentially with the gap.

    Students in the same year share a timetable, an exam calendar and a workload
    rhythm, which matters more in a dormitory than raw age does.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    return math.exp(-abs(first.study_year - second.study_year) / YEAR_DECAY)


def score_languages(first: StudentProfile, second: StudentProfile) -> float:
    """Shared-language score.

    Any overlap earns a floor of 0.8; full overlap earns 1.0.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    shared = first.languages & second.languages
    if not shared:
        return 0.0
    union = first.languages | second.languages
    return 0.8 + 0.2 * (len(shared) / len(union))


def score_interests(first: StudentProfile, second: StudentProfile) -> float:
    """Jaccard similarity over the normalised interest taxonomy.

    Two empty sets score 1.0 rather than the 0.0 plain Jaccard would give: with no
    interests declared on either side there is nothing to disagree about.

    Args:
        first: One student.
        second: The other student.

    Returns:
        A score in ``[0, 1]``.
    """
    if not first.interests and not second.interests:
        return 1.0
    return jaccard(first.interests, second.interests)


# --------------------------------------------------------------------------- #
# Dimension registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Dimension:
    """A scoring dimension: its weight key, scorer, and display strings."""

    key: str
    label: str
    score: ScoreFn
    describe: DescribeFn
    positive: str
    negative: str

    def explain(self, similarity: float) -> str:
        """Render the Turkish explanation for a computed similarity.

        Args:
            similarity: The score this dimension produced.

        Returns:
            The positive phrasing above 0.6, otherwise the negative phrasing.
        """
        return self.positive if similarity >= 0.6 else self.negative


def _score_department(first: StudentProfile, second: StudentProfile) -> float:
    """Placeholder replaced at scoring time by the configured department policy.

    Args:
        first: One student.
        second: The other student.

    Returns:
        1.0 for the same department, 0.5 otherwise.
    """
    return 1.0 if first.department is second.department else 0.5


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        "smoking",
        "Sigara",
        score_smoking,
        lambda s: s.smoking.label_tr,
        "Sigara konusunda uyumlusunuz.",
        "Sigara alışkanlıklarınız çatışıyor.",
    ),
    Dimension(
        "sleep_schedule",
        "Uyku düzeni",
        score_sleep,
        lambda s: s.sleep_schedule.label_tr,
        "Uyku düzeniniz birbirine uyuyor.",
        "Biriniz erken yatarken diğeriniz gece kuşu.",
    ),
    Dimension(
        "cleanliness",
        "Temizlik",
        score_cleanliness,
        lambda s: f"{s.cleanliness}/5",
        "Temizlik standartlarınız örtüşüyor.",
        "Temizlik beklentileriniz farklı.",
    ),
    Dimension(
        "study_location",
        "Ders çalışma yeri",
        score_study_location,
        lambda s: s.study_location.label_tr,
        "Ders çalışma yerleriniz birbirini engellemiyor.",
        "İkiniz de odada çalışıyorsunuz; aynı masa ve aynı sessizlik gerekiyor.",
    ),
    Dimension(
        "noise_tolerance",
        "Gürültü toleransı",
        score_noise,
        lambda s: f"{s.noise_tolerance}/5",
        "Gürültü toleransınız benzer.",
        "Gürültü toleransınız farklı.",
    ),
    Dimension(
        "study_time",
        "Çalışma saatleri",
        score_study_time,
        lambda s: s.study_time.label_tr,
        "Aynı saatlerde ders çalışıyorsunuz.",
        "Ders çalışma saatleriniz çakışmıyor; biri çalışırken diğeri dinleniyor.",
    ),
    Dimension(
        "room_time",
        "Odada geçirilen zaman",
        score_room_time,
        lambda s: f"Haftada {s.room_time_weekdays} akşam",
        "Odada benzer sıklıkta bulunuyorsunuz.",
        "Odada bulunma sıklığınız farklı.",
    ),
    Dimension(
        "guest_frequency",
        "Misafir sıklığı",
        score_guests,
        lambda s: s.guest_frequency.label_tr,
        "Misafir ağırlama alışkanlığınız benzer.",
        "Misafir ağırlama alışkanlığınız farklı.",
    ),
    Dimension(
        "interests",
        "İlgi alanları",
        score_interests,
        lambda s: labels(s.interests),
        "Ortak ilgi alanlarınız var.",
        "Ortak ilgi alanınız az.",
    ),
    Dimension(
        "social_energy",
        "Sosyallik",
        score_social_energy,
        lambda s: f"{s.social_energy}/5",
        "Sosyallik seviyeniz benzer.",
        "Sosyallik seviyeniz farklı.",
    ),
    Dimension(
        "alcohol",
        "Alkol",
        score_alcohol,
        lambda s: s.alcohol.label_tr,
        "Alkol alışkanlıklarınız uyumlu.",
        "Alkol alışkanlıklarınız farklı.",
    ),
    Dimension(
        "department",
        "Bölüm",
        _score_department,
        lambda s: s.department.label_tr,
        "Bölüm açısından uyumlu bir eşleşme.",
        "Bölümleriniz farklı.",
    ),
    Dimension(
        "study_year",
        "Sınıf",
        score_study_year,
        lambda s: f"{s.study_year}. sınıf",
        "Aynı veya yakın sınıftasınız.",
        "Aranızda belirgin bir sınıf farkı var.",
    ),
    Dimension(
        "languages",
        "Ortak dil",
        score_languages,
        lambda s: ", ".join(sorted(lang.label_tr for lang in s.languages)),
        "Ortak konuştuğunuz bir dil var.",
        "Ortak bir diliniz yok.",
    ),
    Dimension(
        "diet",
        "Beslenme",
        score_diet,
        lambda s: s.diet.label_tr,
        "Ortak buzdolabında sorun çıkmaz.",
        "Beslenme tercihleriniz farklı.",
    ),
)
"""All scoring dimensions, in roughly descending order of default weight."""

DIMENSIONS_BY_KEY: dict[str, Dimension] = {dim.key: dim for dim in DIMENSIONS}

SMOKE_ALLERGY = Allergy.SMOKE
"""Re-exported so the constraint layer does not need the enum import."""


def verdict_for(similarity: float, match_threshold: float, clash_threshold: float) -> MatchVerdict:
    """Bucket a similarity into a qualitative verdict.

    Args:
        similarity: The dimension score.
        match_threshold: At or above this, the dimension counts as a match.
        clash_threshold: At or below this, the dimension counts as a clash.

    Returns:
        The corresponding :class:`MatchVerdict`.
    """
    if similarity >= match_threshold:
        return MatchVerdict.MATCH
    if similarity <= clash_threshold:
        return MatchVerdict.CLASH
    return MatchVerdict.PARTIAL


def score_dimensions(
    first: StudentProfile,
    second: StudentProfile,
    preferences: MatchPreferences,
    *,
    match_threshold: float = 0.75,
    clash_threshold: float = 0.35,
) -> tuple[FeatureContribution, ...]:
    """Score every dimension and package the results with their explanations.

    Args:
        first: One student.
        second: The other student.
        preferences: Weights and the department policy.
        match_threshold: Score at or above which a dimension is reported as a match.
        clash_threshold: Score at or below which a dimension is reported as a clash.

    Returns:
        One :class:`FeatureContribution` per dimension with a non-zero weight.
    """
    contributions: list[FeatureContribution] = []
    for dim in DIMENSIONS:
        weight = preferences.weight_for(dim.key)
        if weight <= 0:
            continue
        if dim.key == "department":
            # The department policy is institutional, so it comes from preferences
            # rather than being hard-coded into the scorer.
            same = first.department is second.department
            raw = preferences.same_department_bonus if same else 0.5
        else:
            raw = dim.score(first, second)
        similarity = min(1.0, max(0.0, raw))
        contributions.append(
            FeatureContribution(
                feature=dim.key,
                label=dim.label,
                similarity=round(similarity, 6),
                weight=weight,
                verdict=verdict_for(similarity, match_threshold, clash_threshold),
                first_value=dim.describe(first),
                second_value=dim.describe(second),
                explanation=dim.explain(similarity),
            )
        )
    return tuple(contributions)
