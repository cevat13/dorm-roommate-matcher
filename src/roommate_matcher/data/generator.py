"""Synthetic student generation.

Produces a dormitory population of any size with correlated attributes rather than
independent uniform noise: night owls tolerate more noise and study later, students
who work at the room desk need the room quieter, sociable students host more guests.

The correlations matter for evaluation. A population drawn from independent
distributions has no structure to find, so no metric could outperform another on it.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from faker import Faker

from roommate_matcher.domain.enums import (
    AlcoholHabit,
    Allergy,
    Department,
    DietType,
    GuestFrequency,
    Interest,
    Language,
    SleepSchedule,
    SmokingHabit,
    StudyLocation,
    StudyTime,
)
from roommate_matcher.domain.models import MAX_WEEKDAY_EVENINGS, StudentProfile

BIO_TEMPLATES = (
    "{year}. sınıf {department} öğrencisiyim. {interest} ile ilgileniyorum.",
    "Sakin bir oda arıyorum. Boş zamanlarımda {interest} yapmayı seviyorum.",
    "Genelde {study_location} ders çalışıyorum. {interest} favorim.",
    "Düzenli ve saygılı bir oda arkadaşı arıyorum. İlgi alanım: {interest}.",
    "{department} okuyorum, {study_time} çalışmayı tercih ediyorum.",
)

_CLEANLINESS_WEIGHTS = (0.05, 0.15, 0.30, 0.32, 0.18)
_STUDY_YEAR_WEIGHTS = (0.30, 0.25, 0.20, 0.17, 0.05, 0.03)
_DEPARTMENT_WEIGHTS = (
    0.14,
    0.09,
    0.08,
    0.07,
    0.07,
    0.08,
    0.07,
    0.09,
    0.06,
    0.06,
    0.05,
    0.06,
    0.05,
    0.03,
)

# A dormitory intake is dominated by first and second years, so ages cluster low.
_BASE_AGE = 17


def _weighted_rating(rng: random.Random, weights: tuple[float, ...] = _CLEANLINESS_WEIGHTS) -> int:
    """Draw a 1-5 rating from a realistic, non-uniform distribution.

    Args:
        rng: Seeded random source.
        weights: Probability of each rating from 1 to 5.

    Returns:
        A rating between 1 and 5.
    """
    return rng.choices(range(1, 6), weights=weights, k=1)[0]


def _nudge(value: int, delta: int, low: int = 1, high: int = 5) -> int:
    """Shift a rating while keeping it inside its valid range.

    Args:
        value: The starting rating.
        delta: How far to shift it.
        low: Lower bound.
        high: Upper bound.

    Returns:
        The clamped result.
    """
    return max(low, min(high, value + delta))


def generate_student(student_id: int, rng: random.Random, faker: Faker) -> StudentProfile:
    """Generate one internally consistent synthetic student.

    Args:
        student_id: The id to assign.
        rng: Seeded random source.
        faker: Seeded Faker instance for names and bios.

    Returns:
        A validated :class:`StudentProfile`.
    """
    department = rng.choices(list(Department), weights=_DEPARTMENT_WEIGHTS, k=1)[0]
    study_year = rng.choices(range(1, 7), weights=_STUDY_YEAR_WEIGHTS, k=1)[0]
    age = _BASE_AGE + study_year + rng.randint(0, 2)

    sleep = rng.choices(list(SleepSchedule), weights=(0.30, 0.50, 0.20), k=1)[0]
    cleanliness = _weighted_rating(rng)

    # Night owls tolerate more noise and tend to study later.
    noise_base = 4 if sleep is SleepSchedule.LATE else 2
    noise = _nudge(noise_base, rng.randint(-1, 1))
    social = _nudge(noise, rng.randint(-1, 1))

    if sleep is SleepSchedule.LATE:
        study_time = rng.choices(list(StudyTime), weights=(0.15, 0.35, 0.50), k=1)[0]
    elif sleep is SleepSchedule.EARLY:
        study_time = rng.choices(list(StudyTime), weights=(0.55, 0.35, 0.10), k=1)[0]
    else:
        study_time = rng.choice(list(StudyTime))

    study_location = rng.choices(list(StudyLocation), weights=(0.45, 0.25, 0.30), k=1)[0]
    # Students who work at the room desk spend more evenings there and need it
    # quieter than average.
    if study_location is StudyLocation.ROOM:
        room_time = rng.randint(3, MAX_WEEKDAY_EVENINGS)
        noise = _nudge(noise, -1)
    else:
        room_time = rng.randint(1, 4)

    guest_pool = (
        (GuestFrequency.RARELY, GuestFrequency.SOMETIMES, GuestFrequency.OFTEN)
        if social >= 4
        else (GuestFrequency.RARELY, GuestFrequency.RARELY, GuestFrequency.SOMETIMES)
    )
    guests = rng.choice(guest_pool)

    smoking = rng.choices(list(SmokingHabit), weights=(0.70, 0.24, 0.06), k=1)[0]
    alcohol = rng.choices(list(AlcoholHabit), weights=(0.55, 0.37, 0.08), k=1)[0]
    diet = rng.choices(list(DietType), weights=(0.62, 0.14, 0.04, 0.16, 0.04), k=1)[0]

    allergies: set[Allergy] = set()
    if smoking is SmokingHabit.NONE and rng.random() < 0.22:
        allergies.add(Allergy.SMOKE)
    if rng.random() < 0.14:
        allergies.add(rng.choice([Allergy.POLLEN, Allergy.DUST]))

    interests = frozenset(rng.sample(list(Interest), k=rng.randint(2, 6)))

    languages = {Language.TR}
    if rng.random() < 0.60:
        languages.add(Language.EN)
    if rng.random() < 0.08:
        languages.add(rng.choice([Language.DE, Language.FR, Language.AR, Language.RU]))

    bio = rng.choice(BIO_TEMPLATES).format(
        year=study_year,
        department=department.label_tr,
        study_location=study_location.label_tr.lower(),
        study_time=study_time.label_tr.lower(),
        interest=", ".join(sorted(i.label_tr for i in list(interests)[:2])),
    )

    return StudentProfile(
        student_id=student_id,
        student_no=f"{2020 + (6 - study_year)}{student_id:05d}",
        display_name=faker.name(),
        age=age,
        department=department,
        study_year=study_year,
        languages=frozenset(languages),
        study_location=study_location,
        study_time=study_time,
        sleep_schedule=sleep,
        cleanliness=cleanliness,
        noise_tolerance=noise,
        social_energy=social,
        room_time_weekdays=room_time,
        guest_frequency=guests,
        smoking=smoking,
        alcohol=alcohol,
        diet=diet,
        allergies=frozenset(allergies),
        interests=interests,
        bio=bio,
    )


def generate_students(
    count: int,
    *,
    seed: int = 42,
    start_id: int = 1,
) -> list[StudentProfile]:
    """Generate a reproducible synthetic population.

    Args:
        count: How many students to create.
        seed: Random seed; the same seed always yields the same population.
        start_id: The id assigned to the first student.

    Returns:
        The generated students.
    """
    return list(iter_students(count, seed=seed, start_id=start_id))


def iter_students(count: int, *, seed: int = 42, start_id: int = 1) -> Iterator[StudentProfile]:
    """Stream generated students without materialising the whole list.

    Args:
        count: How many students to create.
        seed: Random seed.
        start_id: The id assigned to the first student.

    Yields:
        One :class:`StudentProfile` at a time.
    """
    rng = random.Random(seed)
    faker = Faker("tr_TR")
    faker.seed_instance(seed)
    for offset in range(count):
        yield generate_student(start_id + offset, rng, faker)
