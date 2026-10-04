"""Interest parsing and normalisation.

Raw interests arrive as comma-separated free text with inconsistent spelling. Every
variant is mapped onto the :class:`~roommate_matcher.domain.enums.Interest` taxonomy
and compared as whole tokens, never as substrings: ``"book"`` must not match
``"books"``.
"""

from __future__ import annotations

from collections.abc import Iterable

from roommate_matcher.domain.enums import Interest

SYNONYMS: dict[str, Interest] = {
    # Singular/plural variants, Turkish spellings and typos seen in the raw data.
    "book": Interest.BOOKS,
    "books": Interest.BOOKS,
    "kitap": Interest.BOOKS,
    "reading": Interest.BOOKS,
    "art": Interest.ART,
    "sanat": Interest.ART,
    "painting": Interest.ART,
    "cooking": Interest.COOKING,
    "cook": Interest.COOKING,
    "yemek": Interest.COOKING,
    "baking": Interest.COOKING,
    "fitness": Interest.FITNESS,
    "gym": Interest.FITNESS,
    "workout": Interest.FITNESS,
    "gaming": Interest.GAMING,
    "game": Interest.GAMING,
    "games": Interest.GAMING,
    "oyun": Interest.GAMING,
    "movie": Interest.MOVIES,
    "movies": Interest.MOVIES,
    "film": Interest.MOVIES,
    "cinema": Interest.MOVIES,
    "series": Interest.MOVIES,
    "music": Interest.MUSIC,
    "muzik": Interest.MUSIC,
    "müzik": Interest.MUSIC,
    "pet": Interest.PETS,
    "pets": Interest.PETS,
    "animals": Interest.PETS,
    "sport": Interest.SPORTS,
    "sports": Interest.SPORTS,
    "spor": Interest.SPORTS,
    "football": Interest.SPORTS,
    "travel": Interest.TRAVEL,
    "traveling": Interest.TRAVEL,
    "travelling": Interest.TRAVEL,
    "seyahat": Interest.TRAVEL,
    "photo": Interest.PHOTOGRAPHY,
    "photography": Interest.PHOTOGRAPHY,
    "fotograf": Interest.PHOTOGRAPHY,
    "tech": Interest.TECHNOLOGY,
    "technology": Interest.TECHNOLOGY,
    "coding": Interest.TECHNOLOGY,
    "programming": Interest.TECHNOLOGY,
    "nature": Interest.NATURE,
    "hiking": Interest.NATURE,
    "camping": Interest.NATURE,
    "doga": Interest.NATURE,
    "dance": Interest.DANCE,
    "dancing": Interest.DANCE,
    "dans": Interest.DANCE,
    "volunteering": Interest.VOLUNTEERING,
    "volunteer": Interest.VOLUNTEERING,
    "charity": Interest.VOLUNTEERING,
}
"""Maps every known spelling to its canonical taxonomy member."""


def normalise_token(token: str) -> Interest | None:
    """Map one raw interest token onto the taxonomy.

    Args:
        token: A single raw token, e.g. ``" Books "``.

    Returns:
        The canonical :class:`Interest`, or ``None`` when the token is unknown.
    """
    key = token.strip().strip("\"'").lower()
    if not key:
        return None
    if key in SYNONYMS:
        return SYNONYMS[key]
    try:
        return Interest(key)
    except ValueError:
        return None


def parse_interests(raw: str | Iterable[str] | None) -> frozenset[Interest]:
    """Parse a comma-separated string (or any iterable) into canonical interests.

    Unknown tokens are dropped rather than entering the feature space.

    Args:
        raw: Comma-separated text such as ``"fitness, movies, music"``, an iterable
            of tokens, or ``None``.

    Returns:
        The set of recognised interests.
    """
    if raw is None:
        return frozenset()
    tokens = raw.split(",") if isinstance(raw, str) else list(raw)
    found = (normalise_token(str(token)) for token in tokens)
    return frozenset(interest for interest in found if interest is not None)


def serialise_interests(interests: Iterable[Interest]) -> str:
    """Render interests back to a stable, sorted, comma-separated string.

    Args:
        interests: The interests to serialise.

    Returns:
        A canonical string such as ``"books,cooking,music"``.
    """
    return ",".join(sorted(interest.value for interest in interests))


def jaccard(left: frozenset[Interest], right: frozenset[Interest]) -> float:
    """Jaccard similarity between two interest sets.

    Args:
        left: First interest set.
        right: Second interest set.

    Returns:
        ``|A n B| / |A u B|``, and ``0.0`` when both sets are empty.
    """
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def labels(interests: Iterable[Interest]) -> str:
    """Turkish, comma-separated labels for display.

    Args:
        interests: The interests to label.

    Returns:
        A display string such as ``"Kitap, Müzik"``.
    """
    return ", ".join(sorted(interest.label_tr for interest in interests)) or "-"
