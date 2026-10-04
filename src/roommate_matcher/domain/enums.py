"""Controlled vocabularies for the categorical student attributes.

Each member carries a stable machine value for storage and a Turkish label for
display, so parsing and presentation share one definition.
"""

from __future__ import annotations

from enum import StrEnum


class LabelledEnum(StrEnum):
    """String enum that also carries a human readable Turkish label."""

    label_tr: str

    def __new__(cls, value: str, label_tr: str = "") -> LabelledEnum:
        """Create a member carrying both a stable value and a display label.

        Args:
            value: The stable machine value persisted in storage.
            label_tr: Turkish label shown in the UI.

        Returns:
            The newly created enum member.
        """
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.label_tr = label_tr or value
        return obj


class SleepSchedule(LabelledEnum):
    """When a student typically goes to bed."""

    EARLY = ("early", "Erken yatan")
    LATE = ("late", "Gece kuşu")
    FLEXIBLE = ("flexible", "Esnek")


class SmokingHabit(LabelledEnum):
    """Smoking behaviour, ordered from least to most intrusive."""

    NONE = ("none", "İçmiyor")
    OUTDOOR_ONLY = ("outdoor_only", "Sadece dışarıda")
    INDOOR = ("indoor", "Odada içiyor")


class AlcoholHabit(LabelledEnum):
    """How often alcohol is consumed in the room."""

    NONE = ("none", "Hiç")
    OCCASIONAL = ("occasional", "Ara sıra")
    REGULAR = ("regular", "Düzenli")


class StudyLocation(LabelledEnum):
    """Where the student does most of their coursework.

    Two students who both work at the room desk compete for the same quiet hours
    and the same table, which is one of the most common sources of friction in a
    shared dormitory room.
    """

    ROOM = ("room", "Odada")
    LIBRARY = ("library", "Kütüphanede")
    MIXED = ("mixed", "Karma")


class StudyTime(LabelledEnum):
    """The part of the day the student studies in."""

    DAYTIME = ("daytime", "Gündüz")
    EVENING = ("evening", "Akşam")
    NIGHT = ("night", "Gece")


class Department(LabelledEnum):
    """Faculty or department the student is enrolled in."""

    COMPUTER_ENG = ("computer_eng", "Bilgisayar Mühendisliği")
    ELECTRICAL_ENG = ("electrical_eng", "Elektrik-Elektronik Mühendisliği")
    MECHANICAL_ENG = ("mechanical_eng", "Makine Mühendisliği")
    CIVIL_ENG = ("civil_eng", "İnşaat Mühendisliği")
    INDUSTRIAL_ENG = ("industrial_eng", "Endüstri Mühendisliği")
    MEDICINE = ("medicine", "Tıp")
    LAW = ("law", "Hukuk")
    BUSINESS = ("business", "İşletme")
    ECONOMICS = ("economics", "Ekonomi")
    PSYCHOLOGY = ("psychology", "Psikoloji")
    ARCHITECTURE = ("architecture", "Mimarlık")
    EDUCATION = ("education", "Eğitim Fakültesi")
    SCIENCE = ("science", "Fen Fakültesi")
    FINE_ARTS = ("fine_arts", "Güzel Sanatlar")


class GuestFrequency(LabelledEnum):
    """How often guests are hosted in the room."""

    RARELY = ("rarely", "Nadiren")
    SOMETIMES = ("sometimes", "Ara sıra")
    OFTEN = ("often", "Sık sık")


class DietType(LabelledEnum):
    """Dietary pattern, relevant for the shared fridge and snacks."""

    OMNIVORE = ("omnivore", "Her şey")
    VEGETARIAN = ("vegetarian", "Vejetaryen")
    VEGAN = ("vegan", "Vegan")
    HALAL = ("halal", "Helal")
    GLUTEN_FREE = ("gluten_free", "Glutensiz")


class Allergy(LabelledEnum):
    """Allergies that constrain who a student can share a room with."""

    POLLEN = ("pollen", "Polen")
    DUST = ("dust", "Toz")
    SMOKE = ("smoke", "Sigara dumanı")


class Interest(LabelledEnum):
    """Canonical interest taxonomy.

    Raw free-text interests are normalised into these members by
    :mod:`roommate_matcher.features.interests`.
    """

    ART = ("art", "Sanat")
    BOOKS = ("books", "Kitap")
    COOKING = ("cooking", "Yemek")
    FITNESS = ("fitness", "Spor salonu")
    GAMING = ("gaming", "Oyun")
    MOVIES = ("movies", "Film & dizi")
    MUSIC = ("music", "Müzik")
    PETS = ("pets", "Hayvanlar")
    SPORTS = ("sports", "Spor")
    TRAVEL = ("travel", "Seyahat")
    PHOTOGRAPHY = ("photography", "Fotoğraf")
    TECHNOLOGY = ("technology", "Teknoloji")
    NATURE = ("nature", "Doğa")
    DANCE = ("dance", "Dans")
    VOLUNTEERING = ("volunteering", "Gönüllülük")


class Language(LabelledEnum):
    """Languages the student speaks."""

    TR = ("tr", "Türkçe")
    EN = ("en", "İngilizce")
    DE = ("de", "Almanca")
    FR = ("fr", "Fransızca")
    AR = ("ar", "Arapça")
    RU = ("ru", "Rusça")


class MatchVerdict(LabelledEnum):
    """How a single feature scored between two students."""

    MATCH = ("match", "Uyumlu")
    PARTIAL = ("partial", "Kısmen uyumlu")
    CLASH = ("clash", "Çatışma")
