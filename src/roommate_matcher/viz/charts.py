"""Chart builders shared by the Streamlit app and the analysis script.

Colour encodes what the data does:

* contribution chart: a state (match / partial / clash), so the status palette.
  Two of those steps fall below 3:1 contrast on a light surface, so every bar also
  carries a value label and the verdict is spelled out in the companion table.
* radar: two students, so the first two categorical slots, with a legend.
* heatmap, distribution and strategy comparison: magnitude, so a single hue.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go

from roommate_matcher.domain.enums import GuestFrequency, MatchVerdict, SmokingHabit, StudyLocation
from roommate_matcher.domain.models import (
    MAX_WEEKDAY_EVENINGS,
    AssignmentPlan,
    FeatureContribution,
    PairScore,
    StudentProfile,
)
from roommate_matcher.matching.explain import impact_table

# Status palette: reserved for state, never reused as a series colour.
STATUS_COLOURS: dict[MatchVerdict, str] = {
    MatchVerdict.MATCH: "#0ca30c",
    MatchVerdict.PARTIAL: "#fab219",
    MatchVerdict.CLASH: "#d03b3b",
}

# Categorical slots 1 and 2, validated as an adjacent pair in both modes.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

# Single-hue sequential ramp for magnitude.
SEQUENTIAL_BLUE = [
    [0.0, "#cde2fb"],
    [0.25, "#9ec5f4"],
    [0.5, "#5598e7"],
    [0.75, "#2a78d6"],
    [1.0, "#0d366b"],
]

GRID = "rgba(128,128,128,0.18)"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

MAX_STUDY_YEAR = 6


def _base_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    """Apply the shared layout: recessive grid, transparent surface, readable type.

    Args:
        fig: The figure to style.
        title: Chart title.
        height: Chart height in pixels.

    Returns:
        The styled figure.
    """
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 10, "r": 20, "t": 50, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "size": 13},
        hoverlabel={"font_size": 13},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def contribution_chart(pair: PairScore) -> go.Figure:
    """Horizontal bars showing how each dimension scored.

    Bars are sorted by how many points each dimension cost, worst at the top.

    Args:
        pair: The explained pair score.

    Returns:
        A Plotly figure.
    """
    rows = impact_table(pair.contributions)
    rows.reverse()  # Plotly draws the first row at the bottom.

    labels = [row.label for row in rows]
    similarities = [row.similarity for row in rows]

    fig = go.Figure(
        go.Bar(
            x=similarities,
            y=labels,
            orientation="h",
            marker={
                "color": [STATUS_COLOURS[row.verdict] for row in rows],
                "line": {"width": 2, "color": "rgba(0,0,0,0)"},
            },
            text=[f"{value:.0%}" for value in similarities],
            textposition="outside",
            customdata=[[row.first_value, row.second_value, row.verdict.label_tr] for row in rows],
            hovertemplate=(
                "<b>%{y}</b><br>Öğrenci A: %{customdata[0]}<br>"
                "Öğrenci B: %{customdata[1]}<br>Benzerlik: %{x:.0%}<br>"
                "Durum: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", title="Uyum")
    return _base_layout(
        fig,
        f"{pair.first_name} + {pair.second_name} · toplam uyum %{pair.percentage}",
        height=max(340, 26 * len(labels) + 90),
    )


def radar_chart(
    first: StudentProfile,
    second: StudentProfile,
    contributions: Sequence[FeatureContribution],
) -> go.Figure:
    """Radar comparing two students across the top weighted dimensions.

    Args:
        first: One student.
        second: The other student.
        contributions: The scored dimensions for this pair.

    Returns:
        A Plotly figure.
    """
    top = sorted(contributions, key=lambda c: c.weight, reverse=True)[:8]
    labels = [c.label for c in top]
    closed = [*labels, labels[0]] if labels else []

    def profile_values(student: StudentProfile) -> list[float]:
        """Normalise each dimension's raw value onto ``[0, 1]`` for this student.

        Args:
            student: The student to plot.

        Returns:
            One value per axis, with the first repeated to close the polygon.
        """
        values = [_normalised(student, c.feature) for c in top]
        return [*values, values[0]] if values else []

    fig = go.Figure()
    for student, colour, name in (
        (first, SERIES_1, first.display_name),
        (second, SERIES_2, second.display_name),
    ):
        fig.add_trace(
            go.Scatterpolar(
                r=profile_values(student),
                theta=closed,
                name=name,
                line={"color": colour, "width": 2},
                fill="toself",
                opacity=0.35,
                hovertemplate="<b>%{theta}</b><br>%{r:.2f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 1], "gridcolor": GRID},
            "angularaxis": {"gridcolor": GRID},
            "bgcolor": "rgba(0,0,0,0)",
        },
        showlegend=True,
    )
    return _base_layout(fig, "Profil karşılaştırması", height=460)


def _normalised(student: StudentProfile, key: str) -> float:
    """Map one student attribute onto ``[0, 1]`` for the radar axes.

    Args:
        student: The student to read.
        key: The dimension key.

    Returns:
        A value in ``[0, 1]``; 0.5 for dimensions with no natural single-person scale.
    """
    match key:
        case "cleanliness":
            return (student.cleanliness - 1) / 4
        case "noise_tolerance":
            return (student.noise_tolerance - 1) / 4
        case "social_energy":
            return (student.social_energy - 1) / 4
        case "room_time":
            return student.room_time_weekdays / MAX_WEEKDAY_EVENINGS
        case "study_year":
            return (student.study_year - 1) / (MAX_STUDY_YEAR - 1)
        case "smoking":
            return {SmokingHabit.NONE: 0.0, SmokingHabit.OUTDOOR_ONLY: 0.5}.get(
                student.smoking, 1.0
            )
        case "guest_frequency":
            return {GuestFrequency.RARELY: 0.0, GuestFrequency.SOMETIMES: 0.5}.get(
                student.guest_frequency, 1.0
            )
        case "study_location":
            return {StudyLocation.LIBRARY: 0.0, StudyLocation.MIXED: 0.5}.get(
                student.study_location, 1.0
            )
        case "interests":
            return student.interest_diversity
        case _:
            return 0.5


def score_distribution(
    scores: Sequence[float],
    highlight: float | None = None,
    *,
    title: str = "Havuzdaki uyum dağılımı",
) -> go.Figure:
    """Histogram of compatibility scores, with one value marked.

    A score is hard to read without the distribution it sits in.

    Args:
        scores: The scores to plot.
        highlight: Optional score to mark with a reference line.
        title: Chart title.

    Returns:
        A Plotly figure.
    """
    fig = go.Figure(
        go.Histogram(
            x=list(scores),
            nbinsx=30,
            marker={"color": SERIES_1, "line": {"width": 2, "color": "rgba(0,0,0,0)"}},
            hovertemplate="Uyum %{x:.0%}<br>%{y} kayıt<extra></extra>",
        )
    )
    if highlight is not None:
        fig.add_vline(
            x=highlight,
            line_width=2,
            line_dash="dash",
            line_color="#d03b3b",
            annotation_text=f"%{round(highlight * 100)}",
            annotation_position="top",
        )
    fig.update_xaxes(tickformat=".0%", title="Uyum skoru")
    fig.update_yaxes(title="Sayı")
    return _base_layout(fig, title, height=320)


def room_score_distribution(plan: AssignmentPlan) -> go.Figure:
    """Histogram of the plan's room scores, with the worst room marked.

    The worst room is what dormitory staff actually act on, so it is called out.

    Args:
        plan: The plan to plot.

    Returns:
        A Plotly figure.
    """
    scores = [room.score for room in plan.rooms]
    return score_distribution(
        scores,
        plan.min_score if scores else None,
        title=(
            f"Oda uyum dağılımı · {plan.room_count} oda · ortalama %{round(plan.mean_score * 100)}"
        ),
    )


def similarity_heatmap(
    matrix: npt.NDArray[np.float64] | Sequence[Sequence[float]],
    labels: Sequence[str],
) -> go.Figure:
    """Pairwise compatibility heatmap for a small sample.

    Args:
        matrix: An ``n x n`` score matrix.
        labels: Row and column labels.

    Returns:
        A Plotly figure.
    """
    fig = go.Figure(
        go.Heatmap(
            z=[list(row) for row in matrix],
            x=list(labels),
            y=list(labels),
            colorscale=SEQUENTIAL_BLUE,
            zmin=0,
            zmax=1,
            colorbar={"title": "Uyum", "tickformat": ".0%"},
            hovertemplate="%{y} / %{x}<br>Uyum: %{z:.0%}<extra></extra>",
        )
    )
    return _base_layout(fig, "İkili uyum matrisi", height=520)


def metric_comparison(rows: Sequence[dict[str, float | str | int]]) -> go.Figure:
    """Bar chart of nDCG@10 per metric.

    One measure on one scale, so magnitude is carried by bar length, not by hue.

    Args:
        rows: Rows produced by :meth:`EvaluationReport.rows`.

    Returns:
        A Plotly figure.
    """
    frame = pd.DataFrame(list(rows)).sort_values("nDCG@10")
    fig = go.Figure(
        go.Bar(
            x=frame["nDCG@10"],
            y=frame["metric"],
            orientation="h",
            marker={"color": SERIES_1, "line": {"width": 2, "color": "rgba(0,0,0,0)"}},
            text=[f"{value:.3f}" for value in frame["nDCG@10"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>nDCG@10: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="nDCG@10", range=[0, max(frame["nDCG@10"]) * 1.2])
    return _base_layout(fig, "Metrik kalitesi (yüksek daha iyi)", height=320)


def strategy_comparison(rows: Sequence[dict[str, float | str | int]]) -> go.Figure:
    """Bar chart of total compatibility per assignment strategy.

    The objective being optimised is one measure on one scale, so bar length
    carries it and a single hue is used throughout.

    Args:
        rows: Rows produced by :meth:`PlanScores.as_row`.

    Returns:
        A Plotly figure.
    """
    frame = pd.DataFrame(list(rows)).sort_values("toplam")
    fig = go.Figure(
        go.Bar(
            x=frame["toplam"],
            y=frame["strateji"],
            orientation="h",
            marker={"color": SERIES_1, "line": {"width": 2, "color": "rgba(0,0,0,0)"}},
            text=[f"{value:.2f}" for value in frame["toplam"]],
            textposition="outside",
            customdata=frame[["fark %", "en kötü oda", "blocking"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Toplam uyum: %{x:.3f}<br>"
                "Optimale fark: %{customdata[0]}%<br>"
                "En kötü oda: %{customdata[1]:.3f}<br>"
                "Blocking pair: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Toplam uyum", range=[0, max(frame["toplam"]) * 1.2])
    return _base_layout(fig, "Yerleşim stratejileri (yüksek daha iyi)", height=320)


def contributions_table(contributions: Sequence[FeatureContribution]) -> pd.DataFrame:
    """The contribution chart as a table, so the data is never colour-only.

    Args:
        contributions: The scored dimensions.

    Returns:
        A display-ready DataFrame.
    """
    frame = pd.DataFrame([row.as_dict() for row in impact_table(contributions)])
    frame = frame[
        [
            "label",
            "first_value",
            "second_value",
            "similarity",
            "weight",
            "share_of_score",
            "verdict",
        ]
    ]
    frame.columns = [
        "Özellik",
        "Öğrenci A",
        "Öğrenci B",
        "Benzerlik",
        "Ağırlık",
        "Skora katkı",
        "Durum",
    ]
    frame["Durum"] = frame["Durum"].map(lambda v: MatchVerdict(v).label_tr)
    return frame
