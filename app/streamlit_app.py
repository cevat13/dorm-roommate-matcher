"""Streamlit front end for dormitory staff.

Run with::

    streamlit run app/streamlit_app.py

Calls the domain layer directly rather than going through HTTP, so running the UI
needs a single process. Scoring and assignment come from the same
:class:`~roommate_matcher.matching.engine.CompatibilityEngine` and
:class:`~roommate_matcher.matching.assignment.RoomAssigner` the API uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:  # pragma: no cover - dev convenience
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from roommate_matcher.data.database import SqlStudentRepository  # noqa: E402
from roommate_matcher.domain.enums import (  # noqa: E402
    AlcoholHabit,
    Allergy,
    Department,
    DietType,
    GuestFrequency,
    Interest,
    SleepSchedule,
    SmokingHabit,
    StudyLocation,
    StudyTime,
)
from roommate_matcher.domain.models import AssignmentPlan, StudentProfile  # noqa: E402
from roommate_matcher.domain.preferences import (  # noqa: E402
    DEFAULT_WEIGHTS,
    Constraints,
    MatchPreferences,
)
from roommate_matcher.matching.assignment import (  # noqa: E402
    RoomAssigner,
    available_strategies,
    get_strategy,
)
from roommate_matcher.matching.constraints import describe_violations  # noqa: E402
from roommate_matcher.matching.engine import CompatibilityEngine  # noqa: E402
from roommate_matcher.matching.evaluation import evaluate_plans  # noqa: E402
from roommate_matcher.matching.explain import compare  # noqa: E402
from roommate_matcher.matching.metrics import available_metrics, get_metric  # noqa: E402
from roommate_matcher.viz.charts import (  # noqa: E402
    contribution_chart,
    contributions_table,
    radar_chart,
    room_score_distribution,
    score_distribution,
    similarity_heatmap,
    strategy_comparison,
)

st.set_page_config(
    page_title="Yurt Oda Eşleştirme",
    page_icon="🛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_repository() -> SqlStudentRepository:
    """Open the repository once per session.

    Returns:
        The shared repository.
    """
    return SqlStudentRepository()


@st.cache_data(show_spinner=False)
def load_students() -> list[StudentProfile]:
    """Load every student, cached across reruns.

    Returns:
        All stored students.
    """
    return load_repository().list_all()


SidebarSettings = tuple[MatchPreferences, str, str, int]
"""Preferences, metric key, strategy key and population cap."""


def sidebar_settings() -> SidebarSettings:
    """Render the sidebar controls once per run.

    Streamlit derives widget ids from their parameters, so these must be built in
    exactly one place; the result is passed to whichever tab needs it.

    Returns:
        The assembled preferences, the metric key, the strategy key and the
        population cap.
    """
    st.sidebar.header("Ayarlar")

    metric_keys = available_metrics()
    metric = st.sidebar.selectbox(
        "Uyum metriği",
        metric_keys,
        index=metric_keys.index("weighted_gower") if "weighted_gower" in metric_keys else 0,
        format_func=lambda key: get_metric(key).label,
    )
    st.sidebar.caption(get_metric(metric).description)

    strategy_keys = available_strategies()
    strategy = st.sidebar.selectbox(
        "Yerleşim stratejisi",
        strategy_keys,
        index=strategy_keys.index("optimal") if "optimal" in strategy_keys else 0,
        format_func=lambda key: get_strategy(key).label,
    )
    st.sidebar.caption(get_strategy(strategy).description)

    limit = st.sidebar.slider("Yerleştirilecek öğrenci sayısı", 10, 400, 80, step=10)

    st.sidebar.subheader("Kısıtlar")
    st.sidebar.caption(
        "İhlaller büyük ceza olarak modellenir: çözüm her zaman bulunur, "
        "kaçınılamayan ihlaller raporlanır."
    )
    separate_smokers = st.sidebar.checkbox("Sigara içen ve içmeyeni ayır", value=True)
    allergies = st.sidebar.checkbox("Sigara alerjisine uy", value=True)
    use_year_gap = st.sidebar.checkbox("Sınıf farkı sınırı", value=False)
    max_year_gap = st.sidebar.slider("En fazla sınıf farkı", 0, 5, 2) if use_year_gap else None

    st.sidebar.subheader("Bölüm politikası")
    department_bonus = st.sidebar.slider(
        "Aynı bölüm skoru",
        0.0,
        1.0,
        1.0,
        0.05,
        help="1.0 aynı bölümü tam uyum sayar, 0.5 nötr, düşük değerler bölüm karıştırır.",
    )

    st.sidebar.subheader("Ağırlıklar")
    weights: dict[str, float] = {}
    with st.sidebar.expander("Ağırlıkları ayarla", expanded=False):
        for key, default in DEFAULT_WEIGHTS.items():
            weights[key] = st.slider(key, 0.0, 5.0, float(default), 0.1, key=f"w_{key}")

    preferences = MatchPreferences(
        weights=weights,
        constraints=Constraints(
            separate_smokers=separate_smokers,
            enforce_allergies=allergies,
            max_year_gap=max_year_gap,
        ),
        same_department_bonus=department_bonus,
    )
    return preferences, metric, strategy, limit


def render_assignment_tab(students: list[StudentProfile], settings: SidebarSettings) -> None:
    """Render the room plan view.

    Args:
        students: The population to assign.
        settings: The shared sidebar selections.
    """
    preferences, metric, strategy, limit = settings
    subset = students[:limit]

    st.subheader("Yerleşim planı")
    st.caption(
        f"{len(subset)} öğrenci, 2 kişilik odalar. Optimal strateji toplam uyumu "
        "maksimize eden kesin çözümü üretir."
    )

    if st.button("Yerleşimi hesapla", type="primary"):
        engine = CompatibilityEngine(metric, preferences)
        assigner = RoomAssigner(strategy, engine)
        with st.spinner(f"{len(subset)} öğrenci yerleştiriliyor..."):
            st.session_state["plan"] = assigner.assign(subset)
            st.session_state["plan_students"] = subset

    plan: AssignmentPlan | None = st.session_state.get("plan")
    if plan is None:
        st.info("Planı görmek için yukarıdaki düğmeye basın.")
        return

    cols = st.columns(5)
    cols[0].metric("Oda", plan.room_count)
    cols[1].metric("Toplam uyum", f"{plan.total_score:.2f}")
    cols[2].metric("Ortalama", f"%{round(plan.mean_score * 100)}")
    cols[3].metric("En kötü oda", f"%{round(plan.min_score * 100)}")
    cols[4].metric("Süre", f"{plan.seconds:.2f} sn")

    if plan.optimal:
        st.success("Bu plan toplam uyumu maksimize eden kesin optimal çözüm.")
    else:
        st.warning(f"{plan.strategy} stratejisi sezgiseldir; optimal olduğu garanti değil.")

    if plan.violations:
        st.error(f"Kaçınılamayan ihlaller: {describe_violations(plan.violations)}")
    else:
        st.success("Hiçbir kısıt ihlali yok.")

    if plan.unplaced:
        st.warning(f"Yerleşmeyen öğrenci: {list(plan.unplaced)} (tek sayıda öğrenci)")

    if plan.blocking_pairs:
        st.caption(
            f"Kararlılık teşhisi: {len(plan.blocking_pairs)} blocking pair. "
            "Toplam uyumu maksimize etmek kararlılığı garanti etmez."
        )
    else:
        st.caption("Kararlılık teşhisi: hiçbir blocking pair yok.")

    st.plotly_chart(room_score_distribution(plan), width="stretch")

    st.markdown("#### Odalar")
    st.dataframe(
        [
            {
                "Oda": room.room_no,
                "Öğrenci A": f"#{room.pair.first_id} {room.pair.first_name}",
                "Öğrenci B": f"#{room.pair.second_id} {room.pair.second_name}",
                "Uyum": f"%{room.pair.percentage}",
                "İhlal": ", ".join(room.pair.violations) or "-",
            }
            for room in plan.rooms
        ],
        width="stretch",
        hide_index=True,
        height=320,
    )

    st.markdown("#### Oda gerekçesi")
    room_numbers = [room.room_no for room in plan.rooms]
    if not room_numbers:
        return
    chosen = st.selectbox("Oda seç", room_numbers, format_func=lambda n: f"Oda {n}")
    room = next(r for r in plan.rooms if r.room_no == chosen)
    by_id = {s.student_id: s for s in st.session_state.get("plan_students", students)}

    st.write(room.pair.summary)
    left, right = st.columns(2)
    left.plotly_chart(contribution_chart(room.pair), width="stretch")
    first = by_id.get(room.pair.first_id)
    second = by_id.get(room.pair.second_id)
    if first is not None and second is not None:
        right.plotly_chart(radar_chart(first, second, room.pair.contributions), width="stretch")
    with st.expander("Tablo görünümü (renk körlüğü / ekran okuyucu için)"):
        st.dataframe(contributions_table(room.pair.contributions), width="stretch", hide_index=True)


def render_student_tab(students: list[StudentProfile], settings: SidebarSettings) -> None:
    """Render the per-student analysis view.

    Args:
        students: The population to analyse.
        settings: The shared sidebar selections.
    """
    preferences, metric, _strategy, _limit = settings
    by_id = {s.student_id: s for s in students}

    student_id = st.selectbox(
        "Öğrenci seç",
        sorted(by_id),
        format_func=lambda sid: f"#{sid} · {by_id[sid].display_name}",
    )
    student = by_id[student_id]

    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"### {student.display_name}")
        st.caption(student.summary())
        st.metric("Sessiz oda arkadaşı indeksi", f"{student.quiet_index:.2f}")
    with right:
        st.markdown("**Profil**")
        st.write(
            {
                "Öğrenci no": student.student_no,
                "Bölüm": student.department.label_tr,
                "Sınıf": f"{student.study_year}. sınıf",
                "Ders çalışma": f"{student.study_location.label_tr} / "
                f"{student.study_time.label_tr}",
                "Uyku": student.sleep_schedule.label_tr,
                "Sigara": student.smoking.label_tr,
                "İlgi alanları": ", ".join(sorted(i.label_tr for i in student.interests)) or "-",
            }
        )

    engine = CompatibilityEngine(metric, preferences)
    pool = [s for s in students if s.student_id != student_id]
    with st.spinner(f"{len(pool)} aday değerlendiriliyor..."):
        report = engine.candidates_for(student, pool, top_n=8)

    st.divider()
    cols = st.columns(4)
    cols[0].metric("Taranan aday", report.considered)
    cols[1].metric("Kısıt nedeniyle elenen", report.filtered_out)
    cols[2].metric("Gösterilen", len(report.results))
    cols[3].metric("En yüksek uyum", f"%{report.best.percentage}" if report.best else "-")

    if report.rejections:
        st.caption(f"Elenenler: {describe_violations(report.rejections)}")

    if not report.results:
        st.warning("Kısıtlara uyan aday bulunamadı. Kısıtları gevşetmeyi deneyin.")
        return

    if len(report.results) > 1:
        st.info(compare(list(report.results)))

    tabs = st.tabs([f"#{p.second_id} · %{p.percentage}" for p in report.results])
    for tab, pair in zip(tabs, report.results, strict=True):
        with tab:
            candidate = by_id[pair.second_id]
            head_left, head_right = st.columns([3, 1])
            head_left.markdown(f"#### {pair.second_name}")
            head_left.caption(candidate.summary())
            head_right.metric("Uyum", f"%{pair.percentage}")

            st.write(pair.summary)
            chart_left, chart_right = st.columns(2)
            chart_left.plotly_chart(contribution_chart(pair), width="stretch")
            chart_right.plotly_chart(
                radar_chart(student, candidate, pair.contributions), width="stretch"
            )
            with st.expander("Tablo görünümü"):
                st.dataframe(
                    contributions_table(pair.contributions), width="stretch", hide_index=True
                )
            if candidate.bio:
                st.caption(f"_{candidate.bio}_")

    all_scores = [engine.score_pair(student, other) for other in pool[: min(len(pool), 400)]]
    st.plotly_chart(
        score_distribution(all_scores, report.best.score if report.best else None),
        width="stretch",
    )


def render_population_tab(students: list[StudentProfile]) -> None:
    """Render the population-level analysis view.

    Args:
        students: The population to analyse.
    """
    st.subheader("Popülasyon analizi")

    from collections import Counter

    cols = st.columns(4)
    cols[0].metric("Öğrenci", len(students))
    cols[1].metric(
        "Odada çalışan",
        sum(1 for s in students if s.study_location is StudyLocation.ROOM),
    )
    cols[2].metric("Sigara içen", sum(1 for s in students if s.smokes))
    cols[3].metric(
        "Ortalama temizlik",
        f"{sum(s.cleanliness for s in students) / max(len(students), 1):.1f}/5",
    )

    distributions = {
        "Sınıf": Counter(f"{s.study_year}. sınıf" for s in students),
        "Uyku düzeni": Counter(s.sleep_schedule.label_tr for s in students),
        "Ders çalışma yeri": Counter(s.study_location.label_tr for s in students),
        "Sigara": Counter(s.smoking.label_tr for s in students),
    }
    st.dataframe(
        [
            {"Dağılım": label, "Değerler": ", ".join(f"{k}: {v}" for k, v in c.most_common(6))}
            for label, c in distributions.items()
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### İkili uyum matrisi")
    sample_size = st.slider("Örneklem boyutu", 5, 40, 16)
    sample = students[:sample_size]
    engine = CompatibilityEngine()
    with st.spinner("Uyum matrisi hesaplanıyor..."):
        matrix = engine.score_matrix(sample, with_violations=False)
    st.plotly_chart(
        similarity_heatmap(matrix.scores, [f"#{s.student_id}" for s in sample]),
        width="stretch",
    )
    st.caption(
        "Matris O(n²) olduğu için yalnızca küçük örneklemlerde gösterilir; "
        "yerleşim hesabı aynı matrisi tüm popülasyon için bir kez kurar."
    )

    st.markdown("#### Strateji karşılaştırması")
    st.caption(
        "Aynı skor matrisi üzerinde dört strateji: optimal çözümün diğerlerinden "
        "ne kadar iyi olduğunu gösterir."
    )
    compare_size = st.slider("Karşılaştırma örneklemi", 20, 200, 60, step=10)
    if st.button("Stratejileri karşılaştır"):
        with st.spinner("Stratejiler çalıştırılıyor..."):
            scores = evaluate_plans(students[:compare_size])
        rows = [s.as_row() for s in scores]
        st.plotly_chart(strategy_comparison(rows), width="stretch")
        st.dataframe(rows, width="stretch", hide_index=True)


def render_add_tab(repository: SqlStudentRepository) -> None:
    """Render the student record form.

    Args:
        repository: The repository new records are written to.
    """
    st.subheader("Öğrenci kaydı ekle")
    st.caption("Kaydedilen öğrenci sonraki yerleşim hesabına dahil olur.")

    with st.form("new_student"):
        col1, col2, col3 = st.columns(3)
        student_no = col1.text_input("Öğrenci no", "202400001")
        name = col2.text_input("İsim", "Yeni Öğrenci")
        age = col3.number_input("Yaş", 17, 40, 20)

        col1, col2, col3 = st.columns(3)
        department = col1.selectbox("Bölüm", list(Department), format_func=lambda d: d.label_tr)
        study_year = col2.number_input("Sınıf", 1, 6, 1)
        sleep = col3.selectbox("Uyku düzeni", list(SleepSchedule), format_func=lambda s: s.label_tr)

        col1, col2, col3 = st.columns(3)
        study_location = col1.selectbox(
            "Ders çalışma yeri", list(StudyLocation), format_func=lambda s: s.label_tr
        )
        study_time = col2.selectbox(
            "Çalışma saatleri", list(StudyTime), format_func=lambda s: s.label_tr
        )
        room_time = col3.slider("Odada geçen akşam (hafta içi)", 0, 5, 3)

        col1, col2, col3 = st.columns(3)
        cleanliness = col1.slider("Temizlik", 1, 5, 4)
        noise = col2.slider("Gürültü toleransı", 1, 5, 3)
        social = col3.slider("Sosyallik", 1, 5, 3)

        col1, col2, col3 = st.columns(3)
        smoking = col1.selectbox("Sigara", list(SmokingHabit), format_func=lambda s: s.label_tr)
        alcohol = col2.selectbox("Alkol", list(AlcoholHabit), format_func=lambda a: a.label_tr)
        guests = col3.selectbox(
            "Misafir sıklığı", list(GuestFrequency), format_func=lambda g: g.label_tr
        )

        diet = st.selectbox("Beslenme", list(DietType), format_func=lambda d: d.label_tr)
        interests = st.multiselect(
            "İlgi alanları", list(Interest), format_func=lambda i: i.label_tr
        )
        allergies = st.multiselect("Alerjiler", list(Allergy), format_func=lambda a: a.label_tr)
        bio = st.text_area("Kısa tanıtım", max_chars=1000)

        submitted = st.form_submit_button("Kaydet", type="primary")

    if not submitted:
        return

    student = StudentProfile(
        student_id=repository.next_id(),
        student_no=student_no,
        display_name=name,
        age=int(age),
        department=department,
        study_year=int(study_year),
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
        interests=frozenset(interests),
        bio=bio,
    )
    repository.add(student)
    load_students.clear()
    st.success(f"Öğrenci kaydedildi. Kimlik: {student.student_id}")


def main() -> None:
    """Compose the page."""
    st.title("🛏️ Yurt Oda Arkadaşı Eşleştirme")
    st.caption(
        "Ağırlıklı Gower uyumluluğu ve maksimum ağırlıklı eşleştirme ile "
        "iki kişilik odalara optimal yerleşim."
    )

    repository = load_repository()
    if repository.count() == 0:
        st.error(
            "Veritabanı boş. Terminalde şunu çalıştırın:\n\n```\nroommate seed --count 400\n```"
        )
        return

    students = load_students()
    settings = sidebar_settings()
    assignment_tab, student_tab, population_tab, add_tab = st.tabs(
        ["Yerleşim", "Öğrenci analizi", "Popülasyon", "Öğrenci ekle"]
    )
    with assignment_tab:
        render_assignment_tab(students, settings)
    with student_tab:
        render_student_tab(students, settings)
    with population_tab:
        render_population_tab(students)
    with add_tab:
        render_add_tab(repository)


if __name__ == "__main__":
    main()
