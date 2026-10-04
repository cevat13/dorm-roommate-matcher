"""End-to-end tests for the Streamlit front end.

Streamlit renders exceptions into the page instead of failing the process, so the
server responding is not proof the script ran. ``AppTest`` executes it and exposes
anything it raised.

The page needs a populated database to render anything beyond its empty-state
warning, so these tests seed their own throwaway one rather than relying on
whatever the developer happens to have locally.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from roommate_matcher.config import get_settings
from roommate_matcher.data.database import SqlStudentRepository
from roommate_matcher.data.generator import generate_students

pytestmark = pytest.mark.integration

APP = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def seeded_database(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point the app at an isolated database holding a small population.

    Args:
        tmp_path_factory: Pytest's temporary directory factory.

    Yields:
        The path of the throwaway database.
    """
    directory = tmp_path_factory.mktemp("streamlit")
    database = directory / "app.db"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RM_DATABASE_URL", f"sqlite:///{database.as_posix()}")
        patch.setenv("RM_DATA_DIR", str(directory))
        patch.setenv("RM_GENERATED_DATA_DIR", str(directory / "generated"))
        get_settings.cache_clear()
        SqlStudentRepository().bulk_add(generate_students(24, seed=7))
        yield database
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def app(seeded_database: Path) -> AppTest:
    """Run the Streamlit script once and return the resulting page.

    Args:
        seeded_database: The throwaway database the page reads from.

    Returns:
        The executed :class:`AppTest`.
    """
    del seeded_database
    return AppTest.from_file(str(APP), default_timeout=180).run()


@pytest.fixture(scope="module")
def after_assignment(app: AppTest) -> AppTest:
    """The page after the assignment button has been pressed.

    The initial render only draws the controls; the plan itself is computed on
    click, so that path needs its own run or the main feature stays untested.

    Args:
        app: The freshly rendered page.

    Returns:
        The page after the click.
    """
    button = next(b for b in app.button if "Yerleşimi hesapla" in b.label)
    return button.click().run()


class TestInitialRender:
    """The page draws without raising."""

    def test_app_runs_without_exceptions(self, app: AppTest) -> None:
        assert not app.exception, [str(e.value) for e in app.exception]

    def test_page_has_its_title(self, app: AppTest) -> None:
        assert any("Oda Arkadaşı" in title.value for title in app.title)

    def test_all_tabs_are_present(self, app: AppTest) -> None:
        assert len(app.tabs) >= 4

    def test_the_empty_state_warning_is_not_shown(self, app: AppTest) -> None:
        """A missing database renders a warning and hides every control."""
        assert not any("Veritabanı boş" in error.value for error in app.error)

    def test_charts_are_rendered(self, app: AppTest) -> None:
        assert len(app.get("plotly_chart")) > 0

    def test_diagnostics_are_surfaced(self, app: AppTest) -> None:
        labels = {metric.label for metric in app.metric}
        assert {"Taranan aday", "Kısıt nedeniyle elenen"} <= labels


class TestSidebar:
    """Every registered metric and strategy is selectable."""

    def test_sidebar_offers_every_metric_and_strategy(self, app: AppTest) -> None:
        """The selectors render Turkish labels, so compare against those, not the keys."""
        from roommate_matcher.matching.assignment import available_strategies, get_strategy
        from roommate_matcher.matching.metrics import available_metrics, get_metric

        options = {option for box in app.sidebar.selectbox for option in (box.options or [])}
        assert {get_metric(key).label for key in available_metrics()} <= options
        assert {get_strategy(key).label for key in available_strategies()} <= options

    def test_sidebar_exposes_weight_sliders(self, app: AppTest) -> None:
        from roommate_matcher.domain.preferences import DEFAULT_WEIGHTS

        keys = {slider.label for slider in app.sidebar.slider}
        assert set(DEFAULT_WEIGHTS) <= keys


class TestAssignmentTab:
    """The main feature runs only when the button is pressed."""

    def test_assignment_button_runs_without_exceptions(self, after_assignment: AppTest) -> None:
        assert not after_assignment.exception, [str(e.value) for e in after_assignment.exception]

    def test_assignment_reports_the_plan_figures(self, after_assignment: AppTest) -> None:
        labels = {metric.label for metric in after_assignment.metric}
        assert {"Oda", "Toplam uyum", "Ortalama", "En kötü oda"} <= labels

    def test_assignment_confirms_the_plan_is_optimal(self, after_assignment: AppTest) -> None:
        assert any("kesin optimal" in success.value for success in after_assignment.success)

    def test_assignment_renders_charts_and_tables(self, after_assignment: AppTest) -> None:
        assert len(after_assignment.get("plotly_chart")) > 0
        assert len(after_assignment.get("dataframe")) > 0


class TestPopulationTab:
    """The strategy comparison is also behind a button."""

    def test_strategy_comparison_runs_without_exceptions(self, app: AppTest) -> None:
        button = next(b for b in app.button if "Stratejileri karşılaştır" in b.label)
        result = button.click().run()
        assert not result.exception, [str(e.value) for e in result.exception]
