"""End-to-end CLI tests.

Each test runs against a throwaway SQLite file rather than the configured
database, so the developer's own data is never touched.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from roommate_matcher.cli.main import app
from roommate_matcher.config import get_settings

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the CLI at an empty database for the duration of one test.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        monkeypatch: Pytest's environment patcher.

    Yields:
        The path of the throwaway database.
    """
    database = tmp_path / "cli.db"
    monkeypatch.setenv("RM_DATABASE_URL", f"sqlite:///{database.as_posix()}")
    monkeypatch.setenv("RM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RM_GENERATED_DATA_DIR", str(tmp_path / "generated"))
    get_settings.cache_clear()
    yield database
    get_settings.cache_clear()


@pytest.fixture
def seeded(isolated_db: Path) -> Path:
    """A database holding a small synthetic population.

    Args:
        isolated_db: The throwaway database path.

    Returns:
        The same path, now populated.
    """
    result = runner.invoke(app, ["seed", "--count", "24"])
    assert result.exit_code == 0, result.output
    return isolated_db


class TestInfoCommands:
    """Commands that need no data."""

    def test_version_prints_the_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_strategies_lists_the_optimal_one(self) -> None:
        result = runner.invoke(app, ["strategies"])
        assert result.exit_code == 0
        assert "optimal" in result.output

    def test_metrics_lists_the_default(self) -> None:
        result = runner.invoke(app, ["metrics"])
        assert result.exit_code == 0
        assert "weighted_gower" in result.output


class TestSeed:
    """Seeding is the first command a new user runs."""

    def test_seed_populates_the_database(self, isolated_db: Path) -> None:
        result = runner.invoke(app, ["seed", "--count", "12"])
        assert result.exit_code == 0
        assert "12 öğrenci yüklendi" in result.output

    def test_seed_refuses_to_overwrite(self, seeded: Path) -> None:
        """A second run must explain itself rather than raising IntegrityError."""
        result = runner.invoke(app, ["seed", "--count", "4"])
        assert result.exit_code == 1
        assert "zaten" in result.output
        assert "--reset" in result.output

    def test_seed_reset_replaces_the_population(self, seeded: Path) -> None:
        result = runner.invoke(app, ["seed", "--count", "8", "--reset"])
        assert result.exit_code == 0
        assert "silindi" in result.output
        assert "8 öğrenci yüklendi" in result.output

    def test_seed_is_reproducible(self, isolated_db: Path) -> None:
        """The same seed must yield the same population, or evaluation drifts."""
        runner.invoke(app, ["seed", "--count", "10", "--seed", "5"])
        first = runner.invoke(app, ["candidates", "1", "--top", "3"]).output
        runner.invoke(app, ["seed", "--count", "10", "--seed", "5", "--reset"])
        second = runner.invoke(app, ["candidates", "1", "--top", "3"]).output
        assert first == second

    def test_seed_writes_a_csv_snapshot(self, isolated_db: Path, tmp_path: Path) -> None:
        result = runner.invoke(app, ["seed", "--count", "6"])
        assert result.exit_code == 0
        assert (tmp_path / "generated" / "students.csv").exists()


class TestEmptyDatabaseGuards:
    """Every data command must fail helpfully on an empty database."""

    @pytest.mark.parametrize(
        "command",
        [["assign"], ["candidates", "1"], ["compare", "1", "2"], ["stats"], ["evaluate"]],
    )
    def test_command_explains_the_empty_database(
        self, isolated_db: Path, command: list[str]
    ) -> None:
        result = runner.invoke(app, command)
        assert result.exit_code == 1
        assert "roommate seed" in result.output

    @pytest.mark.parametrize("command", [["rooms"], ["room", "1"]])
    def test_plan_command_explains_the_missing_plan(self, seeded: Path, command: list[str]) -> None:
        result = runner.invoke(app, command)
        assert result.exit_code == 1
        assert "roommate assign" in result.output


class TestAssign:
    """The main command."""

    def test_assign_reports_the_plan(self, seeded: Path) -> None:
        result = runner.invoke(app, ["assign", "--show", "3"])
        assert result.exit_code == 0
        assert "Oda: 12" in result.output
        assert "kesin optimal" in result.output
        assert "Plan kaydedildi" in result.output

    def test_assign_reports_no_violations_when_there_are_none(self, seeded: Path) -> None:
        result = runner.invoke(app, ["assign", "--show", "1"])
        assert "Hiçbir kısıt ihlali yok" in result.output

    def test_greedy_is_not_labelled_optimal(self, seeded: Path) -> None:
        result = runner.invoke(app, ["assign", "--strategy", "greedy", "--show", "1"])
        assert result.exit_code == 0
        assert "sezgisel" in result.output

    def test_unknown_strategy_fails_with_a_message(self, seeded: Path) -> None:
        result = runner.invoke(app, ["assign", "--strategy", "yok"])
        assert result.exit_code == 1
        assert "optimal" in result.output

    def test_unknown_metric_fails_with_a_message(self, seeded: Path) -> None:
        result = runner.invoke(app, ["assign", "--metric", "yok"])
        assert result.exit_code == 1
        assert "weighted_gower" in result.output

    def test_no_save_leaves_no_plan_behind(self, seeded: Path) -> None:
        runner.invoke(app, ["assign", "--no-save", "--show", "1"])
        result = runner.invoke(app, ["rooms"])
        assert result.exit_code == 1
        assert "Kayıtlı plan yok" in result.output


class TestPlanInspection:
    """Reading a stored plan back."""

    def test_rooms_lists_the_stored_plan(self, seeded: Path) -> None:
        runner.invoke(app, ["assign", "--show", "1"])
        result = runner.invoke(app, ["rooms", "--limit", "4"])
        assert result.exit_code == 0
        assert "optimal" in result.output

    def test_rooms_worst_flag_is_accepted(self, seeded: Path) -> None:
        runner.invoke(app, ["assign", "--show", "1"])
        assert runner.invoke(app, ["rooms", "--worst", "--limit", "3"]).exit_code == 0

    def test_room_explains_one_room(self, seeded: Path) -> None:
        runner.invoke(app, ["assign", "--show", "1"])
        result = runner.invoke(app, ["room", "1"])
        assert result.exit_code == 0
        assert "özellik dökümü" in result.output
        assert "Sigara" in result.output

    def test_unknown_room_fails(self, seeded: Path) -> None:
        runner.invoke(app, ["assign", "--show", "1"])
        result = runner.invoke(app, ["room", "9999"])
        assert result.exit_code == 1
        assert "bu planda yok" in result.output


class TestAnalysisCommands:
    """Staff-facing analysis."""

    def test_candidates_ranks_roommates(self, seeded: Path) -> None:
        result = runner.invoke(app, ["candidates", "1", "--top", "3"])
        assert result.exit_code == 0
        assert "En uyumlu" in result.output

    def test_candidates_detail_prints_the_breakdown(self, seeded: Path) -> None:
        result = runner.invoke(app, ["candidates", "1", "--top", "2", "--detail"])
        assert result.exit_code == 0
        assert "özellik dökümü" in result.output

    def test_candidates_for_an_unknown_student_fails(self, seeded: Path) -> None:
        result = runner.invoke(app, ["candidates", "99999"])
        assert result.exit_code == 1

    def test_compare_explains_a_pair(self, seeded: Path) -> None:
        result = runner.invoke(app, ["compare", "1", "2"])
        assert result.exit_code == 0
        assert "Uyum" in result.output
        assert "özellik dökümü" in result.output

    def test_compare_with_an_unknown_student_fails(self, seeded: Path) -> None:
        assert runner.invoke(app, ["compare", "1", "99999"]).exit_code == 1

    def test_stats_shows_distributions(self, seeded: Path) -> None:
        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0
        assert "Sınıf" in result.output

    def test_evaluate_compares_metrics_and_strategies(self, seeded: Path) -> None:
        result = runner.invoke(app, ["evaluate", "--sample", "8", "--limit", "16"])
        assert result.exit_code == 0
        assert "Metrik kalitesi" in result.output
        assert "Yerleşim kalitesi" in result.output

    def test_evaluate_can_save_json(self, seeded: Path, tmp_path: Path) -> None:
        import json

        target = tmp_path / "out" / "eval.json"
        result = runner.invoke(
            app, ["evaluate", "--sample", "6", "--limit", "12", "--save", str(target)]
        )
        assert result.exit_code == 0
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert {"metrics", "strategies"} <= set(payload)


class TestCreateAdmin:
    """Staff accounts are created from the CLI."""

    def test_admin_is_created(self, isolated_db: Path) -> None:
        result = runner.invoke(
            app, ["create-admin", "--email", "a@b.c", "--password", "parola12345"]
        )
        assert result.exit_code == 0
        assert "oluşturuldu" in result.output

    def test_short_password_is_refused(self, isolated_db: Path) -> None:
        result = runner.invoke(app, ["create-admin", "--email", "a@b.c", "--password", "kisa"])
        assert result.exit_code == 1
        assert "en az 8" in result.output

    def test_duplicate_email_is_refused(self, isolated_db: Path) -> None:
        runner.invoke(app, ["create-admin", "--email", "a@b.c", "--password", "parola12345"])
        result = runner.invoke(
            app, ["create-admin", "--email", "a@b.c", "--password", "parola12345"]
        )
        assert result.exit_code == 1
