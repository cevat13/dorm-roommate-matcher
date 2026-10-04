"""End-to-end API tests.

The app is built against a throwaway SQLite file per test session rather than the
configured database.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from roommate_matcher.api.dependencies import get_repository
from roommate_matcher.api.main import create_app
from roommate_matcher.api.security import hash_password
from roommate_matcher.data.database import SqlStudentRepository, create_db_engine
from roommate_matcher.data.generator import generate_students
from roommate_matcher.matching.assignment import RoomAssigner

pytestmark = pytest.mark.integration

ADMIN_EMAIL = "yonetici@yurt.edu.tr"
ADMIN_PASSWORD = "cokGucluParola42"

NEW_STUDENT = {
    "student_no": "209900001",
    "display_name": "Test Öğrenci",
    "age": 21,
    "department": "law",
    "study_year": 2,
    "study_location": "library",
    "study_time": "daytime",
    "sleep_schedule": "early",
    "cleanliness": 4,
    "noise_tolerance": 2,
    "social_energy": 3,
    "room_time_weekdays": 3,
    "smoking": "none",
    "interests": ["books", "music"],
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """An API client backed by an isolated database.

    Args:
        tmp_path_factory: Pytest's temporary directory factory.

    Yields:
        A configured test client.
    """
    database: Path = tmp_path_factory.mktemp("db") / "test.sqlite"
    repository = SqlStudentRepository(create_db_engine(f"sqlite:///{database.as_posix()}"))
    repository.bulk_add(generate_students(40, seed=77))
    repository.add_admin(ADMIN_EMAIL, hash_password(ADMIN_PASSWORD))

    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repository
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def token(client: TestClient) -> str:
    """A bearer token for the seeded staff account.

    Args:
        client: The API client.

    Returns:
        The access token.
    """
    response = client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    return str(response.json()["access_token"])


@pytest.fixture(scope="module")
def auth(token: str) -> dict[str, str]:
    """The Authorization header for the seeded staff account.

    Args:
        token: The access token.

    Returns:
        A header mapping.
    """
    return {"Authorization": f"Bearer {token}"}


class TestHealth:
    def test_health_reports_the_population(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["students"] == 40
        assert body["admins"] == 1

    def test_health_reports_the_fixed_room_capacity(self, client: TestClient) -> None:
        assert client.get("/health").json()["room_capacity"] == 2

    def test_health_flags_the_development_secret(self, client: TestClient) -> None:
        assert client.get("/health").json()["insecure_secret"] is True

    def test_metrics_endpoint_describes_each_strategy(self, client: TestClient) -> None:
        metrics = client.get("/metrics").json()
        assert {m["key"] for m in metrics} >= {"weighted_gower", "cosine", "euclidean"}
        assert any(m["flag"] for m in metrics)

    def test_strategies_endpoint_marks_the_optimal_one(self, client: TestClient) -> None:
        strategies = client.get("/strategies").json()
        assert {s["key"] for s in strategies} == {"optimal", "greedy", "random", "stable"}
        assert [s["key"] for s in strategies if s["flag"]] == ["optimal"]

    def test_openapi_schema_is_served(self, client: TestClient) -> None:
        assert client.get("/openapi.json").status_code == 200


class TestStudents:
    def test_list_is_paginated(self, client: TestClient) -> None:
        assert len(client.get("/students?limit=5").json()) == 5

    def test_offset_moves_the_window(self, client: TestClient) -> None:
        first = client.get("/students?limit=3").json()
        second = client.get("/students?limit=3&offset=3").json()
        assert {s["student_id"] for s in first}.isdisjoint({s["student_id"] for s in second})

    def test_study_year_filter_applies(self, client: TestClient) -> None:
        students = client.get("/students?study_year=1&limit=50").json()
        assert all(s["study_year"] == 1 for s in students)

    def test_single_student_is_returned(self, client: TestClient) -> None:
        assert client.get("/students/1").json()["student_id"] == 1

    def test_missing_student_is_404(self, client: TestClient) -> None:
        assert client.get("/students/999999").status_code == 404

    def test_creating_a_student_requires_authentication(self, client: TestClient) -> None:
        assert client.post("/students", json=NEW_STUDENT).status_code == 401

    def test_authenticated_creation_succeeds(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        response = client.post("/students", json=NEW_STUDENT, headers=auth)
        assert response.status_code == 201
        assert response.json()["student_no"] == NEW_STUDENT["student_no"]

    def test_invalid_payload_is_rejected(self, client: TestClient, auth: dict[str, str]) -> None:
        payload = {**NEW_STUDENT, "study_year": 99}
        assert client.post("/students", json=payload, headers=auth).status_code == 422


class TestCandidates:
    def test_candidates_are_ranked_and_explained(self, client: TestClient) -> None:
        body = client.get("/students/1/candidates?top=5").json()
        scores = [c["score"] for c in body["candidates"]]
        assert scores == sorted(scores, reverse=True)
        assert all(c["contributions"] for c in body["candidates"])
        assert all(c["summary"] for c in body["candidates"])

    def test_student_is_never_their_own_candidate(self, client: TestClient) -> None:
        body = client.get("/students/1/candidates?top=20").json()
        assert all(c["second_id"] != 1 for c in body["candidates"])

    def test_explain_false_omits_contributions(self, client: TestClient) -> None:
        body = client.get("/students/1/candidates?top=3&explain=false").json()
        assert all(not c["contributions"] for c in body["candidates"])

    def test_unknown_metric_is_400(self, client: TestClient) -> None:
        assert client.get("/students/1/candidates?metric=telepati").status_code == 400

    def test_unknown_student_is_404(self, client: TestClient) -> None:
        assert client.get("/students/999999/candidates").status_code == 404

    def test_top_is_validated(self, client: TestClient) -> None:
        assert client.get("/students/1/candidates?top=0").status_code == 422
        assert client.get("/students/1/candidates?top=1000").status_code == 422

    def test_custom_weights_change_the_ranking(self, client: TestClient) -> None:
        by_smoking = client.post(
            "/students/1/candidates", json={"weights": {"smoking": 5.0}, "top_n": 10}
        ).json()
        by_interests = client.post(
            "/students/1/candidates", json={"weights": {"interests": 5.0}, "top_n": 10}
        ).json()
        assert by_smoking["candidates"] != by_interests["candidates"]

    def test_invalid_weights_are_400(self, client: TestClient) -> None:
        assert (
            client.post("/students/1/candidates", json={"weights": {"yok": 1.0}}).status_code == 400
        )
        assert (
            client.post("/students/1/candidates", json={"weights": {"smoking": -1}}).status_code
            == 400
        )

    def test_relaxing_constraints_widens_the_pool(self, client: TestClient) -> None:
        strict = client.post("/students/1/candidates", json={"top_n": 50}).json()
        relaxed = client.post(
            "/students/1/candidates",
            json={
                "top_n": 50,
                "constraints": {
                    "separate_smokers": False,
                    "enforce_allergies": False,
                    "max_year_gap": None,
                    "max_smoking": None,
                },
            },
        ).json()
        assert relaxed["filtered_out"] <= strict["filtered_out"]

    def test_pair_comparison_explains_both_sides(self, client: TestClient) -> None:
        body = client.get("/students/1/compare/2").json()
        assert body["second_id"] == 2
        assert body["contributions"]
        assert all("first_value" in c and "second_value" in c for c in body["contributions"])

    def test_comparing_a_missing_student_is_404(self, client: TestClient) -> None:
        assert client.get("/students/1/compare/999999").status_code == 404


class TestAuth:
    def test_login_succeeds_with_correct_credentials(self, client: TestClient) -> None:
        response = client.post(
            "/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
        )
        assert response.status_code == 200
        assert response.json()["access_token"]

    def test_login_fails_with_a_wrong_password(self, client: TestClient) -> None:
        response = client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": "yanlis"})
        assert response.status_code == 401

    def test_login_does_not_reveal_whether_an_email_exists(self, client: TestClient) -> None:
        known = client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": "yanlis"})
        unknown = client.post(
            "/auth/login", json={"email": "yok@example.com", "password": "yanlis"}
        )
        assert known.status_code == unknown.status_code == 401
        assert known.json()["detail"] == unknown.json()["detail"]

    def test_me_requires_a_token(self, client: TestClient) -> None:
        assert client.get("/auth/me").status_code == 401

    def test_me_rejects_a_forged_token(self, client: TestClient) -> None:
        response = client.get("/auth/me", headers={"Authorization": "Bearer sahte.token.abc"})
        assert response.status_code == 401

    def test_me_returns_the_authenticated_account(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        assert client.get("/auth/me", headers=auth).json()["email"] == ADMIN_EMAIL

    def test_students_cannot_self_register(self, client: TestClient) -> None:
        """Students are data, not API users, so there is no registration endpoint."""
        assert client.post("/auth/register", json=NEW_STUDENT).status_code == 404


class TestAssignments:
    def test_running_an_assignment_requires_authentication(self, client: TestClient) -> None:
        assert client.post("/assignments/run", json={"limit": 10}).status_code == 401

    def test_assignment_places_everyone(self, client: TestClient, auth: dict[str, str]) -> None:
        body = client.post(
            "/assignments/run", json={"limit": 20, "save": False}, headers=auth
        ).json()
        assert body["room_count"] == 10
        assert body["unplaced"] == []

    def test_optimal_is_flagged_as_optimal(self, client: TestClient, auth: dict[str, str]) -> None:
        body = client.post(
            "/assignments/run", json={"limit": 10, "save": False}, headers=auth
        ).json()
        assert body["strategy"] == "optimal"
        assert body["optimal"] is True

    def test_optimal_scores_at_least_as_high_as_greedy(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        optimal = client.post(
            "/assignments/run",
            json={"limit": 24, "strategy": "optimal", "save": False},
            headers=auth,
        ).json()
        greedy = client.post(
            "/assignments/run",
            json={"limit": 24, "strategy": "greedy", "save": False},
            headers=auth,
        ).json()
        assert optimal["total_score"] >= greedy["total_score"] - 1e-9

    def test_odd_population_reports_one_unplaced(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        body = client.post(
            "/assignments/run", json={"limit": 11, "save": False}, headers=auth
        ).json()
        assert len(body["unplaced"]) == 1

    def test_unknown_strategy_is_400(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post("/assignments/run", json={"strategy": "yok"}, headers=auth)
        assert response.status_code == 400

    def test_too_few_students_is_409(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post("/assignments/run", json={"limit": 1}, headers=auth)
        assert response.status_code == 409

    def test_saved_plan_can_be_read_back(self, client: TestClient, auth: dict[str, str]) -> None:
        client.post("/assignments/run", json={"limit": 20, "save": True}, headers=auth)
        body = client.get("/assignments/latest?limit=5").json()
        assert body["room_count"] == 10
        assert len(body["rooms"]) == 5
        assert body["rooms"][0]["first_name"]

    def test_room_explanation_is_served(self, client: TestClient, auth: dict[str, str]) -> None:
        client.post("/assignments/run", json={"limit": 20, "save": True}, headers=auth)
        body = client.get("/assignments/latest/rooms/1").json()
        assert body["contributions"]
        assert body["percentage"] > 0

    def test_unknown_room_is_404(self, client: TestClient, auth: dict[str, str]) -> None:
        client.post("/assignments/run", json={"limit": 20, "save": True}, headers=auth)
        assert client.get("/assignments/latest/rooms/9999").status_code == 404


class TestRepositoryReset:
    """Re-seeding must not crash on a populated database."""

    def test_reset_clears_students_and_plans(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        """A second seed run would hit the primary key otherwise."""
        from roommate_matcher.data.database import SqlStudentRepository, create_db_engine

        engine = create_db_engine("sqlite:///:memory:")
        repository = SqlStudentRepository(engine)
        repository.bulk_add(generate_students(6, seed=1))
        repository.add_admin("a@b.c", hash_password("parola12345"))

        plan = RoomAssigner("optimal").assign(repository.list_all(), explain=False)
        repository.save_plan(plan, "weighted_gower")
        assert repository.latest_plan_row() is not None

        removed = repository.reset()

        assert removed == 6
        assert repository.count() == 0
        assert repository.latest_plan_row() is None
        # Staff accounts survive, so a reset cannot lock the operator out.
        assert repository.admin_count() == 1

    def test_reseeding_after_reset_succeeds(self) -> None:
        from roommate_matcher.data.database import SqlStudentRepository, create_db_engine

        repository = SqlStudentRepository(create_db_engine("sqlite:///:memory:"))
        repository.bulk_add(generate_students(4, seed=2))
        repository.reset()
        assert repository.bulk_add(generate_students(4, seed=2)) == 4
