"""HTTP routes.

Endpoints validate, delegate and serialise. The matching logic itself lives in
:mod:`roommate_matcher.matching`.

Reading is open; computing or storing a plan requires staff authentication.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from roommate_matcher.api.dependencies import CurrentAdminDep, RepositoryDep
from roommate_matcher.api.schemas import (
    AdminOut,
    AssignRequest,
    CandidateQuery,
    CandidateResponse,
    HealthResponse,
    LoginRequest,
    NamedInfo,
    PairOut,
    PlanResponse,
    RoomOut,
    StudentCreate,
    StudentOut,
    TokenResponse,
)
from roommate_matcher.api.security import (
    create_access_token,
    using_insecure_secret,
    verify_password,
)
from roommate_matcher.config import get_settings
from roommate_matcher.domain.preferences import MatchPreferences
from roommate_matcher.exceptions import (
    DuplicateStudentError,
    RoommateMatcherError,
    StudentNotFoundError,
)
from roommate_matcher.matching.assignment import (
    RoomAssigner,
    available_strategies,
    get_strategy,
)
from roommate_matcher.matching.engine import CompatibilityEngine
from roommate_matcher.matching.metrics import available_metrics, get_metric

health_router = APIRouter(tags=["health"])
auth_router = APIRouter(prefix="/auth", tags=["auth"])
students_router = APIRouter(prefix="/students", tags=["students"])
assignments_router = APIRouter(prefix="/assignments", tags=["assignments"])


# --------------------------------------------------------------------------- #
# Health and metadata
# --------------------------------------------------------------------------- #


@health_router.get("/health", response_model=HealthResponse, summary="Servis durumu")
def health(repository: RepositoryDep) -> HealthResponse:
    """Report service health and current configuration.

    Args:
        repository: The student repository.

    Returns:
        A health summary including a warning flag for the development JWT secret.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.api_version,
        students=repository.count(),
        admins=repository.admin_count(),
        metrics=available_metrics(),
        strategies=available_strategies(),
        default_metric=settings.default_metric,
        default_strategy=settings.default_strategy,
        room_capacity=settings.room_capacity,
        insecure_secret=using_insecure_secret(),
    )


@health_router.get("/metrics", response_model=list[NamedInfo], summary="Uyum metrikleri")
def list_metrics() -> list[NamedInfo]:
    """List every registered compatibility metric.

    Returns:
        One entry per metric; ``flag`` marks the explainable ones.
    """
    return [
        NamedInfo(
            key=metric.key,
            label=metric.label,
            description=metric.description,
            flag=metric.explainable,
        )
        for metric in (get_metric(key) for key in available_metrics())
    ]


@health_router.get("/strategies", response_model=list[NamedInfo], summary="Yerleşim stratejileri")
def list_strategies() -> list[NamedInfo]:
    """List every registered assignment strategy.

    Returns:
        One entry per strategy; ``flag`` marks the provably optimal ones.
    """
    return [
        NamedInfo(
            key=strategy.key,
            label=strategy.label,
            description=strategy.description,
            flag=strategy.optimal,
        )
        for strategy in (get_strategy(key) for key in available_strategies())
    ]


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


@auth_router.post("/login", response_model=TokenResponse, summary="Yönetici girişi")
def login(payload: LoginRequest, repository: RepositoryDep) -> TokenResponse:
    """Exchange staff credentials for an access token.

    Args:
        payload: Email and password.
        repository: The student repository.

    Returns:
        A freshly issued token.

    Raises:
        HTTPException: 401 when the credentials do not match.
    """
    admin = repository.find_admin(str(payload.email).strip().lower())
    # One message for both branches, so the response cannot be used to enumerate
    # registered email addresses.
    if admin is None or not verify_password(payload.password, admin.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="E-posta veya parola hatalı."
        )
    settings = get_settings()
    return TokenResponse(
        access_token=create_access_token(admin.id),
        admin_id=int(admin.id),
        expires_in_minutes=settings.jwt_expire_minutes,
    )


@auth_router.get("/me", response_model=AdminOut, summary="Oturum bilgisi")
def me(current_admin: CurrentAdminDep) -> AdminOut:
    """Return the authenticated staff account.

    Args:
        current_admin: The authenticated staff row.

    Returns:
        The caller's account.
    """
    return AdminOut(admin_id=int(current_admin.id), email=str(current_admin.email))


# --------------------------------------------------------------------------- #
# Students
# --------------------------------------------------------------------------- #


@students_router.get("", response_model=list[StudentOut], summary="Öğrencileri listele")
def list_students(
    repository: RepositoryDep,
    study_year: int | None = Query(default=None, ge=1, le=6, description="Sınıfa göre filtrele"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[StudentOut]:
    """List students with optional filtering and pagination.

    Args:
        repository: The student repository.
        study_year: Optional study-year filter.
        limit: Page size.
        offset: How many records to skip.

    Returns:
        The requested page of students.
    """
    students = repository.list_all()
    if study_year is not None:
        students = [s for s in students if s.study_year == study_year]
    return [StudentOut.from_profile(s) for s in students[offset : offset + limit]]


@students_router.get("/{student_id}", response_model=StudentOut, summary="Tek öğrenci")
def get_student(student_id: int, repository: RepositoryDep) -> StudentOut:
    """Fetch a single student.

    Args:
        student_id: The id to fetch.
        repository: The student repository.

    Returns:
        The requested record.

    Raises:
        HTTPException: 404 when the id does not exist.
    """
    try:
        return StudentOut.from_profile(repository.get(student_id))
    except StudentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@students_router.post(
    "",
    response_model=StudentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Öğrenci ekle",
)
def create_student(
    payload: StudentCreate,
    repository: RepositoryDep,
    current_admin: CurrentAdminDep,
) -> StudentOut:
    """Add a student record.

    Args:
        payload: The student payload.
        repository: The student repository.
        current_admin: The authenticated staff account.

    Returns:
        The created record.

    Raises:
        HTTPException: 409 when the id or student number already exists.
    """
    del current_admin
    student = payload.to_profile(repository.next_id())
    try:
        repository.add(student)
    except DuplicateStudentError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return StudentOut.from_profile(student)


def _preferences_from(
    weights: dict[str, float] | None,
    constraints: object | None,
) -> MatchPreferences:
    """Build preferences from optional request overrides.

    Args:
        weights: Per-dimension weight overrides.
        constraints: Constraint overrides.

    Returns:
        The assembled preferences.

    Raises:
        HTTPException: 400 when the weights are invalid.
    """
    preferences = MatchPreferences()
    try:
        if weights is not None:
            preferences = MatchPreferences(weights=weights, constraints=preferences.constraints)
        if constraints is not None:
            preferences = preferences.model_copy(update={"constraints": constraints})
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return preferences


@students_router.get(
    "/{student_id}/candidates",
    response_model=CandidateResponse,
    summary="Uyumlu adayları getir",
)
def get_candidates(
    student_id: int,
    repository: RepositoryDep,
    metric: str | None = Query(default=None, description="Kullanılacak metrik"),
    top: int = Query(default=10, ge=1, le=100),
    explain: bool = Query(default=True, description="Açıklama üret"),
) -> CandidateResponse:
    """Rank the most compatible roommates for one student.

    Args:
        student_id: The student to analyse.
        repository: The student repository.
        metric: Optional metric override.
        top: How many candidates to return.
        explain: Whether to include per-dimension breakdowns.

    Returns:
        Ranked candidates and diagnostics.
    """
    return _run_candidates(
        student_id, repository, CandidateQuery(metric=metric, top_n=top, explain=explain)
    )


@students_router.post(
    "/{student_id}/candidates",
    response_model=CandidateResponse,
    summary="Özel ağırlıklarla adayları getir",
)
def post_candidates(
    student_id: int,
    query: CandidateQuery,
    repository: RepositoryDep,
) -> CandidateResponse:
    """Rank candidates using caller-supplied weights and constraints.

    Args:
        student_id: The student to analyse.
        query: Metric, weights, constraints and page size.
        repository: The student repository.

    Returns:
        Ranked candidates and diagnostics.
    """
    return _run_candidates(student_id, repository, query)


def _run_candidates(
    student_id: int,
    repository: RepositoryDep,
    query: CandidateQuery,
) -> CandidateResponse:
    """Shared implementation behind the GET and POST candidate endpoints.

    Args:
        student_id: The student to analyse.
        repository: The student repository.
        query: Metric, weights, constraints and page size.

    Returns:
        Ranked candidates and diagnostics.

    Raises:
        HTTPException: 404 for an unknown student, 400 for an unknown metric or
            invalid weights.
    """
    try:
        student = repository.get(student_id)
    except StudentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    preferences = _preferences_from(query.weights, query.constraints)
    try:
        engine = CompatibilityEngine(query.metric or get_settings().default_metric, preferences)
    except RoommateMatcherError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    report = engine.candidates_for(
        student, repository.others(student_id), top_n=query.top_n, explain=query.explain
    )
    return CandidateResponse(
        student_id=student_id,
        metric=report.metric,
        considered=report.considered,
        filtered_out=report.filtered_out,
        rejections=report.rejections,
        candidates=[PairOut.from_pair(pair) for pair in report.results],
    )


@students_router.get(
    "/{student_id}/compare/{other_id}",
    response_model=PairOut,
    summary="İki öğrenciyi karşılaştır",
)
def compare_students(
    student_id: int,
    other_id: int,
    repository: RepositoryDep,
    metric: str | None = Query(default=None),
) -> PairOut:
    """Explain the compatibility of one specific pair.

    Args:
        student_id: The first student.
        other_id: The second student.
        repository: The student repository.
        metric: Optional metric override.

    Returns:
        The scored pair with its full breakdown.

    Raises:
        HTTPException: 404 when either id does not exist.
    """
    try:
        first = repository.get(student_id)
        second = repository.get(other_id)
    except StudentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    engine = CompatibilityEngine(metric or get_settings().default_metric)
    return PairOut.from_pair(engine.explain_pair(first, second))


# --------------------------------------------------------------------------- #
# Assignments
# --------------------------------------------------------------------------- #


@assignments_router.post("/run", response_model=PlanResponse, summary="Yerleşimi hesapla")
def run_assignment(
    payload: AssignRequest,
    repository: RepositoryDep,
    current_admin: CurrentAdminDep,
) -> PlanResponse:
    """Compute a room plan for the population.

    Args:
        payload: Strategy, metric, weights and constraints.
        repository: The student repository.
        current_admin: The authenticated staff account.

    Returns:
        The resulting plan.

    Raises:
        HTTPException: 400 for an unknown strategy or metric, 409 when there are
            not enough students to form a room.
    """
    del current_admin
    settings = get_settings()
    students = repository.list_all()
    if payload.limit:
        students = students[: payload.limit]
    if len(students) < 2:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Yerleşim için en az iki öğrenci gerekiyor.",
        )

    preferences = _preferences_from(payload.weights, payload.constraints)
    try:
        engine = CompatibilityEngine(payload.metric or settings.default_metric, preferences)
        assigner = RoomAssigner(payload.strategy or settings.default_strategy, engine)
    except RoommateMatcherError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    plan = assigner.assign(students)
    if payload.save:
        repository.save_plan(plan, engine.metric.key)
    return PlanResponse.from_plan(plan)


@assignments_router.get("/latest", response_model=PlanResponse, summary="Son plan")
def latest_plan(
    repository: RepositoryDep,
    limit: int = Query(default=50, ge=1, le=1000, description="Kaç oda döndürülsün"),
) -> PlanResponse:
    """Return the most recently stored plan.

    Args:
        repository: The student repository.
        limit: How many rooms to include.

    Returns:
        The stored plan's figures and a page of its rooms.

    Raises:
        HTTPException: 404 when no plan has been computed yet.
    """
    row = repository.latest_plan_row()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Henüz bir yerleşim planı hesaplanmadı.",
        )
    stored = repository.plan_rooms(int(row.id))[:limit]
    by_id = {s.student_id: s for s in repository.list_all()}
    rooms = [
        RoomOut(
            room_no=item.room_no,
            first_id=item.first_id,
            second_id=item.second_id,
            first_name=by_id[item.first_id].display_name if item.first_id in by_id else "",
            second_name=by_id[item.second_id].display_name if item.second_id in by_id else "",
            score=item.score,
            percentage=round(item.score * 100),
            violations=[v for v in item.violations.split(",") if v],
            summary="",
        )
        for item in stored
    ]
    unplaced = [int(value) for value in str(row.unplaced).split(",") if value]
    return PlanResponse(
        strategy=str(row.strategy),
        optimal=str(row.strategy) == "optimal",
        room_count=int(row.room_count),
        total_score=float(row.total_score),
        mean_score=round(float(row.total_score) / max(int(row.room_count), 1), 6),
        min_score=float(row.min_score),
        unplaced=unplaced,
        violations={},
        blocking_pair_count=int(row.blocking_pair_count),
        is_stable=int(row.blocking_pair_count) == 0,
        seconds=float(row.seconds),
        rooms=rooms,
    )


@assignments_router.get(
    "/latest/rooms/{room_no}", response_model=PairOut, summary="Bir odanın gerekçesi"
)
def explain_room(room_no: int, repository: RepositoryDep) -> PairOut:
    """Explain why one room of the stored plan holds those two students.

    Args:
        room_no: The room to explain.
        repository: The student repository.

    Returns:
        The scored pair with its full breakdown.

    Raises:
        HTTPException: 404 when no plan exists or the room is not in it.
    """
    row = repository.latest_plan_row()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Henüz bir yerleşim planı hesaplanmadı.",
        )
    stored = {item.room_no: item for item in repository.plan_rooms(int(row.id))}
    if room_no not in stored:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Oda {room_no} bu planda yok."
        )
    target = stored[room_no]
    try:
        first = repository.get(target.first_id)
        second = repository.get(target.second_id)
    except StudentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    engine = CompatibilityEngine(str(row.metric))
    return PairOut.from_pair(engine.explain_pair(first, second))
