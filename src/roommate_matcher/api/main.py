"""FastAPI application factory.

Run locally with::

    uvicorn roommate_matcher.api.main:app --reload

Interactive documentation is then served at ``/docs``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from roommate_matcher.api.routes import (
    assignments_router,
    auth_router,
    health_router,
    students_router,
)
from roommate_matcher.api.security import using_insecure_secret
from roommate_matcher.config import get_settings
from roommate_matcher.exceptions import (
    EmptyCandidatePoolError,
    NoPlanError,
    RoommateMatcherError,
    StudentNotFoundError,
    UnknownMetricError,
    UnknownStrategyError,
)
from roommate_matcher.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
Yurt oda arkadaşı eşleştirme ve analiz motoru.

* **Ağırlıklı Gower** metriği ile karışık tipli öğrenci profillerini karşılaştırır.
* **Maksimum ağırlıklı eşleştirme** (Blossom) ile toplam uyumu maksimize eden
  kesin optimal yerleşimi hesaplar; herkes bir odaya yerleşir.
* Kısıt ihlalleri büyük ceza olarak modellenir, böylece çözüm her zaman bulunur ve
  kaçınılamayan ihlaller raporlanır.
* Her oda, hangi özelliğin ne kadar katkı verdiğini gösteren bir dökümle gelir.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare and tear down process-wide resources.

    Args:
        app: The application instance.

    Yields:
        Control back to the server while the app runs.
    """
    del app
    configure_logging()
    settings = get_settings()
    settings.ensure_directories()
    if using_insecure_secret():
        logger.warning("RM_JWT_SECRET is still the development default. Set it before deploying.")
    logger.info("%s v%s ready", settings.api_title, settings.api_version)
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Build the configured FastAPI application.

    Returns:
        The application instance.
    """
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=DESCRIPTION,
        lifespan=lifespan,
        contact={"name": "Roommate Matcher"},
        license_info={"name": "MIT"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(StudentNotFoundError)
    @app.exception_handler(NoPlanError)
    async def _not_found(request: Request, exc: RoommateMatcherError) -> JSONResponse:
        """Map a missing student or plan onto 404.

        Args:
            request: The incoming request.
            exc: The raised error.

        Returns:
            A 404 response.
        """
        del request
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc), "error_type": type(exc).__name__},
        )

    @app.exception_handler(UnknownMetricError)
    @app.exception_handler(UnknownStrategyError)
    @app.exception_handler(EmptyCandidatePoolError)
    async def _bad_request(request: Request, exc: RoommateMatcherError) -> JSONResponse:
        """Map caller mistakes onto 400.

        Args:
            request: The incoming request.
            exc: The raised error.

        Returns:
            A 400 response.
        """
        del request
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc), "error_type": type(exc).__name__},
        )

    @app.exception_handler(RoommateMatcherError)
    async def _domain_error(request: Request, exc: RoommateMatcherError) -> JSONResponse:
        """Catch-all for domain errors that have no more specific mapping.

        Args:
            request: The incoming request.
            exc: The raised error.

        Returns:
            A 422 response.
        """
        del request
        logger.exception("Unhandled domain error")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "error_type": type(exc).__name__},
        )

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(students_router)
    app.include_router(assignments_router)
    return app


app = create_app()
