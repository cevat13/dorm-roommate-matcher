"""Application settings.

Every tunable value is defined here and can be overridden with an environment
variable (prefix ``RM_``) or a local ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


class Settings(BaseSettings):
    """Runtime configuration resolved from environment, ``.env`` and defaults."""

    model_config = SettingsConfigDict(
        env_prefix="RM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths -------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    generated_data_dir: Path = PROJECT_ROOT / "data" / "generated"

    # --- Storage -----------------------------------------------------------
    database_url: str = Field(
        default=f"sqlite:///{(PROJECT_ROOT / 'data' / 'dormitory.db').as_posix()}",
        description="SQLAlchemy database URL. Swap for postgresql+psycopg://... in production.",
    )
    seed_csv: Path = PROJECT_ROOT / "data" / "generated" / "students.csv"

    # --- Compatibility -----------------------------------------------------
    default_metric: str = "weighted_gower"
    default_top_n: int = 10
    max_top_n: int = 100
    score_precision: int = 4

    # --- Assignment --------------------------------------------------------
    default_strategy: str = "optimal"
    room_capacity: int = Field(
        default=2,
        ge=2,
        le=2,
        description="Beds per room. Fixed at two: the optimal solver is a matching.",
    )
    check_stability: bool = Field(
        default=True,
        description="Search the resulting plan for blocking pairs as a diagnostic.",
    )

    # --- Explainability ----------------------------------------------------
    explanation_top_factors: int = 4
    clash_threshold: float = 0.35
    match_threshold: float = 0.75

    # --- Synthetic data ----------------------------------------------------
    synthetic_student_count: int = 400
    random_seed: int = 42

    # --- Security ----------------------------------------------------------
    jwt_secret: str = Field(
        default="dev-only-insecure-secret-change-me",
        description="HMAC secret for JWTs; override via RM_JWT_SECRET in production.",
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    # --- API ---------------------------------------------------------------
    api_title: str = "Yurt Oda Eşleştirme API"
    api_version: str = "1.0.0"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8501"]

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    def ensure_directories(self) -> None:
        """Create the data directories that the app writes into."""
        for directory in (self.data_dir, self.generated_data_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Returns:
        A cached :class:`Settings` instance.
    """
    return Settings()
