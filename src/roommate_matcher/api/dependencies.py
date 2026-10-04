"""FastAPI dependency wiring.

Routers declare what they need and the construction happens here; tests override
these providers to inject their own repository.

Only dormitory staff authenticate. Students are records in the database, not API
users, so there is a single role and no self-registration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from roommate_matcher.api.security import decode_access_token
from roommate_matcher.data.database import AdminRow, SqlStudentRepository
from roommate_matcher.exceptions import AuthenticationError

_bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _repository_singleton() -> SqlStudentRepository:
    """Build the process-wide repository.

    Returns:
        The shared :class:`SqlStudentRepository`.
    """
    return SqlStudentRepository()


def get_repository() -> SqlStudentRepository:
    """Provide the student repository.

    Returns:
        The shared repository instance.
    """
    return _repository_singleton()


RepositoryDep = Annotated[SqlStudentRepository, Depends(get_repository)]


def get_current_admin(
    repository: RepositoryDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> AdminRow:
    """Resolve the authenticated staff account from the bearer token.

    Args:
        repository: The student repository, which also stores staff accounts.
        credentials: The parsed ``Authorization`` header, if present.

    Returns:
        The authenticated staff row.

    Raises:
        HTTPException: 401 when the token is missing, invalid or points nowhere.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bu işlem için yönetici girişi gerekiyor.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = decode_access_token(credentials.credentials)
        admin = repository.admin_by_id(int(claims["sub"]))
    except (AuthenticationError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    if admin is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yönetici hesabı bulunamadı.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return admin


CurrentAdminDep = Annotated[AdminRow, Depends(get_current_admin)]
