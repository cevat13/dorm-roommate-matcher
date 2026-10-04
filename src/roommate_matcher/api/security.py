"""Password hashing and JWT issuing.

Passwords are hashed with bcrypt and never stored or logged in clear text. Tokens
are signed with ``RM_JWT_SECRET``, which must be set in production; the development
default is named so that a misconfiguration is visible in ``/health``.

bcrypt is used directly rather than through passlib, which reads
``bcrypt.__about__`` and therefore breaks against current bcrypt releases.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from roommate_matcher.config import get_settings
from roommate_matcher.exceptions import AuthenticationError

INSECURE_DEFAULT_SECRET = "dev-only-insecure-secret-change-me"
BCRYPT_ROUNDS = 12


def _prepare(password: str) -> bytes:
    """Pre-hash a password so bcrypt's 72-byte input limit cannot truncate it.

    bcrypt ignores everything past the 72nd byte, which would make two different
    long passwords interchangeable. A fixed-length SHA-256 digest avoids the limit.

    Args:
        password: The plaintext password.

    Returns:
        A 44-byte base64-encoded digest, safe to pass to bcrypt.
    """
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    """Hash a plaintext password.

    Args:
        password: The plaintext password.

    Returns:
        The bcrypt hash, as an ASCII string.
    """
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    """Check a password against its hash in constant time.

    Args:
        password: The plaintext candidate.
        hashed: The stored hash.

    Returns:
        ``True`` when the password matches.
    """
    try:
        return bcrypt.checkpw(_prepare(password), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str | int, **claims: Any) -> str:
    """Issue a signed JWT.

    Args:
        subject: The user id the token identifies.
        **claims: Extra claims to embed.

    Returns:
        The encoded token.
    """
    settings = get_settings()
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
        **claims,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify and decode a JWT.

    Args:
        token: The encoded token.

    Returns:
        The decoded claims.

    Raises:
        AuthenticationError: If the token is expired, malformed or badly signed.
    """
    settings = get_settings()
    try:
        decoded: dict[str, Any] = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token süresi doldu.") from exc
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Geçersiz token.") from exc
    return decoded


def using_insecure_secret() -> bool:
    """Whether the process is still running on the development secret.

    Returns:
        ``True`` when the JWT secret has not been overridden.
    """
    return hmac.compare_digest(get_settings().jwt_secret, INSECURE_DEFAULT_SECRET)
