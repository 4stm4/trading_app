"""
Password hashing and stateless access-token helpers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any


PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 120_000
TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "86400"))


def hash_password(password: str) -> str:
    raw_password = str(password or "")
    if not raw_password:
        raise ValueError("password is required")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        raw_password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return (
        f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}$"
        f"{_b64url_encode(salt)}${_b64url_encode(digest)}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, iterations_raw, salt_encoded, digest_encoded = str(password_hash or "").split("$", 3)
    except ValueError:
        return False
    if scheme != PASSWORD_SCHEME:
        return False
    try:
        iterations = int(iterations_raw)
        salt = _b64url_decode(salt_encoded)
        expected_digest = _b64url_decode(digest_encoded)
    except (TypeError, ValueError):
        return False

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        str(password or "").encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(digest, expected_digest)


def issue_access_token(*, user_email: str, expires_in_seconds: int | None = None) -> str:
    ttl = max(60, int(expires_in_seconds if expires_in_seconds is not None else TOKEN_TTL_SECONDS))
    now_ts = int(time.time())
    payload: dict[str, Any] = {
        "sub": str(user_email or "").strip().lower(),
        "iat": now_ts,
        "exp": now_ts + ttl,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_encoded = _b64url_encode(payload_bytes)
    signature = _sign(payload_encoded.encode("ascii"))
    signature_encoded = _b64url_encode(signature)
    return f"{payload_encoded}.{signature_encoded}"


def resolve_user_email_from_token(token: str) -> str | None:
    raw_token = str(token or "").strip()
    if not raw_token or "." not in raw_token:
        return None
    payload_encoded, signature_encoded = raw_token.split(".", 1)
    if not payload_encoded or not signature_encoded:
        return None
    try:
        signature = _b64url_decode(signature_encoded)
    except (TypeError, ValueError):
        return None
    expected_signature = _sign(payload_encoded.encode("ascii"))
    if not hmac.compare_digest(signature, expected_signature):
        return None
    try:
        payload_raw = _b64url_decode(payload_encoded)
        payload = json.loads(payload_raw.decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    email = str(payload.get("sub") or "").strip().lower()
    if not email:
        return None
    exp = payload.get("exp")
    try:
        exp_ts = int(exp)
    except (TypeError, ValueError):
        return None
    if exp_ts <= int(time.time()):
        return None
    return email


def parse_bearer_token(authorization: str | None) -> str | None:
    raw_header = str(authorization or "").strip()
    if not raw_header:
        return None
    prefix = "bearer "
    if raw_header.lower().startswith(prefix):
        token = raw_header[len(prefix):].strip()
        return token or None
    return None


def _sign(payload: bytes) -> bytes:
    secret = _token_secret()
    return hmac.new(secret, payload, hashlib.sha256).digest()


def _token_secret() -> bytes:
    raw = str(os.getenv("AUTH_TOKEN_SECRET") or "").strip()
    if raw:
        return raw.encode("utf-8")
    return b"trading-app-dev-secret-change-me"


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    raw = str(value or "").encode("ascii")
    padding = b"=" * ((4 - len(raw) % 4) % 4)
    return base64.urlsafe_b64decode(raw + padding)
