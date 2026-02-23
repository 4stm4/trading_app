"""Error hierarchy for API service layer."""

from __future__ import annotations


class ApiServiceError(Exception):
    status_code = 500


class ApiValidationError(ApiServiceError):
    status_code = 400


class ApiNotFoundError(ApiServiceError):
    status_code = 404


class ApiUnauthorizedError(ApiServiceError):
    status_code = 401


class ApiConflictError(ApiServiceError):
    status_code = 409
