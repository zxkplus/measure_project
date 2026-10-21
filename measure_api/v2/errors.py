"""Errors shared by process calculations and the HTTP adapter."""

from typing import Any, Dict, Optional


class ServiceError(Exception):
    """A stable error with an HTTP status and structured context."""

    def __init__(
        self, code: str, message: str, status: int = 422,
        details: Optional[Dict[str, Any]] = None, field_path: Optional[str] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}
        self.field_path = field_path
        self.retryable = retryable

    def payload(self, trace_id: str) -> Dict[str, Any]:
        """Return a JSON-safe public error envelope.

        Args:
            trace_id: Trace identity echoed in the public operation and error envelope.

        Returns:
            JSON-safe code, message, details, field path, retryability and trace identity.
        """
        return {
            "code": self.code, "message": self.message, "details": self.details,
            "field_path": self.field_path, "trace_id": trace_id,
            "retryable": self.retryable,
        }
