"""Errores tipados y seguros para el flujo de la aplicación."""

from __future__ import annotations

import re
from typing import Any


_SECRET_PATTERN = re.compile(r"\b(?:AIza|sk-[A-Za-z0-9_-]{6})[A-Za-z0-9_-]*\b")


def safe_error_message(message: str) -> str:
    """Evita persistir API keys o tokens completos en logs."""
    return _SECRET_PATTERN.sub(lambda match: f"{match.group(0)[:6]}***", message)


class LLMCLIError(Exception):
    """Base para errores esperables del sistema."""

    code = "llm_cli_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool | None = None,
        model: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = safe_error_message(message)
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        self.model = model
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "model": self.model,
            "status_code": self.status_code,
        }


class ConfigurationError(LLMCLIError):
    code = "configuration_error"


class InvalidRequestError(LLMCLIError):
    code = "invalid_request"


class ProviderError(LLMCLIError):
    code = "provider_error"


class QuotaExceededError(ProviderError):
    code = "quota_exceeded"
    retryable = True


class TransientProviderError(ProviderError):
    code = "transient_provider_error"
    retryable = True


class ModelUnavailableError(ProviderError):
    code = "model_unavailable"
    retryable = True


class RequestExecutionError(LLMCLIError):
    code = "request_execution_error"

    def __init__(
        self,
        message: str,
        *,
        trace_id: str,
        question: str,
        user_id: str = "local-user",
        session_id: str = "default-session",
        conversation_id: str = "default-conversation",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.trace_id = trace_id
        self.question = question
        self.user_id = user_id
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update({
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
        })
        return data


class RetrievalError(LLMCLIError):
    code = "retrieval_error"


class MemoryPersistenceError(LLMCLIError):
    code = "memory_persistence_error"


class ToolExecutionError(LLMCLIError):
    code = "tool_execution_error"


def _status_code_from_message(message: str) -> int | None:
    match = re.search(r"\b(4\d\d|5\d\d)\b", message)
    return int(match.group(1)) if match else None


def classify_provider_error(exc: Exception, *, model: str | None = None) -> ProviderError:
    """Convierte errores del SDK en categorías útiles para fallback y observabilidad."""
    message = str(exc)
    upper_message = message.upper()
    status_code = _status_code_from_message(message)

    if "429" in upper_message or "RESOURCE_EXHAUSTED" in upper_message:
        return QuotaExceededError(
            "El proveedor agotó la cuota disponible.",
            model=model,
            status_code=429,
        )
    if status_code == 404 or "NOT_FOUND" in upper_message:
        return ModelUnavailableError(
            "El modelo solicitado no está disponible.",
            model=model,
            status_code=404,
        )
    if status_code in {408, 500, 502, 503, 504} or any(
        token in upper_message for token in ("TIMEOUT", "UNAVAILABLE", "CONNECTION RESET")
    ):
        return TransientProviderError(
            "El proveedor devolvió un error temporal.",
            model=model,
            status_code=status_code,
        )
    return ProviderError(
        "El proveedor rechazó o no pudo procesar la solicitud.",
        model=model,
        status_code=status_code,
    )
