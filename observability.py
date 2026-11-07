"""Trazabilidad ligera por solicitud, sin introducir un sistema externo."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import time
import uuid
from collections.abc import Iterator
from typing import Any

from errors import LLMCLIError, safe_error_message
from models import CallMetrics, ErrorMetric, StageMetric


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def capture_usage(
    sink: dict[str, Any],
    response: Any,
    *,
    model: str | None = None,
    calls_key: str = "calls",
    prefix: str = "",
) -> dict[str, Any]:
    """Extrae usage_metadata del SDK y lo acumula sin asumir que siempre exista."""
    usage = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_token_count", 0)
        or (input_tokens + output_tokens)
    )
    key_prefix = f"{prefix}_" if prefix else ""
    sink[f"{key_prefix}input_tokens"] = sink.get(f"{key_prefix}input_tokens", 0) + input_tokens
    sink[f"{key_prefix}output_tokens"] = sink.get(f"{key_prefix}output_tokens", 0) + output_tokens
    sink[f"{key_prefix}total_tokens"] = sink.get(f"{key_prefix}total_tokens", 0) + total_tokens
    sink[f"{key_prefix}{calls_key}"] = sink.get(f"{key_prefix}{calls_key}", 0) + 1
    usage_key = f"{key_prefix}usage_available"
    sink[usage_key] = sink.get(usage_key, True) and usage is not None
    if model:
        sink[f"{key_prefix}model"] = model
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_available": usage is not None,
        "model": model,
    }


class RequestTrace:
    """Acumula etapas, errores y fallbacks de una solicitud completa."""

    def __init__(self) -> None:
        self.trace_id = f"req_{uuid.uuid4().hex[:12]}"
        self.started_at = utc_now_iso()
        self._started = time.perf_counter()
        self.stages: list[dict[str, Any]] = []
        self.errors: list[ErrorMetric] = []
        self.fallbacks: list[dict[str, Any]] = []

    @contextmanager
    def stage(self, name: str, **metadata: Any) -> Iterator[dict[str, Any]]:
        event: dict[str, Any] = {
            "name": name,
            "status": "ok",
            "started_at": utc_now_iso(),
            "metadata": metadata,
        }
        started = time.perf_counter()
        try:
            yield event
        except Exception as exc:
            event["status"] = "error"
            error = self._error_metric(exc, stage=name)
            event["error"] = error.model_dump()
            self.errors.append(error)
            raise
        finally:
            event["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self.stages.append(event)

    def add_fallback(self, component: str, reason: str, **metadata: Any) -> None:
        self.fallbacks.append({
            "component": component,
            "reason": safe_error_message(reason),
            "metadata": metadata,
            "timestamp": utc_now_iso(),
        })

    def add_error(self, exc: Exception, *, stage: str | None = None) -> None:
        self.errors.append(self._error_metric(exc, stage=stage))

    def apply_to_metrics(
        self,
        metrics: CallMetrics,
        *,
        route_source: str,
    ) -> CallMetrics:
        metrics.trace_id = self.trace_id
        metrics.request_started_at = self.started_at
        metrics.total_latency_ms = round((time.perf_counter() - self._started) * 1000, 2)
        metrics.route_source = route_source
        metrics.stage_metrics = [StageMetric.model_validate(stage) for stage in self.stages]
        metrics.errors = self.errors
        metrics.fallbacks = self.fallbacks
        return metrics

    def _error_metric(self, exc: Exception, *, stage: str | None) -> ErrorMetric:
        if isinstance(exc, LLMCLIError):
            data = exc.to_dict()
            return ErrorMetric(
                type=data["type"],
                code=data["code"],
                message=data["message"],
                retryable=data["retryable"],
                stage=stage,
                model=data.get("model"),
                status_code=data.get("status_code"),
            )
        return ErrorMetric(
            type=type(exc).__name__,
            code="unexpected_error",
            message=safe_error_message(str(exc)),
            retryable=False,
            stage=stage,
        )
