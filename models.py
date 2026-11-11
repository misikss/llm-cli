"""
models.py — Esquemas de validación con Pydantic para entradas, salidas y métricas.
"""

from typing import Any, Literal
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator


class RouteDecision(BaseModel):
    """Decisión de enrutamiento unificada y semántica."""
    route: Literal["general", "weather", "web_grounding", "rag_memory"] = Field(
        ...,
        description="Ruta seleccionada según la intención: "
                    "'web_grounding' para temas dinámicos, novedades, tendencias o estado del arte en tiempo real; "
                    "'weather' para meteorología; "
                    "'rag_memory' si requiere buscar recuerdos de conversaciones pasadas en ChromaDB; "
                    "'general' para explicaciones atemporales, razonamiento, creatividad o código directo."
    )
    is_ephemeral: bool = Field(
        default=False,
        description="True si la información solicitada es volátil/pasajera (clima de hoy, eventos en vivo, cotizaciones)."
    )
    needs_memory_rag: bool = Field(
        default=False,
        description="True ÚNICAMENTE si la consulta requiere buscar recuerdos en la base vectorial ChromaDB."
    )
    reason: str = Field(..., description="Justificación breve de la decisión de enrutamiento.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Nivel de certidumbre en la clasificación.")


class LLMResponse(BaseModel):
    """Estructura que esperamos que Gemini devuelva en cada respuesta."""
    answer: str = Field(
        ...,
        description="Respuesta principal a la pregunta del usuario"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Nivel de confianza del modelo (0.0 = nada seguro, 1.0 = muy seguro)"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Lista de puntos clave de la respuesta (máximo 5)"
    )
    category: str = Field(
        ...,
        description="Categoría de la pregunta: tecnología, ciencia, historia, cultura, clima, actualidad, otro"
    )
    sources_suggested: list[str] = Field(
        default_factory=list,
        description="Fuentes sugeridas o enlaces de grounding consultados"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return round(v, 2)

    @field_validator("key_points")
    @classmethod
    def limit_key_points(cls, v: list[str]) -> list[str]:
        return v[:5]


class ErrorMetric(BaseModel):
    """Error sanitizado asociado a una etapa de una solicitud."""

    type: str
    code: str
    message: str
    retryable: bool = False
    stage: str | None = None
    model: str | None = None
    status_code: int | None = None


class StageMetric(BaseModel):
    """Latencia y resultado de una etapa del pipeline."""

    name: str
    status: Literal["ok", "fallback", "error"] = "ok"
    started_at: str | None = None
    latency_ms: float = 0.0
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: ErrorMetric | None = None


class CallMetrics(BaseModel):
    """Métricas de uso y costo de una llamada a la API."""
    timestamp: str
    model: str
    question: str
    file_attached: str | None = None
    routed_as: str = "general"
    grounding_used: bool = False
    grounding_sources: list[str] = Field(default_factory=list)

    # Tokens exactos extraídos de usage_metadata (Auditoría completa)
    tokens_entrada: int
    tokens_salida: int
    tokens_thinking: int = 0
    tokens_query_eng_in: int = 0
    tokens_query_eng_out: int = 0
    tokens_rerank_in: int = 0
    tokens_rerank_out: int = 0
    tokens_router_in: int = 0
    tokens_router_out: int = 0
    tokens_memory_update_in: int = 0
    tokens_memory_update_out: int = 0
    tokens_embedding_in: int = 0
    tokens_embedding_out: int = 0
    embedding_calls: int = 0
    tokens_totales: int

    # Tarifas aplicadas (USD por 1M tokens)
    tarifa_input_por_millon: float
    tarifa_output_por_millon: float
    tarifa_thinking_por_millon: float = 0.0

    # Costos calculados (USD)
    costo_entrada_usd: float
    costo_salida_usd: float
    costo_thinking_usd: float = 0.0
    costo_rag_overhead_usd: float = 0.0
    costo_total_usd: float

    response: dict  # La respuesta validada como dict

    # Observabilidad de la solicitud completa
    event_type: Literal["llm_call"] = "llm_call"
    user_id: str = "local-user"
    session_id: str = "default-session"
    conversation_id: str = "default-conversation"
    trace_id: str = Field(default_factory=lambda: f"req_{uuid4().hex[:12]}")
    request_started_at: str | None = None
    total_latency_ms: float = 0.0
    route_source: Literal["llm", "manual", "fallback"] = "llm"
    token_usage_scope: Literal["generation", "pipeline", "pipeline_partial"] = "generation"
    stage_metrics: list[StageMetric] = Field(default_factory=list)
    errors: list[ErrorMetric] = Field(default_factory=list)
    fallbacks: list[dict[str, Any]] = Field(default_factory=list)

