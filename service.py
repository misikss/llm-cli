"""Caso de uso de la aplicación, independiente de la interfaz CLI."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from google import genai

from client import determine_route, execute_llm_call
from config import GEMINI_API_KEY, GEMINI_MODEL
from errors import ConfigurationError, InvalidRequestError, RequestExecutionError
from memory import HybridMemory
from models import LLMResponse, CallMetrics, RouteDecision
from observability import RequestTrace
from rag import rewrite_query


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AskRequest:
    question: str
    file_path: str | None = None
    use_grounding: bool = False
    use_weather: bool = False


@dataclass(frozen=True)
class AskResult:
    response: LLMResponse
    metrics: CallMetrics
    route_decision: RouteDecision


class AssistantService:
    """Orquesta routing, retrieval, generación y actualización de memoria."""

    def __init__(
        self,
        *,
        client: genai.Client | None = None,
        memory: HybridMemory | None = None,
        db_path: str = "chroma_db",
    ) -> None:
        if client is None and not GEMINI_API_KEY:
            raise ConfigurationError(
                "GEMINI_API_KEY no está configurada. Configúrala en el archivo .env."
            )
        self.client = client or genai.Client(api_key=GEMINI_API_KEY)
        self.memory = memory or HybridMemory(db_path=db_path)

    def ask(self, request: AskRequest) -> AskResult:
        question = request.question.strip()
        if not question:
            raise InvalidRequestError("La pregunta no puede estar vacía.")

        trace = RequestTrace()
        route_source = "manual" if request.use_grounding or request.use_weather else "llm"

        try:
            active_subject = self.memory.active_subject
            recent_context = "; ".join(self.memory.key_facts)

            with trace.stage("route", source=route_source) as event:
                if request.use_weather:
                    route_decision = RouteDecision(
                        route="weather",
                        is_ephemeral=True,
                        needs_memory_rag=False,
                        reason="Flag manual --clima activado",
                    )
                elif request.use_grounding:
                    route_decision = RouteDecision(
                        route="web_grounding",
                        is_ephemeral=False,
                        needs_memory_rag=False,
                        reason="Flag manual --grounding activado",
                    )
                else:
                    route_decision = determine_route(
                        client=self.client,
                        question=question,
                        active_subject=active_subject,
                        recent_context=recent_context,
                    )
                    if route_decision.reason.startswith("Fallback"):
                        route_source = "fallback"
                        trace.add_fallback(
                            "semantic_router",
                            route_decision.reason,
                            selected_route=route_decision.route,
                        )
                event["metadata"].update({
                    "route": route_decision.route,
                    "confidence": route_decision.confidence,
                })

            rag_tokens = {
                "query_eng_in": 0,
                "query_eng_out": 0,
                "rerank_in": 0,
                "rerank_out": 0,
            }

            if route_decision.needs_memory_rag:
                with trace.stage("query_rewrite") as event:
                    rewritten, t_qe_in, t_qe_out = rewrite_query(
                        client=self.client,
                        question=question,
                        active_subject=active_subject,
                        recent_context=recent_context,
                    )
                    rag_tokens["query_eng_in"] = t_qe_in
                    rag_tokens["query_eng_out"] = t_qe_out
                    event["input_tokens"] = t_qe_in
                    event["output_tokens"] = t_qe_out

                with trace.stage("retrieval") as event:
                    context_prefix, t_rr_in, t_rr_out = self.memory.get_context_prompt(
                        self.client,
                        query=rewritten.rewritten_query,
                        keywords=rewritten.search_keywords,
                    )
                    rag_tokens["rerank_in"] = t_rr_in
                    rag_tokens["rerank_out"] = t_rr_out
                    event["input_tokens"] = t_rr_in
                    event["output_tokens"] = t_rr_out
            else:
                with trace.stage("working_memory") as event:
                    context_prefix = self.memory.get_working_memory_context()
                    event["metadata"]["has_context"] = bool(context_prefix)

            with trace.stage("generation") as event:
                validated, metrics = execute_llm_call(
                    question=question,
                    file_path=request.file_path,
                    use_grounding=request.use_grounding,
                    use_weather=request.use_weather,
                    route_decision=route_decision,
                    context_prefix=context_prefix,
                    rag_tokens=rag_tokens,
                    client=self.client,
                )
                event["model"] = metrics.model
                event["input_tokens"] = metrics.tokens_entrada
                event["output_tokens"] = metrics.tokens_salida
                if metrics.model != GEMINI_MODEL:
                    trace.add_fallback(
                        "generation",
                        "Se utilizó un modelo fallback.",
                        selected_model=metrics.model,
                    )

            with trace.stage("memory_update"):
                category = (
                    "actualidad_efimera"
                    if route_decision.is_ephemeral
                    else validated.category
                )
                self.memory.update(
                    self.client,
                    question,
                    validated.answer,
                    category=category,
                )

            metrics.routed_as = route_decision.route
            trace.apply_to_metrics(metrics, route_source=route_source)
            return AskResult(
                response=validated,
                metrics=metrics,
                route_decision=route_decision,
            )
        except Exception as exc:
            if not trace.errors:
                trace.add_error(exc, stage="request")
            logger.exception("request_failed trace_id=%s", trace.trace_id)
            raise RequestExecutionError(
                "No se pudo completar la solicitud. Consulta el trace_id para revisar el error.",
                trace_id=trace.trace_id,
                question=question,
                cause=exc,
            ) from exc
