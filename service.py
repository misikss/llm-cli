"""Caso de uso de la aplicación, independiente de la interfaz CLI."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from uuid import uuid4

from google import genai

from client import determine_route, execute_llm_call
from config import GEMINI_API_KEY, GEMINI_MODEL
from errors import ConfigurationError, InvalidRequestError, RequestExecutionError
from document_rag import DocumentRAG, choose_document_plan
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
    user_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None


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
        user_id: str = "local-user",
        session_id: str | None = None,
        conversation_id: str = "default-conversation",
    ) -> None:
        if client is None and not GEMINI_API_KEY:
            raise ConfigurationError(
                "GEMINI_API_KEY no está configurada. Configúrala en el archivo .env."
            )
        self.client = client or genai.Client(api_key=GEMINI_API_KEY)
        self.db_path = db_path
        self.user_id = user_id
        self.session_id = session_id or f"sess_{uuid4().hex[:12]}"
        self.conversation_id = conversation_id
        self._memory_registry: dict[tuple[str, str, str], HybridMemory] = {}
        self.document_rag = DocumentRAG(db_path=db_path)
        self.memory = memory or self._create_memory(
            user_id=self.user_id,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
        )
        self._memory_registry[self._memory_key(
            self.user_id,
            self.session_id,
            self.conversation_id,
        )] = self.memory

    @staticmethod
    def _memory_key(user_id: str, session_id: str, conversation_id: str) -> tuple[str, str, str]:
        return user_id, session_id, conversation_id

    def _create_memory(
        self,
        *,
        user_id: str,
        session_id: str,
        conversation_id: str,
    ) -> HybridMemory:
        return HybridMemory(
            db_path=self.db_path,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
        )

    def _resolve_context(self, request: AskRequest) -> tuple[str, str, str]:
        return (
            request.user_id or self.user_id,
            request.session_id or self.session_id,
            request.conversation_id or self.conversation_id,
        )

    def _memory_for(self, context: tuple[str, str, str]) -> HybridMemory:
        if context not in self._memory_registry:
            self._memory_registry[context] = self._create_memory(
                user_id=context[0],
                session_id=context[1],
                conversation_id=context[2],
            )
        return self._memory_registry[context]

    def ask(self, request: AskRequest) -> AskResult:
        question = request.question.strip()
        if not question:
            raise InvalidRequestError("La pregunta no puede estar vacía.")

        user_id, session_id, conversation_id = self._resolve_context(request)
        memory = self._memory_for((user_id, session_id, conversation_id))
        trace = RequestTrace(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
        )
        route_source = "manual" if request.use_grounding or request.use_weather else "llm"

        try:
            active_subject = memory.active_subject
            recent_context = "; ".join(memory.key_facts)
            route_usage: dict = {}

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
                        usage_sink=route_usage,
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
                event["model"] = route_usage.get("model")
                event["input_tokens"] = route_usage.get("input_tokens", 0)
                event["output_tokens"] = route_usage.get("output_tokens", 0)

            document_plan = None
            document_context = ""
            document_usage: dict = {}
            if request.file_path:
                try:
                    with trace.stage("document_analysis") as event:
                        document_plan = choose_document_plan(
                            request.file_path,
                            user_id=user_id,
                        )
                        route_decision = route_decision.model_copy(update={
                            "document_mode": document_plan.mode,
                            "document_reason": document_plan.reason,
                        })
                        event["metadata"].update({
                            "mode": document_plan.mode,
                            "document_id": document_plan.document_id,
                            "file_size_bytes": document_plan.file_size_bytes,
                            "previous_queries": document_plan.previous_queries,
                        })
                        if document_plan.mode == "document_rag":
                            retrieved = self.document_rag.retrieve(
                                self.client,
                                request.file_path,
                                plan=document_plan,
                                user_id=user_id,
                                conversation_id=conversation_id,
                                query=question,
                                usage_sink=document_usage,
                            )
                            document_context = retrieved.context
                            event["metadata"].update({
                                "chunks_indexed": retrieved.chunks_indexed,
                                "chunks_retrieved": retrieved.chunks_retrieved,
                                "retrieval_mode": retrieved.retrieval_mode,
                                "reranker_used": retrieved.reranker_used,
                                "vector_candidates": retrieved.vector_candidates,
                                "bm25_candidates": retrieved.bm25_candidates,
                                "source_ids": retrieved.source_ids,
                                "rerank_input_tokens": document_usage.get("rerank_input_tokens", 0),
                                "rerank_output_tokens": document_usage.get("rerank_output_tokens", 0),
                            })
                            event["model"] = "gemini-embedding-2"
                            event["input_tokens"] = document_usage.get("embedding_input_tokens", 0)
                            event["output_tokens"] = document_usage.get("embedding_output_tokens", 0)
                        else:
                            event["metadata"]["delivery"] = "archivo completo al modelo multimodal"
                except Exception as exc:
                    # Si la extracción/indexación local no está disponible, el archivo
                    # conserva el camino multimodal original para no romper la consulta.
                    trace.add_fallback(
                        "document_rag",
                        "Se usó análisis multimodal directo porque el retrieval documental falló.",
                        error_type=type(exc).__name__,
                    )
                    document_context = ""
                    document_usage = {}
                    if document_plan is not None:
                        route_decision = route_decision.model_copy(update={
                            "document_mode": "multimodal_direct",
                            "document_reason": "Fallback multimodal por fallo en extracción o retrieval documental.",
                        })

            rag_tokens = {
                "query_eng_in": 0,
                "query_eng_out": 0,
                "rerank_in": document_usage.get("rerank_input_tokens", 0),
                "rerank_out": document_usage.get("rerank_output_tokens", 0),
            }
            rewrite_usage: dict = {}
            retrieval_usage: dict = {}

            if route_decision.needs_memory_rag:
                with trace.stage("query_rewrite") as event:
                    rewritten, t_qe_in, t_qe_out = rewrite_query(
                        client=self.client,
                        question=question,
                        active_subject=active_subject,
                        recent_context=recent_context,
                        usage_sink=rewrite_usage,
                    )
                    rag_tokens["query_eng_in"] = t_qe_in
                    rag_tokens["query_eng_out"] = t_qe_out
                    event["input_tokens"] = t_qe_in
                    event["output_tokens"] = t_qe_out
                    event["model"] = rewrite_usage.get("model")

                with trace.stage("retrieval") as event:
                    context_prefix, t_rr_in, t_rr_out = memory.get_context_prompt(
                        self.client,
                        query=rewritten.rewritten_query,
                        keywords=rewritten.search_keywords,
                        usage_sink=retrieval_usage,
                    )
                    rag_tokens["rerank_in"] += t_rr_in
                    rag_tokens["rerank_out"] += t_rr_out
                    event["input_tokens"] = t_rr_in
                    event["output_tokens"] = t_rr_out
                    event["model"] = retrieval_usage.get("model")
                    event["metadata"].update({
                        "embedding_calls": retrieval_usage.get("embedding_calls", 0),
                        "embedding_input_tokens": retrieval_usage.get("embedding_input_tokens", 0),
                        "embedding_output_tokens": retrieval_usage.get("embedding_output_tokens", 0),
                        "embedding_usage_available": retrieval_usage.get("embedding_usage_available"),
                    })
            else:
                with trace.stage("working_memory") as event:
                    context_prefix = memory.get_working_memory_context()
                    event["metadata"]["has_context"] = bool(context_prefix)

            if document_context:
                context_prefix = (
                    "[Contexto recuperado del documento adjunto]\n"
                    + document_context
                    + "\n\n"
                    + context_prefix
                )

            with trace.stage("generation") as event:
                validated, metrics = execute_llm_call(
                    question=question,
                    file_path=(
                        None
                        if route_decision.document_mode == "document_rag"
                        else request.file_path
                    ),
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

            memory_usage: dict = {}
            with trace.stage("memory_update") as event:
                category = (
                    "actualidad_efimera"
                    if route_decision.is_ephemeral
                    else validated.category
                )
                memory.update(
                    self.client,
                    question,
                    validated.answer,
                    category=category,
                    usage_sink=memory_usage,
                )
                event["model"] = memory_usage.get("llm_model")
                event["input_tokens"] = memory_usage.get("llm_input_tokens", 0)
                event["output_tokens"] = memory_usage.get("llm_output_tokens", 0)
                event["metadata"].update({
                    "embedding_calls": memory_usage.get("embedding_calls", 0),
                    "embedding_input_tokens": memory_usage.get("embedding_input_tokens", 0),
                    "embedding_output_tokens": memory_usage.get("embedding_output_tokens", 0),
                    "embedding_usage_available": memory_usage.get("embedding_usage_available"),
                })

            metrics.routed_as = route_decision.route
            metrics.tokens_router_in = route_usage.get("input_tokens", 0)
            metrics.tokens_router_out = route_usage.get("output_tokens", 0)
            metrics.tokens_memory_update_in = memory_usage.get("llm_input_tokens", 0)
            metrics.tokens_memory_update_out = memory_usage.get("llm_output_tokens", 0)
            metrics.tokens_embedding_in = (
                retrieval_usage.get("embedding_input_tokens", 0)
                if route_decision.needs_memory_rag
                else 0
            ) + memory_usage.get("embedding_input_tokens", 0)
            metrics.tokens_embedding_out = (
                retrieval_usage.get("embedding_output_tokens", 0)
                if route_decision.needs_memory_rag
                else 0
            ) + memory_usage.get("embedding_output_tokens", 0)
            metrics.embedding_calls = (
                retrieval_usage.get("embedding_calls", 0)
                if route_decision.needs_memory_rag
                else 0
            ) + memory_usage.get("embedding_calls", 0) + document_usage.get("embedding_calls", 0)
            metrics.tokens_embedding_in += document_usage.get("embedding_input_tokens", 0)
            metrics.tokens_embedding_out += document_usage.get("embedding_output_tokens", 0)
            metrics.tokens_totales += (
                metrics.tokens_router_in
                + metrics.tokens_router_out
                + metrics.tokens_memory_update_in
                + metrics.tokens_memory_update_out
                + metrics.tokens_embedding_in
                + metrics.tokens_embedding_out
            )
            usage_available = all([
                route_usage.get("usage_available", True),
                rewrite_usage.get("usage_available", True),
                memory_usage.get("llm_usage_available", True),
                retrieval_usage.get("usage_available", True)
                if route_decision.needs_memory_rag
                else True,
                memory_usage.get("embedding_usage_available", True),
                retrieval_usage.get("embedding_usage_available", True)
                if route_decision.needs_memory_rag
                else True,
                document_usage.get("embedding_usage_available", True)
                if document_plan is not None and document_plan.mode == "document_rag"
                else True,
                document_usage.get("rerank_usage_available", True)
                if document_usage.get("reranker_calls", 0) > 0
                else True,
            ])
            metrics.token_usage_scope = "pipeline" if usage_available else "pipeline_partial"
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
                user_id=user_id,
                session_id=session_id,
                conversation_id=conversation_id,
                cause=exc,
            ) from exc
