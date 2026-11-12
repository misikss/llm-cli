import unittest
from unittest.mock import patch

from errors import QuotaExceededError, classify_provider_error
from memory import EpisodicMemory, HybridMemory
from models import CallMetrics, LLMResponse, RouteDecision
from rag import RetrievalDocument
from service import AskRequest, AssistantService


class FakeMemory:
    active_subject = ""
    key_facts: list[str] = []

    def __init__(self) -> None:
        self.updated = False

    def get_working_memory_context(self) -> str:
        return ""

    def update(self, *args, **kwargs) -> None:
        self.updated = True


class FakeEpisodicMemory:
    def __init__(self, vector_documents: list[RetrievalDocument]) -> None:
        self._documents = [
            RetrievalDocument(id="memory-1", text="alpha project"),
            RetrievalDocument(id="memory-2", text="beta project"),
        ]
        self._vector_documents = vector_documents

    def get_bm25_index(self):
        class FakeBM25Index:
            documents = self._documents

            @staticmethod
            def search(*args, **kwargs) -> list[str]:
                return ["memory-1"]

        return FakeBM25Index()

    def retrieve_vector_candidates(self, *args, **kwargs) -> list[RetrievalDocument]:
        return self._vector_documents


def fake_metrics() -> CallMetrics:
    return CallMetrics(
        timestamp="2026-11-06T00:00:00+00:00",
        model="gemini-test",
        question="hello",
        tokens_entrada=10,
        tokens_salida=5,
        tokens_totales=15,
        tarifa_input_por_millon=0.075,
        tarifa_output_por_millon=0.30,
        costo_entrada_usd=0.000001,
        costo_salida_usd=0.000002,
        costo_total_usd=0.000003,
        response={"answer": "ok"},
    )


class ServiceTests(unittest.TestCase):
    def test_memory_reranker_is_skipped_when_rankings_agree(self) -> None:
        memory = HybridMemory.__new__(HybridMemory)
        memory.active_subject = ""
        memory.key_facts = []
        memory.episodic = FakeEpisodicMemory([
            RetrievalDocument(id="memory-1", text="alpha project"),
            RetrievalDocument(id="memory-2", text="beta project"),
        ])
        usage: dict = {}

        with patch("rag.rerank_documents") as rerank:
            context, tokens_in, tokens_out = memory.get_context_prompt(
                object(),
                query="alpha",
                usage_sink=usage,
            )

        rerank.assert_not_called()
        self.assertIn("Recuerdos híbridos", context)
        self.assertEqual((tokens_in, tokens_out), (0, 0))
        self.assertFalse(usage["reranker_used"])
        self.assertEqual(usage["retrieval_mode"], "hybrid_rrf")

    def test_memory_reranker_runs_when_rankings_disagree(self) -> None:
        memory = HybridMemory.__new__(HybridMemory)
        memory.active_subject = ""
        memory.key_facts = []
        memory.episodic = FakeEpisodicMemory([
            RetrievalDocument(id="memory-2", text="beta project"),
            RetrievalDocument(id="memory-1", text="alpha project"),
        ])
        usage: dict = {}

        with patch(
            "rag.rerank_documents",
            return_value=(["alpha project"], 7, 3),
        ) as rerank:
            context, tokens_in, tokens_out = memory.get_context_prompt(
                object(),
                query="alpha",
                usage_sink=usage,
            )

        rerank.assert_called_once()
        self.assertIn("re-rankeados", context)
        self.assertEqual((tokens_in, tokens_out), (7, 3))
        self.assertTrue(usage["reranker_used"])
        self.assertEqual(usage["retrieval_mode"], "hybrid_rrf_reranked")

    def test_context_limit_error_is_detected_through_exception_chain(self) -> None:
        root = ValueError("input token count exceeds the maximum number of tokens")
        wrapped = RuntimeError("provider rejected the request")
        wrapped.__cause__ = root

        self.assertTrue(AssistantService._is_context_limit_error(wrapped))

    def test_request_context_can_override_service_defaults(self) -> None:
        service = AssistantService(
            client=object(),
            memory=FakeMemory(),
            user_id="user-default",
            session_id="session-default",
            conversation_id="conversation-default",
        )

        self.assertEqual(
            service._resolve_context(AskRequest(
                question="hello",
                user_id="user-2",
                session_id="session-2",
                conversation_id="conversation-2",
            )),
            ("user-2", "session-2", "conversation-2"),
        )

    def test_episodic_scope_does_not_filter_by_session(self) -> None:
        memory = EpisodicMemory.__new__(EpisodicMemory)
        memory.user_id = "user-1"
        memory.session_id = "session-2"
        memory.conversation_id = "conversation-1"

        self.assertEqual(
            memory._scope_where(),
            {"$and": [{"user_id": "user-1"}, {"conversation_id": "conversation-1"}]},
        )

    def test_service_orchestrates_request_without_cli(self) -> None:
        memory = FakeMemory()
        response = LLMResponse(
            answer="ok",
            confidence=0.9,
            category="general",
        )
        route = RouteDecision(
            route="general",
            reason="test",
            needs_memory_rag=False,
        )

        with patch("service.determine_route", return_value=route), patch(
            "service.execute_llm_call",
            return_value=(response, fake_metrics()),
        ):
            result = AssistantService(
                client=object(),
                memory=memory,
                user_id="user-1",
                session_id="session-1",
                conversation_id="conversation-1",
            ).ask(
                AskRequest(question="hello")
            )

        self.assertEqual(result.response.answer, "ok")
        self.assertTrue(memory.updated)
        self.assertTrue(result.metrics.trace_id.startswith("req_"))
        self.assertEqual(result.metrics.user_id, "user-1")
        self.assertEqual(result.metrics.session_id, "session-1")
        self.assertEqual(result.metrics.conversation_id, "conversation-1")
        self.assertEqual(
            [stage.name for stage in result.metrics.stage_metrics],
            ["route", "working_memory", "generation", "memory_update"],
        )

    def test_provider_quota_errors_are_classified(self) -> None:
        error = classify_provider_error(Exception("429 RESOURCE_EXHAUSTED"), model="test")

        self.assertIsInstance(error, QuotaExceededError)
        self.assertEqual(error.code, "quota_exceeded")
        self.assertTrue(error.retryable)


if __name__ == "__main__":
    unittest.main()
