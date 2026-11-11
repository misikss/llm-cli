import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import dashboard
from models import CallMetrics, LLMResponse, RouteDecision


def fake_result() -> SimpleNamespace:
    return SimpleNamespace(
        response=LLMResponse(
            answer="Archivo recibido",
            confidence=0.9,
            category="general",
            key_points=["La subida llegó al servicio"],
        ),
        route_decision=RouteDecision(
            route="general",
            reason="test",
            needs_memory_rag=False,
        ),
        metrics=CallMetrics(
            timestamp="2026-11-11T00:00:00+00:00",
            model="gemini-test",
            question="analiza el archivo",
            tokens_entrada=10,
            tokens_salida=5,
            tokens_totales=15,
            tarifa_input_por_millon=0.075,
            tarifa_output_por_millon=0.30,
            costo_entrada_usd=0.000001,
            costo_salida_usd=0.000002,
            costo_total_usd=0.000003,
            response={"answer": "Archivo recibido"},
        ),
    )


class DashboardTests(unittest.TestCase):
    def test_xlsx_upload_is_forwarded_with_context_ids(self) -> None:
        with patch("dashboard._run_request", return_value=fake_result()) as run_request:
            response = TestClient(dashboard.app).post(
                "/api/ask",
                data={
                    "question": "analiza el archivo",
                    "user_id": "user-1",
                    "session_id": "session-1",
                    "conversation_id": "conversation-1",
                },
                files={
                    "file": (
                        "ventas.xlsx",
                        b"fake-xlsx-content",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["request"]["file"]["name"], "ventas.xlsx")
        self.assertEqual(payload["request"]["conversation_id"], "conversation-1")
        forwarded_request = run_request.call_args.args[0]
        self.assertEqual(forwarded_request.user_id, "user-1")
        self.assertEqual(forwarded_request.conversation_id, "conversation-1")
        self.assertEqual(forwarded_request.file_path.endswith(".xlsx"), True)

    def test_unsupported_file_type_is_rejected(self) -> None:
        response = TestClient(dashboard.app).post(
            "/api/ask",
            data={"question": "analiza el archivo"},
            files={"file": ("malware.exe", b"not allowed", "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
