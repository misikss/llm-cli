"""Dashboard web local para inspeccionar el pipeline del asistente.

La interfaz es deliberadamente delgada: recibe input y archivos, delega en
AssistantService y devuelve una representación JSON observable del resultado.
La lógica de routing, RAG, generación y memoria sigue viviendo en el servicio.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from config import ERROR_LOG_FILE, GEMINI_API_KEY, LOG_FILE
from display import save_error_to_log, save_to_log
from errors import LLMCLIError, RequestExecutionError
from service import AskRequest, AssistantService


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR / "dashboard"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".mp3",
    ".wav",
    ".ogg",
    ".xlsx",
    ".xls",
    ".csv",
    ".tsv",
    ".txt",
    ".docx",
}

app = FastAPI(
    title="LLM CLI Control Room",
    description="Dashboard local para probar el pipeline multimodal y observar sus métricas.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="dashboard-static")

_service: AssistantService | None = None
_service_lock = threading.Lock()
# El servicio mantiene Working Memory en RAM. Un worker local evita carreras
# durante esta demo; la separación entre chats sigue estando dada por los IDs.
_request_lock = threading.Lock()


def get_service() -> AssistantService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = AssistantService()
    return _service


def _clean_id(value: str | None, fallback: str, label: str) -> str:
    cleaned = (value or fallback).strip()
    if not cleaned or len(cleaned) > 120 or any(char in cleaned for char in "\r\n\t"):
        raise HTTPException(status_code=400, detail=f"{label} no es válido.")
    return cleaned


def _checkbox(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "on", "yes"}


def _read_history(limit: int, user_id: str | None = None, conversation_id: str | None = None) -> list[dict]:
    if not LOG_FILE.exists():
        return []

    entries: list[dict] = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("event_type", "llm_call") != "llm_call":
            continue
        if user_id and entry.get("user_id") != user_id:
            continue
        if conversation_id and entry.get("conversation_id") != conversation_id:
            continue
        entries.append(entry)
    return entries[-max(1, min(limit, 100)):]


def _serialize_result(result, request: AskRequest, file_metadata: dict | None) -> dict:
    return {
        "request": {
            "question": request.question,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
            "use_grounding": request.use_grounding,
            "use_weather": request.use_weather,
            "file": file_metadata,
        },
        "response": result.response.model_dump(),
        "route_decision": result.route_decision.model_dump(),
        "metrics": result.metrics.model_dump(),
    }


def _run_request(request: AskRequest, file_metadata: dict | None):
    with _request_lock:
        result = get_service().ask(request)
        if file_metadata:
            # No exponemos el path temporal del servidor en la UI ni en JSONL.
            result.metrics.file_attached = file_metadata["name"]
        save_to_log(result.metrics)
        return result


@app.get("/", response_class=FileResponse)
async def dashboard_index() -> Path:
    return DASHBOARD_DIR / "index.html"


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "configured": bool(GEMINI_API_KEY),
        "service": "AssistantService",
    }


@app.get("/api/history")
async def history(
    limit: int = 20,
    user_id: str | None = None,
    conversation_id: str | None = None,
) -> dict:
    return {
        "entries": _read_history(
            limit=limit,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    }


@app.post("/api/ask")
async def ask(
    question: str = Form(...),
    user_id: str = Form("local-user"),
    session_id: str = Form("dashboard-session"),
    conversation_id: str = Form("default-conversation"),
    use_grounding: str | None = Form(None),
    use_weather: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> dict:
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    clean_user_id = _clean_id(user_id, "local-user", "user_id")
    clean_session_id = _clean_id(session_id, "dashboard-session", "session_id")
    clean_conversation_id = _clean_id(
        conversation_id,
        "default-conversation",
        "conversation_id",
    )

    temporary_path: str | None = None
    file_metadata: dict | None = None
    try:
        if file and file.filename:
            original_name = Path(file.filename).name
            extension = Path(original_name).suffix.lower()
            if extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Tipo de archivo no soportado: {extension or 'sin extensión'}.",
                )

            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="El archivo supera el límite de 25 MB.",
                )

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=extension,
                prefix="llm_cli_upload_",
                delete=False,
            ) as temporary_file:
                temporary_file.write(content)
                temporary_path = temporary_file.name

            file_metadata = {
                "name": original_name,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "extension": extension,
            }

        request = AskRequest(
            question=question,
            file_path=temporary_path,
            use_grounding=_checkbox(use_grounding),
            use_weather=_checkbox(use_weather),
            user_id=clean_user_id,
            session_id=clean_session_id,
            conversation_id=clean_conversation_id,
        )

        try:
            result = await run_in_threadpool(_run_request, request, file_metadata)
        except LLMCLIError as exc:
            save_error_to_log(exc)
            status_code = 500 if isinstance(exc, RequestExecutionError) else 400
            raise HTTPException(status_code=status_code, detail=exc.message) from exc
        except Exception as exc:
            logger.exception("dashboard_request_failed")
            save_error_to_log(exc)
            raise HTTPException(
                status_code=500,
                detail="No se pudo completar la solicitud desde el dashboard.",
            ) from exc

        return _serialize_result(result, request, file_metadata)
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard:app", host="127.0.0.1", port=8000, reload=False)
