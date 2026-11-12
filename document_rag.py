"""Estrategia de análisis documental directo o mediante RAG persistente.

Los archivos pequeños se mantienen en el flujo multimodal de Gemini. Los
archivos extraíbles que superan el umbral, o que ya se consultaron varias
veces, se convierten en texto, se fragmentan y se indexan en una colección
separada de ChromaDB.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import logging
from pathlib import Path
import threading
from typing import Literal

import chromadb
from chromadb.config import Settings
from google import genai

from config import LOG_DIR
from errors import RetrievalError
from observability import capture_usage


logger = logging.getLogger(__name__)

DIRECT_MAX_BYTES = 2 * 1024 * 1024
REUSE_THRESHOLD = 3
CHUNK_SIZE = 1_800
CHUNK_OVERLAP = 250
REGISTRY_PATH = LOG_DIR / "document_registry.json"

EXTRACTABLE_EXTENSIONS = {".txt", ".csv", ".tsv", ".xlsx", ".xls", ".pdf", ".docx"}
MULTIMODAL_ONLY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".ogg"}

_registry_lock = threading.Lock()


@dataclass(frozen=True)
class DocumentPlan:
    """Decisión observable sobre cómo se procesará el archivo adjunto."""

    mode: Literal["multimodal_direct", "document_rag"]
    document_id: str
    reason: str
    file_size_bytes: int
    previous_queries: int


@dataclass(frozen=True)
class DocumentRetrieval:
    """Contexto recuperado y metadatos de indexación para una consulta."""

    context: str
    chunks_indexed: int
    chunks_retrieved: int


def _load_registry() -> dict[str, dict[str, int]]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("document_registry_read_failed error_type=%s", type(exc).__name__)
        return {}
    return data if isinstance(data, dict) else {}


def _record_query(user_id: str, document_id: str) -> int:
    """Registra una consulta por usuario y documento sin guardar el archivo."""
    key = f"{user_id}:{document_id}"
    with _registry_lock:
        registry = _load_registry()
        current = registry.get(key, {})
        previous_queries = int(current.get("query_count", 0))
        registry[key] = {"query_count": previous_queries + 1}
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            REGISTRY_PATH.write_text(
                json.dumps(registry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("document_registry_write_failed error_type=%s", type(exc).__name__)
        return previous_queries


def document_id(file_path: str | Path) -> str:
    """Obtiene un identificador estable basado en el contenido del archivo."""
    path = Path(file_path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:24]


def choose_document_plan(
    file_path: str | Path,
    *,
    user_id: str,
) -> DocumentPlan:
    """Selecciona multimodal directo o RAG documental según tamaño y reutilización."""
    path = Path(file_path)
    if not path.exists():
        raise RetrievalError(f"No se encontró el documento para planificar: {path}")

    size_bytes = path.stat().st_size
    doc_id = document_id(path)
    previous_queries = _record_query(user_id, doc_id)
    extension = path.suffix.lower()

    if extension not in EXTRACTABLE_EXTENSIONS:
        return DocumentPlan(
            mode="multimodal_direct",
            document_id=doc_id,
            reason="El formato requiere análisis multimodal directo.",
            file_size_bytes=size_bytes,
            previous_queries=previous_queries,
        )

    if size_bytes > DIRECT_MAX_BYTES:
        reason = "El documento supera 2 MB y se beneficia de extracción, chunks y retrieval."
        mode = "document_rag"
    elif previous_queries + 1 >= REUSE_THRESHOLD:
        reason = "El mismo documento ya fue consultado varias veces; se reutiliza su índice RAG."
        mode = "document_rag"
    else:
        reason = "Documento pequeño y de primera consulta: análisis multimodal directo."
        mode = "multimodal_direct"

    return DocumentPlan(
        mode=mode,
        document_id=doc_id,
        reason=reason,
        file_size_bytes=size_bytes,
        previous_queries=previous_queries,
    )


def _rows_to_text(rows: list[list[object]], title: str = "") -> str:
    lines = [f"[{title}]" if title else ""]
    for row in rows:
        values = ["" if value is None else str(value) for value in row]
        if any(values):
            lines.append(" | ".join(values))
    return "\n".join(line for line in lines if line).strip()


def extract_document_text(file_path: str | Path) -> str:
    """Extrae texto localmente para los formatos compatibles con document RAG."""
    path = Path(file_path)
    extension = path.suffix.lower()

    try:
        if extension in {".txt", ".csv", ".tsv"}:
            if extension == ".txt":
                return path.read_text(encoding="utf-8", errors="replace").strip()
            delimiter = "," if extension == ".csv" else "\t"
            with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
                rows = list(csv.reader(stream, delimiter=delimiter))
            return _rows_to_text(rows, title=path.name)

        if extension in {".xlsx", ".xls"}:
            if extension == ".xls":
                raise RetrievalError(
                    "La extracción local de .xls requiere convertirlo a .xlsx o .csv."
                )
            from openpyxl import load_workbook

            workbook = load_workbook(path, read_only=True, data_only=True)
            sections = []
            for sheet in workbook.worksheets:
                sections.append(_rows_to_text(list(sheet.iter_rows(values_only=True)), sheet.title))
            return "\n\n".join(section for section in sections if section).strip()

        if extension == ".pdf":
            from pypdf import PdfReader

            pages = []
            for index, page in enumerate(PdfReader(str(path)).pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(f"[Página {index}]\n{text}")
            return "\n\n".join(pages).strip()

        if extension == ".docx":
            from docx import Document

            document = Document(str(path))
            paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
            tables = []
            for index, table in enumerate(document.tables, start=1):
                rows = [[cell.text for cell in row.cells] for row in table.rows]
                tables.append(_rows_to_text(rows, f"Tabla {index}"))
            return "\n\n".join([*filter(None, paragraphs), *filter(None, tables)]).strip()
    except RetrievalError:
        raise
    except (OSError, ImportError, ValueError) as exc:
        raise RetrievalError(
            f"No se pudo extraer texto de {path.name}: {type(exc).__name__}."
        ) from exc

    raise RetrievalError(f"El formato {extension or 'desconocido'} no admite extracción RAG local.")


def chunk_text(text: str, *, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide el texto en fragmentos con solapamiento para preservar contexto."""
    clean = " ".join(text.split())
    if not clean:
        return []
    if overlap >= size:
        raise ValueError("El solapamiento debe ser menor que el tamaño del chunk.")
    step = size - overlap
    chunks = []
    for start in range(0, len(clean), step):
        chunk = clean[start : start + size]
        if start and len(chunk) < overlap:
            break
        chunks.append(chunk)
    return chunks


class DocumentRAG:
    """Índice persistente de chunks documentales aislado por usuario y archivo."""

    def __init__(self, db_path: str = "chroma_db") -> None:
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model = "gemini-embedding-2"
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _scope_where(self, *, user_id: str, conversation_id: str, doc_id: str) -> dict:
        return {
            "$and": [
                {"user_id": user_id},
                {"conversation_id": conversation_id},
                {"document_id": doc_id},
            ]
        }

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _embed(
        self,
        client: genai.Client,
        text: str,
        *,
        usage_sink: dict,
    ) -> list[float]:
        response = client.models.embed_content(model=self.embedding_model, contents=text)
        capture_usage(
            usage_sink,
            response,
            model=self.embedding_model,
            calls_key="calls",
            prefix="embedding",
        )
        if not getattr(response, "embeddings", None):
            raise RetrievalError("El proveedor no devolvió un embedding para el documento.")
        return response.embeddings[0].values

    def _ensure_indexed(
        self,
        client: genai.Client,
        file_path: str | Path,
        *,
        plan: DocumentPlan,
        user_id: str,
        conversation_id: str,
        usage_sink: dict,
    ) -> tuple[int, int]:
        where = self._scope_where(
            user_id=user_id,
            conversation_id=conversation_id,
            doc_id=plan.document_id,
        )
        existing = self.collection.get(where=where)
        existing_ids = existing.get("ids", []) or []
        if existing_ids:
            return len(existing_ids), 0

        text = extract_document_text(file_path)
        chunks = chunk_text(text)
        if not chunks:
            raise RetrievalError("El documento no contiene texto extraíble para construir RAG.")

        scope_id = hashlib.sha256(
            f"{user_id}:{conversation_id}:{plan.document_id}".encode("utf-8")
        ).hexdigest()[:16]
        ids = [f"doc_{scope_id}_{index}" for index in range(len(chunks))]
        embeddings = [self._embed(client, chunk, usage_sink=usage_sink) for chunk in chunks]
        metadata = [
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "document_id": plan.document_id,
                "chunk_index": index,
                "source_name": Path(file_path).name,
            }
            for index in range(len(chunks))
        ]
        self.collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadata)
        return len(chunks), len(chunks)

    def retrieve(
        self,
        client: genai.Client,
        file_path: str | Path,
        *,
        plan: DocumentPlan,
        user_id: str,
        conversation_id: str,
        query: str,
        usage_sink: dict | None = None,
        n_results: int = 6,
    ) -> DocumentRetrieval:
        """Indexa si hace falta y recupera los chunks relevantes para la consulta."""
        sink = usage_sink if usage_sink is not None else {}
        lock = self._lock_for(f"{user_id}:{conversation_id}:{plan.document_id}")
        with lock:
            chunks_indexed, chunks_added = self._ensure_indexed(
                client,
                file_path,
                plan=plan,
                user_id=user_id,
                conversation_id=conversation_id,
                usage_sink=sink,
            )

        query_vector = self._embed(client, query, usage_sink=sink)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(n_results, chunks_indexed),
            where=self._scope_where(
                user_id=user_id,
                conversation_id=conversation_id,
                doc_id=plan.document_id,
            ),
        )
        documents = (results.get("documents") or [[]])[0]
        if not documents:
            raise RetrievalError("El retrieval documental no devolvió fragmentos relevantes.")

        source = Path(file_path).name
        context = "\n\n".join(
            f"[Fragmento documental {index + 1} · {source}]\n{document}"
            for index, document in enumerate(documents)
        )
        return DocumentRetrieval(
            context=context,
            chunks_indexed=chunks_indexed,
            chunks_retrieved=len(documents),
        )
