"""Estrategia de análisis documental directo o mediante RAG persistente.

Los archivos pequeños se mantienen en el flujo multimodal de Gemini. Los
archivos extraíbles que superan el umbral, o que ya se consultaron varias
veces, conservan su estructura, se indexan en ChromaDB y se recuperan con
BM25, búsqueda vectorial, RRF y reranking LLM condicional.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Literal

import chromadb
from chromadb.config import Settings
from google import genai

from config import (
    DOCUMENT_BM25_TOP_K,
    DOCUMENT_EMBED_MAX_CHUNKS_BATCH,
    DOCUMENT_EMBED_MAX_TOKENS_BATCH,
    DOCUMENT_EMBED_MAX_TOKENS_PER_TEXT,
    DOCUMENT_EMBED_BACKOFF_SECONDS,
    DOCUMENT_EMBED_MAX_BACKOFF_SECONDS,
    DOCUMENT_EMBED_RETRIES,
    DOCUMENT_HYBRID_TOP_K,
    DOCUMENT_DIRECT_TOKEN_LIMIT,
    DOCUMENT_INDEX_WORKERS,
    DOCUMENT_RERANK_MIN_SCORE,
    DOCUMENT_RERANK_TOP_N,
    DOCUMENT_VECTOR_TOP_K,
    LOG_DIR,
)
from errors import RetrievalError
from observability import capture_usage
from rag import (
    BM25Index,
    RetrievalDocument,
    hybrid_search,
    rerank_retrieval_documents,
)


logger = logging.getLogger(__name__)

DIRECT_MAX_BYTES = 2 * 1024 * 1024
REUSE_THRESHOLD = 3
CHUNK_SIZE = 1_800
CHUNK_OVERLAP = 250
DOCUMENT_INDEX_VERSION = 2
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
    estimated_text_tokens: int = 0


@dataclass(frozen=True)
class DocumentSection:
    """Sección extraída conservando su ubicación lógica en el archivo."""

    text: str
    locator: str


@dataclass(frozen=True)
class DocumentChunk:
    """Chunk recuperable con ID y procedencia estable."""

    id: str
    text: str
    locator: str


@dataclass(frozen=True)
class DocumentRetrieval:
    """Contexto recuperado y telemetría del pipeline documental."""

    context: str
    chunks_indexed: int
    chunks_retrieved: int
    retrieval_mode: Literal["hybrid_rrf", "hybrid_rrf_reranked"]
    reranker_used: bool
    vector_candidates: int
    bm25_candidates: int
    source_ids: list[str]


@dataclass(frozen=True)
class DocumentIndexStatus:
    """Estado observable de una indexación documental en segundo plano."""

    status: Literal["not_indexed", "queued", "running", "ready", "failed"]
    document_id: str
    error: str | None = None


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


def estimate_document_tokens(text: str) -> int:
    """Estima tokens con margen preventivo antes de usar multimodal directo."""
    if not text:
        return 0
    return math.ceil(len(text) / 3.5)


def _is_embedding_quota_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return "RESOURCE_EXHAUSTED" in message or "429" in message or "QUOTA" in message


def _embedding_retry_delay(exc: Exception, attempt: int) -> float:
    match = re.search(r"retry in\s+([\d.]+)s", str(exc).lower())
    if match:
        return min(float(match.group(1)), DOCUMENT_EMBED_MAX_BACKOFF_SECONDS)
    return min(
        DOCUMENT_EMBED_BACKOFF_SECONDS * (2 ** attempt),
        DOCUMENT_EMBED_MAX_BACKOFF_SECONDS,
    )


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

    estimated_text_tokens = 0
    try:
        sections = extract_document_sections(path)
        estimated_text_tokens = estimate_document_tokens(
            "\n\n".join(section.text for section in sections)
        )
    except RetrievalError as exc:
        logger.warning(
            "document_token_estimate_failed error_type=%s document_id=%s",
            type(exc).__name__,
            doc_id,
        )

    if estimated_text_tokens > DOCUMENT_DIRECT_TOKEN_LIMIT:
        reason = (
            f"El texto extraído se estima en {estimated_text_tokens:,} tokens; "
            f"se evita multimodal directo y se fuerza Document RAG."
        )
        mode = "document_rag"
    elif size_bytes > DIRECT_MAX_BYTES:
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
        estimated_text_tokens=estimated_text_tokens,
    )


def _rows_to_text(rows: list[list[object]], title: str = "") -> str:
    lines = [f"[{title}]" if title else ""]
    for row in rows:
        values = ["" if value is None else str(value) for value in row]
        if any(values):
            lines.append(" | ".join(values))
    return "\n".join(line for line in lines if line).strip()


def _rows_to_sections(
    rows: list[list[object]],
    *,
    title: str,
    rows_per_section: int = 40,
) -> list[DocumentSection]:
    if not rows:
        return []
    sections = []
    for start in range(0, len(rows), rows_per_section):
        end = min(start + rows_per_section, len(rows))
        locator = f"{title} · filas {start + 1}-{end}"
        text = _rows_to_text(rows[start:end], title=locator)
        if text:
            sections.append(DocumentSection(text=text, locator=locator))
    return sections


def extract_document_sections(file_path: str | Path) -> list[DocumentSection]:
    """Extrae secciones locales conservando página, hoja, tabla o filas."""
    path = Path(file_path)
    extension = path.suffix.lower()

    try:
        if extension == ".txt":
            paragraphs = [part.strip() for part in path.read_text(
                encoding="utf-8",
                errors="replace",
            ).split("\n\n") if part.strip()]
            return [
                DocumentSection(text=paragraph, locator=f"{path.name} · párrafo {index}")
                for index, paragraph in enumerate(paragraphs, start=1)
            ]

        if extension in {".csv", ".tsv"}:
            delimiter = "," if extension == ".csv" else "\t"
            with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
                rows = list(csv.reader(stream, delimiter=delimiter))
            return _rows_to_sections(rows, title=path.name)

        if extension in {".xlsx", ".xls"}:
            if extension == ".xls":
                raise RetrievalError(
                    "La extracción local de .xls requiere convertirlo a .xlsx o .csv."
                )
            from openpyxl import load_workbook

            workbook = load_workbook(path, read_only=True, data_only=True)
            try:
                sections = []
                for sheet in workbook.worksheets:
                    rows = list(sheet.iter_rows(values_only=True))
                    sections.extend(_rows_to_sections(rows, title=f"Hoja {sheet.title}"))
                return sections
            finally:
                workbook.close()

        if extension == ".pdf":
            from pypdf import PdfReader

            sections = []
            for index, page in enumerate(PdfReader(str(path)).pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    sections.append(DocumentSection(text=text, locator=f"Página {index}"))
            return sections

        if extension == ".docx":
            from docx import Document

            document = Document(str(path))
            sections = []
            paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
            for start in range(0, len(paragraphs), 8):
                end = min(start + 8, len(paragraphs))
                sections.append(DocumentSection(
                    text="\n".join(paragraphs[start:end]),
                    locator=f"Párrafos {start + 1}-{end}",
                ))
            for index, table in enumerate(document.tables, start=1):
                rows = [[cell.text for cell in row.cells] for row in table.rows]
                text = _rows_to_text(rows, f"Tabla {index}")
                if text:
                    sections.append(DocumentSection(text=text, locator=f"Tabla {index}"))
            return sections
    except RetrievalError:
        raise
    except (OSError, ImportError, ValueError) as exc:
        raise RetrievalError(
            f"No se pudo extraer texto de {path.name}: {type(exc).__name__}."
        ) from exc

    raise RetrievalError(f"El formato {extension or 'desconocido'} no admite extracción RAG local.")


def extract_document_text(file_path: str | Path) -> str:
    """Compatibilidad: devuelve el texto unido de las secciones extraídas."""
    return "\n\n".join(section.text for section in extract_document_sections(file_path)).strip()


def chunk_text(text: str, *, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fallback de ventana fija con solapamiento para texto sin estructura."""
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


def chunk_document_sections(
    sections: list[DocumentSection],
    *,
    doc_id: str,
) -> list[DocumentChunk]:
    """Fragmenta secciones y conserva un ID estable de procedencia."""
    chunks: list[DocumentChunk] = []
    max_chars = int(DOCUMENT_EMBED_MAX_TOKENS_PER_TEXT * 3.5)
    for section in sections:
        section_chunks = chunk_text(
            section.text,
            size=min(CHUNK_SIZE, max_chars),
        )
        for fragment_index, text in enumerate(section_chunks, start=1):
            locator = section.locator
            if len(section_chunks) > 1:
                locator = f"{locator} · fragmento {fragment_index}"
            chunk_index = len(chunks)
            chunks.append(DocumentChunk(
                id=f"document_chunk:{doc_id}:{chunk_index}",
                text=text,
                locator=locator,
            ))
    return chunks


def iter_embedding_batches(chunks: list[DocumentChunk]):
    """Divide chunks por presupuesto de tokens y cantidad máxima por request."""
    batch: list[DocumentChunk] = []
    token_budget = 0
    for chunk in chunks:
        chunk_tokens = estimate_document_tokens(chunk.text)
        if chunk_tokens > DOCUMENT_EMBED_MAX_TOKENS_PER_TEXT:
            raise RetrievalError(
                "Un chunk documental supera el límite preventivo de tokens por texto."
            )
        reaches_token_limit = (
            batch
            and token_budget + chunk_tokens > DOCUMENT_EMBED_MAX_TOKENS_BATCH
        )
        reaches_chunk_limit = len(batch) >= DOCUMENT_EMBED_MAX_CHUNKS_BATCH
        if reaches_token_limit or reaches_chunk_limit:
            yield batch
            batch = []
            token_budget = 0
        batch.append(chunk)
        token_budget += chunk_tokens
    if batch:
        yield batch


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
        self._bm25_indexes: dict[str, tuple[int, BM25Index]] = {}
        self._index_executor = ThreadPoolExecutor(
            max_workers=DOCUMENT_INDEX_WORKERS,
            thread_name_prefix="document-indexer",
        )
        self._index_jobs: dict[str, Future] = {}
        self._index_status: dict[str, DocumentIndexStatus] = {}
        self._index_jobs_lock = threading.Lock()

    def _scope_where(self, *, user_id: str, conversation_id: str, doc_id: str) -> dict:
        return {
            "$and": [
                {"user_id": user_id},
                {"conversation_id": conversation_id},
                {"document_id": doc_id},
                {"index_version": DOCUMENT_INDEX_VERSION},
            ]
        }

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _scope_key(self, *, user_id: str, conversation_id: str, doc_id: str) -> str:
        return f"{user_id}:{conversation_id}:{doc_id}"

    def _cached_source_path(self, file_path: str | Path, doc_id: str) -> Path:
        """Conserva una copia local para que el worker sobreviva al upload HTTP."""
        source = Path(file_path)
        cache_dir = LOG_DIR / "document_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{doc_id}{source.suffix.lower()}"
        if not cached.exists():
            temporary = cached.with_name(f".{cached.name}.tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(cached)
        return cached

    def _scoped_chunk_count(self, where: dict) -> int:
        data = self.collection.get(where=where)
        return len(data.get("ids", []) or [])

    def get_index_status(
        self,
        *,
        user_id: str,
        conversation_id: str,
        doc_id: str,
    ) -> DocumentIndexStatus:
        """Consulta el estado sin bloquear esperando al worker."""
        key = self._scope_key(
            user_id=user_id,
            conversation_id=conversation_id,
            doc_id=doc_id,
        )
        where = self._scope_where(
            user_id=user_id,
            conversation_id=conversation_id,
            doc_id=doc_id,
        )
        try:
            data = self.collection.get(where=where, include=["metadatas"])
            ids = data.get("ids", []) or []
            metadatas = data.get("metadatas", []) or []
            totals = {
                int(metadata.get("chunk_total"))
                for metadata in metadatas
                if metadata and metadata.get("chunk_total") is not None
            }
            # Índices anteriores se escribían de una sola vez y no tienen
            # chunk_total; los nuevos pueden quedar parciales si la cuota corta
            # un lote de embeddings.
            complete = bool(ids) and (not totals or len(ids) >= max(totals))
            if complete:
                status = DocumentIndexStatus(status="ready", document_id=doc_id)
                with self._index_jobs_lock:
                    self._index_status[key] = status
                return status
        except Exception as exc:
            logger.warning(
                "document_index_status_failed error_type=%s document_id=%s",
                type(exc).__name__,
                doc_id,
            )

        with self._index_jobs_lock:
            return self._index_status.get(
                key,
                DocumentIndexStatus(status="not_indexed", document_id=doc_id),
            )

    def queue_indexing(
        self,
        client: genai.Client,
        file_path: str | Path,
        *,
        plan: DocumentPlan,
        user_id: str,
        conversation_id: str,
    ) -> DocumentIndexStatus:
        """Copia y encola la indexación sin bloquear la respuesta del chat."""
        key = self._scope_key(
            user_id=user_id,
            conversation_id=conversation_id,
            doc_id=plan.document_id,
        )
        current = self.get_index_status(
            user_id=user_id,
            conversation_id=conversation_id,
            doc_id=plan.document_id,
        )
        if current.status == "ready":
            return current

        cached_path = self._cached_source_path(file_path, plan.document_id)
        source_name = Path(file_path).name
        with self._index_jobs_lock:
            future = self._index_jobs.get(key)
            if future is not None and not future.done():
                return self._index_status.get(
                    key,
                    DocumentIndexStatus(status="queued", document_id=plan.document_id),
                )

            queued = DocumentIndexStatus(status="queued", document_id=plan.document_id)
            self._index_status[key] = queued
            self._index_jobs[key] = self._index_executor.submit(
                self._run_index_job,
                client,
                cached_path,
                plan,
                user_id,
                conversation_id,
                source_name,
                key,
            )
            return queued

    def _run_index_job(
        self,
        client: genai.Client,
        file_path: Path,
        plan: DocumentPlan,
        user_id: str,
        conversation_id: str,
        source_name: str,
        key: str,
    ) -> None:
        with self._index_jobs_lock:
            self._index_status[key] = DocumentIndexStatus(
                status="running",
                document_id=plan.document_id,
            )

        try:
            usage_sink: dict = {}
            lock = self._lock_for(key)
            with lock:
                chunks_indexed, chunks_added = self._ensure_indexed(
                    client,
                    file_path,
                    plan=plan,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    usage_sink=usage_sink,
                    source_name=source_name,
                )
            with self._index_jobs_lock:
                self._index_status[key] = DocumentIndexStatus(
                    status="ready",
                    document_id=plan.document_id,
                )
            logger.info(
                "document_index_ready document_id=%s chunks=%s added=%s embedding_calls=%s",
                plan.document_id,
                chunks_indexed,
                chunks_added,
                usage_sink.get("embedding_calls", 0),
            )
        except Exception as exc:
            status = DocumentIndexStatus(
                status="failed",
                document_id=plan.document_id,
                error=type(exc).__name__,
            )
            with self._index_jobs_lock:
                self._index_status[key] = status
            logger.warning(
                "document_index_failed document_id=%s error_type=%s",
                plan.document_id,
                type(exc).__name__,
            )

    def _get_scoped_documents(self, where: dict) -> list[RetrievalDocument]:
        data = self.collection.get(where=where)
        ids = data.get("ids", []) or []
        documents = data.get("documents", []) or []
        return [
            RetrievalDocument(id=str(chunk_id), text=text)
            for chunk_id, text in zip(ids, documents)
            if text
        ]

    def _get_bm25_index(self, *, where: dict, scope_key: str) -> BM25Index:
        documents = self._get_scoped_documents(where)
        cached = self._bm25_indexes.get(scope_key)
        if cached is None or cached[0] != len(documents):
            index = BM25Index(documents)
            self._bm25_indexes[scope_key] = (len(documents), index)
            return index
        return cached[1]

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

    def _embed_batch(
        self,
        client: genai.Client,
        texts: list[str],
        *,
        usage_sink: dict,
    ) -> list[list[float]]:
        """Solicita embeddings en lote y conserva fallback por fragmento."""
        batch_tokens = sum(estimate_document_tokens(text) for text in texts)
        if (
            len(texts) > DOCUMENT_EMBED_MAX_CHUNKS_BATCH
            or batch_tokens > DOCUMENT_EMBED_MAX_TOKENS_BATCH
            or any(
                estimate_document_tokens(text) > DOCUMENT_EMBED_MAX_TOKENS_PER_TEXT
                for text in texts
            )
        ):
            raise RetrievalError("El lote de embeddings supera el presupuesto preventivo.")
        for attempt in range(DOCUMENT_EMBED_RETRIES + 1):
            try:
                response = client.models.embed_content(
                    model=self.embedding_model,
                    contents=texts,
                )
                embeddings = getattr(response, "embeddings", None) or []
                if len(embeddings) != len(texts):
                    raise RetrievalError("El proveedor devolvió un lote de embeddings incompleto.")
                capture_usage(
                    usage_sink,
                    response,
                    model=self.embedding_model,
                    calls_key="calls",
                    prefix="embedding",
                )
                return [embedding.values for embedding in embeddings]
            except Exception as exc:
                if _is_embedding_quota_error(exc):
                    if attempt >= DOCUMENT_EMBED_RETRIES:
                        raise
                    delay = _embedding_retry_delay(exc, attempt)
                    logger.warning(
                        "document_embedding_quota_backoff attempt=%s delay_seconds=%.1f batch_size=%s",
                        attempt + 1,
                        delay,
                        len(texts),
                    )
                    time.sleep(delay)
                    continue
                logger.warning(
                    "document_embedding_batch_fallback error_type=%s batch_size=%s",
                    type(exc).__name__,
                    len(texts),
                )
                return [self._embed(client, text, usage_sink=usage_sink) for text in texts]
        raise RetrievalError("No se pudo obtener el lote de embeddings.")

    def _ensure_indexed(
        self,
        client: genai.Client,
        file_path: str | Path,
        *,
        plan: DocumentPlan,
        user_id: str,
        conversation_id: str,
        usage_sink: dict,
        source_name: str | None = None,
    ) -> tuple[int, int]:
        where = self._scope_where(
            user_id=user_id,
            conversation_id=conversation_id,
            doc_id=plan.document_id,
        )
        existing = self.collection.get(where=where)
        existing_ids = set(existing.get("ids", []) or [])

        sections = extract_document_sections(file_path)
        chunks = chunk_document_sections(sections, doc_id=plan.document_id)
        if not chunks:
            raise RetrievalError("El documento no contiene texto extraíble para construir RAG.")

        scope_id = hashlib.sha256(
            f"{user_id}:{conversation_id}:{plan.document_id}".encode("utf-8")
        ).hexdigest()[:16]
        ids = [f"doc_{scope_id}_{index}" for index in range(len(chunks))]
        chunks_added = 0
        embedding_batches = 0
        for batch in iter_embedding_batches(chunks):
            pending = [
                (index, chunk, ids[index])
                for index, chunk in enumerate(chunks)
                if ids[index] not in existing_ids and chunk in batch
            ]
            if not pending:
                continue
            embedding_batches += 1
            batch_embeddings = self._embed_batch(
                client,
                [chunk.text for _, chunk, _ in pending],
                usage_sink=usage_sink,
            )
            self.collection.add(
                ids=[chunk_id for _, _, chunk_id in pending],
                embeddings=batch_embeddings,
                documents=[chunk.text for _, chunk, _ in pending],
                metadatas=[
                    {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "document_id": plan.document_id,
                        "chunk_index": index,
                        "chunk_total": len(chunks),
                        "source_name": source_name or Path(file_path).name,
                        "source_id": chunk.id,
                        "locator": chunk.locator,
                        "index_version": DOCUMENT_INDEX_VERSION,
                    }
                    for index, chunk, _ in pending
                ],
            )
            existing_ids.update(chunk_id for _, _, chunk_id in pending)
            chunks_added += len(pending)
        usage_sink["embedding_batches"] = embedding_batches
        return len(existing_ids), chunks_added

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
        n_results: int = DOCUMENT_HYBRID_TOP_K,
        index_if_missing: bool = True,
    ) -> DocumentRetrieval:
        """Indexa si hace falta y ejecuta retrieval híbrido documental."""
        sink = usage_sink if usage_sink is not None else {}
        scope_key = self._scope_key(
            user_id=user_id,
            conversation_id=conversation_id,
            doc_id=plan.document_id,
        )
        where = self._scope_where(
            user_id=user_id,
            conversation_id=conversation_id,
            doc_id=plan.document_id,
        )
        lock = self._lock_for(scope_key)
        with lock:
            if index_if_missing:
                chunks_indexed, _chunks_added = self._ensure_indexed(
                    client,
                    file_path,
                    plan=plan,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    usage_sink=sink,
                )
            else:
                chunks_indexed = self._scoped_chunk_count(where)
                if not chunks_indexed:
                    raise RetrievalError("El índice documental todavía no está listo.")

        query_vector = self._embed(client, query, usage_sink=sink)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(DOCUMENT_VECTOR_TOP_K, chunks_indexed),
            where=where,
        )
        vector_documents = [
            RetrievalDocument(id=str(chunk_id), text=text)
            for chunk_id, text in zip(
                (results.get("ids") or [[]])[0],
                (results.get("documents") or [[]])[0],
            )
            if text
        ]
        if not vector_documents:
            raise RetrievalError("El retrieval documental no devolvió fragmentos relevantes.")

        corpus_documents = self._get_scoped_documents(where)
        bm25_index = self._get_bm25_index(where=where, scope_key=scope_key)
        vector_ids = [document.id for document in vector_documents]
        bm25_ids = bm25_index.search(query, top_k=DOCUMENT_BM25_TOP_K)
        hybrid_documents = hybrid_search(
            query_text=query,
            keywords=[],
            corpus_docs=corpus_documents,
            vector_results=vector_documents,
            top_k=min(n_results, chunks_indexed),
            bm25_index=bm25_index,
        )
        if not hybrid_documents:
            raise RetrievalError("El retrieval híbrido no devolvió candidatos documentales.")

        # Si BM25 y el vectorial discrepan en los primeros resultados, la
        # consulta es ambigua y merece una llamada adicional de reranking.
        top_vector = set(vector_ids[:3])
        top_bm25 = set(bm25_ids[:3])
        ranking_disagreement = bool(
            top_bm25
            and top_vector
            and (
                vector_ids[0] != bm25_ids[0]
                or len(top_vector & top_bm25) < min(len(top_vector), len(top_bm25))
            )
        )
        reranker_used = False
        reranked_documents = hybrid_documents
        if ranking_disagreement and len(hybrid_documents) > 1:
            rerank_usage: dict = {}
            reranked_documents, rerank_in, rerank_out = rerank_retrieval_documents(
                client=client,
                query=query,
                documents=hybrid_documents,
                top_n=DOCUMENT_RERANK_TOP_N,
                min_score=DOCUMENT_RERANK_MIN_SCORE,
                usage_sink=rerank_usage,
            )
            sink["rerank_input_tokens"] = sink.get("rerank_input_tokens", 0) + rerank_in
            sink["rerank_output_tokens"] = sink.get("rerank_output_tokens", 0) + rerank_out
            sink["rerank_usage_available"] = rerank_usage.get("usage_available", True)
            sink["reranker_calls"] = sink.get("reranker_calls", 0) + 1
            reranker_used = True
            if not reranked_documents:
                reranked_documents = hybrid_documents[:DOCUMENT_RERANK_TOP_N]

        metadata = self.collection.get(
            ids=[document.id for document in reranked_documents],
            include=["metadatas"],
        ).get("metadatas", []) or []
        metadata_by_id = {
            document_id: dict(item or {})
            for document_id, item in zip(
                [document.id for document in reranked_documents],
                metadata,
            )
        }
        context_parts = []
        source_ids = []
        for document in reranked_documents:
            item = metadata_by_id.get(document.id, {})
            source_id = str(item.get("source_id") or document.id)
            locator = str(item.get("locator") or Path(file_path).name)
            source_ids.append(source_id)
            context_parts.append(
                f"[source_id={source_id} · {locator}]\n{document.text}"
            )

        context = "\n\n".join(
            context_parts
        )
        return DocumentRetrieval(
            context=context,
            chunks_indexed=chunks_indexed,
            chunks_retrieved=len(reranked_documents),
            retrieval_mode=(
                "hybrid_rrf_reranked" if reranker_used else "hybrid_rrf"
            ),
            reranker_used=reranker_used,
            vector_candidates=len(vector_documents),
            bm25_candidates=len(bm25_ids),
            source_ids=source_ids,
        )
