"""
memory.py — Sistema de memoria híbrida:
  1. Working Memory: Memoria a corto plazo en RAM (sujeto activo y último contexto inmediato).
  2. Episodic Memory: Memoria a largo plazo persistente en ChromaDB con embeddings (gemini-embedding-2).
"""

import logging
import time
import uuid
from pathlib import Path
from pydantic import BaseModel, Field

# Silenciar telemetría anónima de ChromaDB (evita avisos internos de PostHog en consola)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types as genai_types

from config import GEMINI_API_KEY, GEMINI_MODEL
from rag import BM25Index, RetrievalDocument


class MemoryState(BaseModel):
    """Estado compacto de la memoria de la sesión (Working Memory)."""
    active_subject: str = Field(
        default="",
        description="Sujeto, entidad o tema principal activo del que se habla actualmente (ej. 'Lionel Messi', 'FastAPI', 'Clima en Tokio')."
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Lista de hechos o datos clave aprendidos en el turno (máximo 4 bullets breves)."
    )


class EpisodicMemory:
    """
    Memoria episódica a largo plazo usando ChromaDB persistente en disco.
    Guarda recuerdos importantes y recupera solo lo relevante mediante similitud semántica.
    """

    def __init__(self, db_path: str = "chroma_db", collection_name: str = "episodic_memory"):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = "gemini-embedding-2"
        self._bm25_index: BM25Index | None = None
        self._bm25_doc_count = -1

    def _get_embedding(self, genai_client: genai.Client, text: str) -> list[float]:
        """Calcula el vector de embedding usando gemini-embedding-2."""
        res = genai_client.models.embed_content(
            model=self.embedding_model,
            contents=text,
        )
        return res.embeddings[0].values

    def store_memory(
        self,
        genai_client: genai.Client,
        question: str,
        answer: str,
        subject: str = "",
        category: str = "general"
    ) -> None:
        """Guarda un recuerdo sintetizado de la interacción en ChromaDB."""
        # Filtramos saludos o textos vacíos
        if len(question.strip()) < 5:
            return

        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        memory_text = f"[{date_str}] Tema: {subject or 'General'}. Pregunta: {question}. Respuesta: {answer[:300]}"
        try:
            vector = self._get_embedding(genai_client, memory_text)
            memory_id = f"mem_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            self.collection.add(
                ids=[memory_id],
                embeddings=[vector],
                documents=[memory_text],
                metadatas=[{
                    "subject": subject or "general",
                    "category": category,
                    "date": date_str,
                    "timestamp": str(int(time.time()))
                }]
            )
            self._bm25_index = None
            self._bm25_doc_count = -1
        except Exception:
            # Si falla la indexación en segundo plano, la conversación no se bloquea
            pass

    def get_all_documents(self) -> list[RetrievalDocument]:
        """Obtiene documentos e IDs estables para indexar en BM25."""
        try:
            data = self.collection.get()
            ids = data.get("ids", []) or []
            documents = data.get("documents", []) or []
            return [
                RetrievalDocument(id=str(memory_id), text=document)
                for memory_id, document in zip(ids, documents)
                if document
            ]
        except Exception:
            return []

    def get_bm25_index(self) -> BM25Index:
        """Devuelve un índice BM25 cacheado e invalida el caché al crecer ChromaDB."""
        try:
            collection_count = self.collection.count()
        except Exception:
            collection_count = 0

        if self._bm25_index is None or collection_count != self._bm25_doc_count:
            documents = self.get_all_documents()
            self._bm25_index = BM25Index(documents)
            self._bm25_doc_count = collection_count

        return self._bm25_index

    def retrieve_vector_candidates(
        self,
        genai_client: genai.Client,
        query: str,
        n_results: int = 5
    ) -> list[str]:
        """Recupera candidatos vectoriales para alimentar la búsqueda híbrida."""
        try:
            count = self.collection.count()
            if count == 0:
                return []

            query_vector = self._get_embedding(genai_client, query)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(n_results, count)
            )
            if results and results.get("documents") and results.get("ids"):
                documents = results["documents"][0]
                ids = results["ids"][0]
                return [
                    RetrievalDocument(id=str(memory_id), text=document)
                    for memory_id, document in zip(ids, documents)
                    if document
                ]
            return []
        except Exception:
            return []

    def retrieve_relevant_memories(
        self,
        genai_client: genai.Client,
        query: str,
        similarity_threshold: float = 0.40,
        n_results: int = 2
    ) -> list[str]:
        """
        Recupera recuerdos del pasado SOLO si superan el umbral de similitud (distancia coseno < threshold).
        Si la distancia es alta (tema diferente), devuelve lista vacía para no gastar tokens.
        """
        try:
            count = self.collection.count()
            if count == 0:
                return []

            query_vector = self._get_embedding(genai_client, query)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(n_results, count)
            )

            relevant_docs = []
            if results and results.get("documents") and results.get("distances"):
                docs = results["documents"][0]
                distances = results["distances"][0]
                for doc, dist in zip(docs, distances):
                    # En distancia coseno de ChromaDB: 0.0 es idéntico, 1.0 es ortogonal.
                    # Menor a 0.40 indica fuerte relevancia conceptual.
                    if dist <= similarity_threshold:
                        relevant_docs.append(doc)

            return relevant_docs
        except Exception:
            return []


class HybridMemory:
    """
    Coordinador de Memoria Híbrida:
      - Working Memory: Para saber de qué se habla en el turno actual (ej. 'él' -> Messi).
      - Episodic Memory: Para recordar conversaciones pasadas en ChromaDB si el tema vuelve a surgir.
    """

    def __init__(self, db_path: str = "chroma_db"):
        self.active_subject: str = ""
        self.key_facts: list[str] = []
        self.episodic = EpisodicMemory(db_path=db_path)

    def get_working_memory_context(self) -> str:
        """Devuelve el contexto inmediato en RAM (sujeto y hechos activos) a costo 0 de tokens."""
        parts = []
        if self.active_subject:
            parts.append(f"[Tema actual en conversación]: {self.active_subject}")
        if self.key_facts:
            parts.append("[Datos inmediatos]: " + "; ".join(self.key_facts))
        return "\n".join(parts) + "\n\n" if parts else ""

    def get_context_prompt(
        self,
        genai_client: genai.Client,
        query: str,
        keywords: list[str] | None = None
    ) -> tuple[str, int, int]:
        """
        Construye el contexto usando Hybrid Search (Vectorial + BM25) y LLM Reranking.
        Retorna: (bloque_contexto, tokens_rerank_in, tokens_rerank_out).
        """
        context_parts = []
        tokens_rerank_in = 0
        tokens_rerank_out = 0

        # 1. Working Memory (inmediata en RAM a costo 0)
        wm = self.get_working_memory_context()
        if wm:
            context_parts.append(wm.strip())

        # 2. Hybrid Search (ChromaDB + BM25 cacheado)
        bm25_index = self.episodic.get_bm25_index()
        vector_candidates = self.episodic.retrieve_vector_candidates(genai_client, query, n_results=5)

        from rag import hybrid_search, rerank_documents
        from config import RERANK_MIN_SCORE, RERANK_TOP_N

        hybrid_candidates = hybrid_search(
            query_text=query,
            keywords=keywords or [],
            corpus_docs=bm25_index.documents,
            vector_results=vector_candidates,
            top_k=4,
            bm25_index=bm25_index,
        )

        # 3. LLM Reranking
        if hybrid_candidates:
            candidate_texts = [document.text for document in hybrid_candidates]
            top_memories, tokens_rerank_in, tokens_rerank_out = rerank_documents(
                client=genai_client,
                query=query,
                documents=candidate_texts,
                top_n=RERANK_TOP_N,
                min_score=RERANK_MIN_SCORE,
            )
            if top_memories:
                recalled_text = " | ".join(top_memories)
                context_parts.append(f"[Recuerdos filtrados y re-rankeados de ChromaDB]: {recalled_text}")

        if not context_parts:
            return "", tokens_rerank_in, tokens_rerank_out

        return "\n".join(context_parts) + "\n\n", tokens_rerank_in, tokens_rerank_out

    def update(
        self,
        genai_client: genai.Client,
        user_message: str,
        model_answer: str,
        category: str = "general"
    ) -> None:
        """
        Actualiza tanto la Working Memory como la Memoria Episódica en ChromaDB.
        """
        if len(user_message.strip()) <= 3:
            return

        # 1. Actualizar Working Memory (extrayendo sujeto)
        update_prompt = (
            f"Contexto previo: Sujeto actual: '{self.active_subject}'\n\n"
            f"Interacción:\n"
            f"Usuario: {user_message}\n"
            f"Asistente: {model_answer[:250]}\n\n"
            f"Determina cuál es el sujeto/tema principal activo y hasta 3 hechos clave."
        )

        detected_subject = self.active_subject
        try:
            from client import call_with_fallback
            config = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MemoryState,
                temperature=0.0,
            )
            res, _ = call_with_fallback(genai_client, update_prompt, config)
            if res.parsed:
                st: MemoryState = res.parsed
                if st.active_subject:
                    self.active_subject = st.active_subject
                    detected_subject = st.active_subject
                self.key_facts = st.key_facts[:3]
        except Exception:
            # Fallback simple
            words = [w for w in user_message.split() if len(w) > 3 and w.istitle()]
            if words:
                self.active_subject = " ".join(words[:2])
                detected_subject = self.active_subject

        # 2. Guardar en Episodic Memory (ChromaDB) SOLO si NO es información efímera/volátil
        # Evitamos ensuciar la base vectorial con clima de hoy o noticias pasajeras
        if category in ("clima", "actualidad_efimera", "volatil"):
            # Información efímera: no se indexa permanentemente
            return

        self.episodic.store_memory(
            genai_client=genai_client,
            question=user_message,
            answer=model_answer,
            subject=detected_subject,
            category=category,
        )

    def reset(self) -> None:
        """Reinicia la memoria de trabajo de la sesión en RAM."""
        self.active_subject = ""
        self.key_facts = []
