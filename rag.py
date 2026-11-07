"""
rag.py — Módulo de Advanced RAG:
  1. Query Engineering: Rewriting & Expansión de preguntas ambiguas.
  2. Hybrid Search: Búsqueda Semántica (ChromaDB) + Búsqueda Léxica (BM25) fusionadas con RRF (Reciprocal Rank Fusion).
  3. Re-ranking: Evaluación Cross-Encoder de relevancia antes de entregar el contexto al generador final.
"""

import re
from typing import Literal
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types as genai_types

from config import GEMINI_MODEL, RERANK_MIN_SCORE, RERANK_TOP_N


# ──────────────────────────────────────────────────────────────────────────────
# 1. ESQUEMAS PYDANTIC PARA QUERY ENGINEERING & RERANKING
# ──────────────────────────────────────────────────────────────────────────────

class RewrittenQuery(BaseModel):
    """Resultado del Query Engineering / Reescritura."""
    rewritten_query: str = Field(
        ...,
        description="Pregunta reescrita en forma autónoma y completa, resolviendo pronombres y ambigüedades con el contexto previo."
    )
    search_keywords: list[str] = Field(
        default_factory=list,
        description="Palabras clave exactas para búsqueda léxica (BM25)."
    )


class RerankItem(BaseModel):
    index: int = Field(..., description="Índice original del fragmento evaluado.")
    score: float = Field(..., ge=0.0, le=1.0, description="Puntaje de relevancia de 0.0 (inútil) a 1.0 (responde perfectamente).")
    reason: str = Field(..., description="Justificación breve de la puntuación.")


class RerankResult(BaseModel):
    ranking: list[RerankItem] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# 2. QUERY ENGINEERING (REWRITING & EXPANSION)
# ──────────────────────────────────────────────────────────────────────────────

def rewrite_query(
    client: genai.Client,
    question: str,
    active_subject: str = "",
    recent_context: str = ""
) -> tuple[RewrittenQuery, int, int]:
    """
    Transforma preguntas ambiguas ('¿cuándo llegó él?', '¿cuánto cuesta eso?') en
    preguntas autónomas de alta precisión para búsqueda vectorial y BM25.
    Retorna: (RewrittenQuery, tokens_entrada, tokens_salida).
    """
    from datetime import datetime
    current_date_str = datetime.now().strftime("%Y-%m-%d (%A)")

    prompt = (
        f"Fecha actual del sistema: {current_date_str}\n"
        f"Eres un experto en Query Engineering para sistemas de recuperación de información (RAG).\n"
        f"Contexto previo de la conversación:\n"
        f"- Sujeto/Entidad activa: '{active_subject}'\n"
        f"- Resumen reciente: '{recent_context}'\n\n"
        f"Pregunta original del usuario: '{question}'\n\n"
        f"Tareas:\n"
        f"1. Reescribe la pregunta para que sea totalmente explícita y autónoma, resolviendo pronombres o referencias elípticas con el sujeto activo si corresponde.\n"
        f"2. Extrae de 2 a 5 palabras clave exactas para búsqueda léxica (BM25)."
    )

    try:
        from client import call_with_fallback
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RewrittenQuery,
            temperature=0.0,
        )
        res, _ = call_with_fallback(client, prompt, config)
        uso = res.usage_metadata
        t_in = getattr(uso, "prompt_token_count", 0) or 0
        t_out = getattr(uso, "candidates_token_count", 0) or 0

        if res.parsed:
            return res.parsed, t_in, t_out
    except Exception:
        pass

    # Fallback si falla el llamado
    tokens = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 3]
    return RewrittenQuery(rewritten_query=question, search_keywords=tokens[:3]), 0, 0


# ──────────────────────────────────────────────────────────────────────────────
# 3. HYBRID SEARCH (VECTORIAL + BM25 CON FUSIÓN RRF)
# ──────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_rankings: list[str],
    bm25_rankings: list[str],
    k: int = 60
) -> list[tuple[str, float]]:
    """
    Algoritmo estándar de la industria (RRF - Reciprocal Rank Fusion) para combinar
    búsquedas densas (vectores) y búsquedas dispersas (BM25) de forma equilibrada.
    Score = sum(1 / (k + rank_i))
    """
    scores: dict[str, float] = {}

    for rank, doc in enumerate(vector_rankings):
        scores[doc] = scores.get(doc, 0.0) + (1.0 / (k + rank + 1))

    for rank, doc in enumerate(bm25_rankings):
        scores[doc] = scores.get(doc, 0.0) + (1.0 / (k + rank + 1))

    # Ordenar de mayor a menor puntuación RRF
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs


def hybrid_search(
    query_text: str,
    keywords: list[str],
    corpus_docs: list[str],
    vector_results: list[str],
    top_k: int = 5
) -> list[str]:
    """
    Ejecuta búsqueda híbrida combinando los mejores resultados de ChromaDB con BM25.
    """
    if not corpus_docs:
        return vector_results[:top_k]

    # 1. Búsqueda Léxica BM25
    tokenized_corpus = [re.findall(r"\w+", doc.lower()) for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    # Combinamos la query con las palabras clave extraídas en Query Engineering
    search_terms = " ".join([query_text] + keywords)
    tokenized_query = re.findall(r"\w+", search_terms.lower())

    bm25_scores = bm25.get_scores(tokenized_query)
    # Ordenar documentos según BM25
    bm25_indexed = sorted(range(len(corpus_docs)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_docs = [corpus_docs[i] for i in bm25_indexed if bm25_scores[i] > 0][:top_k * 2]

    # 2. Fusión RRF
    fused = reciprocal_rank_fusion(vector_rankings=vector_results, bm25_rankings=bm25_docs)
    return [doc for doc, _ in fused[:top_k]]


# ──────────────────────────────────────────────────────────────────────────────
# 4. CROSS-ENCODER RERANKING
# ──────────────────────────────────────────────────────────────────────────────

def rerank_documents(
    client: genai.Client,
    query: str,
    documents: list[str],
    top_n: int = RERANK_TOP_N,
    min_score: float = RERANK_MIN_SCORE
) -> tuple[list[str], int, int]:
    """
    Aplica reordenamiento Cross-Encoder sobre los candidatos híbridos.
    Evalúa la relevancia exacta de cada fragmento respecto a la pregunta del usuario.
    Retorna: (documentos_filtrados, tokens_in, tokens_out).
    """
    if not documents:
        return [], 0, 0

    if len(documents) == 1:
        return documents, 0, 0

    prompt = (
        f"Eres un modelo de Re-ranking de alta precisión para RAG.\n"
        f"Evalúa críticamente qué tan útil y relevante es cada fragmento para responder a la pregunta.\n\n"
        f"Pregunta: '{query}'\n\n"
        f"Fragmentos candidatos:\n"
    )
    for idx, doc in enumerate(documents):
        prompt += f"[{idx}] {doc}\n"

    try:
        from client import call_with_fallback
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RerankResult,
            temperature=0.0,
        )
        res, _ = call_with_fallback(client, prompt, config)
        uso = res.usage_metadata
        t_in = getattr(uso, "prompt_token_count", 0) or 0
        t_out = getattr(uso, "candidates_token_count", 0) or 0

        if res.parsed:
            rerank_data: RerankResult = res.parsed
            # Ordenar por score descendente
            ranked = sorted(rerank_data.ranking, key=lambda r: r.score, reverse=True)
            selected = [
                documents[item.index]
                for item in ranked
                if item.index < len(documents) and item.score >= min_score
            ]
            return selected[:top_n], t_in, t_out
    except Exception:
        pass

    # Fallback si falla el reranker: retornar los primeros candidatos
    return documents[:top_n], 0, 0

