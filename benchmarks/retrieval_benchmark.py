"""Benchmark reproducible para comparar las estrategias de retrieval del proyecto.

Modo offline:
    python benchmarks/retrieval_benchmark.py

Benchmark completo con embeddings Gemini y LLM reranking:
    python benchmarks/retrieval_benchmark.py --live --output benchmarks/results.md

Si la cuota generativa está agotada pero quieres medir embeddings y retrieval:
    python benchmarks/retrieval_benchmark.py --live --skip-reranker
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GEMINI_API_KEY, GEMINI_MODEL  # noqa: E402
from rag import (  # noqa: E402
    BM25Index,
    RetrievalDocument,
    hybrid_search,
    rerank_documents,
)


DATASET_PATH = Path(__file__).with_name("retrieval_dataset.json")
EMBEDDING_MODEL = "gemini-embedding-2"
TOP_K = 3


def load_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    """Carga y valida el dataset versionado del benchmark."""
    dataset = json.loads(path.read_text(encoding="utf-8"))
    corpus = dataset.get("corpus", [])
    queries = dataset.get("queries", [])
    if not corpus or not queries:
        raise ValueError("El dataset debe contener corpus y queries no vacíos.")

    corpus_ids = {item["id"] for item in corpus}
    for query in queries:
        unknown_ids = set(query["relevant_ids"]) - corpus_ids
        if unknown_ids:
            raise ValueError(
                f"La query {query['id']} referencia IDs inexistentes: {sorted(unknown_ids)}"
            )
    return dataset


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calcula similitud coseno sin depender de NumPy."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def embed_text(client: Any, text: str) -> list[float]:
    """Genera un embedding Gemini para un texto del benchmark."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return list(response.embeddings[0].values)


def vector_search(
    query_vector: list[float],
    documents: list[RetrievalDocument],
    document_vectors: dict[str, list[float]],
    top_k: int = TOP_K,
) -> list[RetrievalDocument]:
    """Ordena documentos por similitud vectorial."""
    ranked = [
        (
            cosine_similarity(query_vector, document_vectors[document.id]),
            document,
        )
        for document in documents
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in ranked[:top_k]]


def retrieval_metrics(ranked_ids: list[str], relevant_ids: list[str]) -> dict[str, float]:
    """Calcula Hit@1, Hit@3 y MRR@3 para una query."""
    relevant = set(relevant_ids)
    top_k_ids = ranked_ids[:TOP_K]
    first_relevant_rank = next(
        (rank for rank, doc_id in enumerate(top_k_ids, start=1) if doc_id in relevant),
        None,
    )
    return {
        "hit_at_1": float(bool(top_k_ids and top_k_ids[0] in relevant)),
        "hit_at_3": float(bool(set(top_k_ids) & relevant)),
        "mrr_at_3": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
    }


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"hit_at_1": 0.0, "hit_at_3": 0.0, "mrr_at_3": 0.0}
    return {
        name: round(sum(row[name] for row in rows) / len(rows), 3)
        for name in ("hit_at_1", "hit_at_3", "mrr_at_3")
    }


def run_benchmark(
    dataset: dict[str, Any],
    live: bool = False,
    limit: int | None = None,
    include_reranker: bool = True,
) -> dict[str, Any]:
    documents = [
        RetrievalDocument(id=item["id"], text=item["text"])
        for item in dataset["corpus"]
    ]
    queries = dataset["queries"][:limit] if limit else dataset["queries"]
    bm25_index = BM25Index(documents)

    client = None
    document_vectors: dict[str, list[float]] = {}
    embedding_ms = 0.0
    if live:
        if not GEMINI_API_KEY:
            raise RuntimeError("--live requiere GEMINI_API_KEY en el entorno.")
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        start = time.perf_counter()
        for document in documents:
            document_vectors[document.id] = embed_text(client, document.text)
        embedding_ms = (time.perf_counter() - start) * 1000

    methods = ["BM25"]
    if live:
        methods.extend(["Vectorial", "Híbrido RRF", "Híbrido + LLM reranker"])
        if not include_reranker:
            methods.remove("Híbrido + LLM reranker")
    method_rows: dict[str, list[dict[str, float]]] = {method: [] for method in methods}
    latencies: dict[str, list[float]] = {method: [] for method in methods}
    details: list[dict[str, Any]] = []
    rerank_tokens = {"input": 0, "output": 0}

    for query in queries:
        question = query["question"]
        relevant_ids = query["relevant_ids"]
        detail: dict[str, Any] = {
            "id": query["id"],
            "question": question,
            "relevant_ids": relevant_ids,
        }

        start = time.perf_counter()
        bm25_ids = bm25_index.search(question, top_k=TOP_K)
        bm25_ms = (time.perf_counter() - start) * 1000
        detail["BM25"] = bm25_ids
        method_rows["BM25"].append(retrieval_metrics(bm25_ids, relevant_ids))
        latencies["BM25"].append(bm25_ms)

        if not live:
            details.append(detail)
            continue

        assert client is not None
        embedding_start = time.perf_counter()
        query_vector = embed_text(client, question)
        query_embedding_ms = (time.perf_counter() - embedding_start) * 1000

        vector_start = time.perf_counter()
        vector_documents = vector_search(query_vector, documents, document_vectors)
        vector_ms = query_embedding_ms + (time.perf_counter() - vector_start) * 1000
        vector_ids = [document.id for document in vector_documents]
        detail["Vectorial"] = vector_ids
        method_rows["Vectorial"].append(retrieval_metrics(vector_ids, relevant_ids))
        latencies["Vectorial"].append(vector_ms)

        hybrid_start = time.perf_counter()
        hybrid_documents = hybrid_search(
            query_text=question,
            keywords=[],
            corpus_docs=documents,
            vector_results=vector_documents,
            top_k=TOP_K,
            bm25_index=bm25_index,
        )
        hybrid_ms = vector_ms + (time.perf_counter() - hybrid_start) * 1000
        hybrid_ids = [document.id for document in hybrid_documents]
        detail["Híbrido RRF"] = hybrid_ids
        method_rows["Híbrido RRF"].append(retrieval_metrics(hybrid_ids, relevant_ids))
        latencies["Híbrido RRF"].append(hybrid_ms)

        if include_reranker:
            rerank_start = time.perf_counter()
            reranked_texts, tokens_in, tokens_out = rerank_documents(
                client=client,
                query=question,
                documents=[document.text for document in hybrid_documents],
                top_n=TOP_K,
                min_score=0.0,
            )
            text_to_id = {document.text: document.id for document in hybrid_documents}
            reranked_ids = [text_to_id[text] for text in reranked_texts if text in text_to_id]
            rerank_ms = hybrid_ms + (time.perf_counter() - rerank_start) * 1000
            detail["Híbrido + LLM reranker"] = reranked_ids
            method_rows["Híbrido + LLM reranker"].append(
                retrieval_metrics(reranked_ids, relevant_ids)
            )
            latencies["Híbrido + LLM reranker"].append(rerank_ms)
            rerank_tokens["input"] += tokens_in
            rerank_tokens["output"] += tokens_out
        details.append(detail)

    summary = {}
    for method in methods:
        metrics = average_metrics(method_rows[method])
        metrics["avg_latency_ms"] = round(
            sum(latencies[method]) / len(latencies[method]), 2
            if latencies[method]
            else 0.0,
        )
        summary[method] = metrics

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "live" if live else "offline",
        "reranker_enabled": bool(live and include_reranker),
        "model": GEMINI_MODEL if live else None,
        "embedding_model": EMBEDDING_MODEL if live else None,
        "queries": len(queries),
        "corpus_documents": len(documents),
        "corpus_embedding_ms": round(embedding_ms, 2),
        "rerank_tokens": rerank_tokens,
        "summary": summary,
        "details": details,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Retrieval benchmark",
        "",
        f"- Generado: `{result['generated_at']}`",
        f"- Modo: `{result['mode']}`",
        f"- Corpus: `{result['corpus_documents']}` documentos / `{result['queries']}` queries",
    ]
    if result["mode"] == "live":
        lines.extend([
            f"- Embeddings: `{result['embedding_model']}`",
            f"- Modelo del proyecto: `{result['model']}`",
            f"- Tiempo de embeddings del corpus: `{result['corpus_embedding_ms']} ms`",
            f"- Tokens del reranker: `{result['rerank_tokens']['input']}` input / `{result['rerank_tokens']['output']}` output",
        ])
        if not result["reranker_enabled"]:
            lines.append("- Nota: el LLM reranker fue omitido con `--skip-reranker`.")
    else:
        lines.append("- Nota: el modo offline ejecuta solo BM25; vectorial y LLM reranker requieren `--live`.")

    lines.extend([
        "",
        "## Resumen",
        "",
        "| Método | Hit@1 | Hit@3 | MRR@3 | Latencia media |",
        "|---|---:|---:|---:|---:|",
    ])
    for method, metrics in result["summary"].items():
        lines.append(
            f"| {method} | {metrics['hit_at_1']:.3f} | {metrics['hit_at_3']:.3f} | "
            f"{metrics['mrr_at_3']:.3f} | {metrics['avg_latency_ms']:.2f} ms |"
        )

    lines.extend(["", "## Detalle por query", "", "| Query | Relevantes | BM25"])
    if result["mode"] == "live":
        lines[-1] += " | Vectorial | Híbrido RRF"
        if result["reranker_enabled"]:
            lines[-1] += " | Híbrido + LLM reranker"
    lines[-1] += " |"

    separator = "|---|---|---|"
    if result["mode"] == "live":
        separator += "---|---|"
        if result["reranker_enabled"]:
            separator += "---|"
    lines.append(separator)
    for detail in result["details"]:
        row = [
            detail["id"],
            ", ".join(detail["relevant_ids"]),
            ", ".join(detail.get("BM25", [])),
        ]
        if result["mode"] == "live":
            row.extend([
                ", ".join(detail.get("Vectorial", [])),
                ", ".join(detail.get("Híbrido RRF", [])),
            ])
            if result["reranker_enabled"]:
                row.append(", ".join(detail.get("Híbrido + LLM reranker", [])))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara las estrategias de retrieval del LLM CLI.")
    parser.add_argument("--live", action="store_true", help="Usa embeddings Gemini y el LLM reranker.")
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Mide embeddings/vectorial/híbrido sin llamar al modelo generativo.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limita el número de queries.")
    parser.add_argument("--output", type=Path, help="Guarda el reporte Markdown en esta ruta.")
    args = parser.parse_args()

    if args.skip_reranker and not args.live:
        parser.error("--skip-reranker requiere --live.")

    result = run_benchmark(
        load_dataset(),
        live=args.live,
        limit=args.limit,
        include_reranker=not args.skip_reranker,
    )
    report = render_markdown(result)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Reporte guardado en: {args.output}")


if __name__ == "__main__":
    main()
