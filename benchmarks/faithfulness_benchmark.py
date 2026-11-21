"""Reporta faithfulness a partir de anotaciones de claims y fuentes.

Este script no llama a un LLM ni inventa etiquetas de soporte. Consume un
JSONL donde cada claim ya fue revisado por una persona o por un juez LLM y
calcula métricas agregadas sobre esa anotación.

Uso:
    python benchmarks/faithfulness_benchmark.py \
        --input benchmarks/faithfulness_annotations.jsonl \
        --output benchmarks/results-faithfulness.md

Formato por línea:
    {
      "id": "q01",
      "answer": "...",
      "gold_source_ids": ["memory-001"],
      "claims": [
        {
          "text": "...",
          "supported": true,
          "source_ids": ["memory-001"]
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_annotations(path: Path) -> list[dict[str, Any]]:
    """Carga anotaciones JSONL y valida el contrato mínimo."""
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON inválido en línea {line_number}.") from exc
        if not isinstance(row, dict) or not row.get("id"):
            raise ValueError(f"La línea {line_number} debe tener un `id`.")
        if not isinstance(row.get("gold_source_ids"), list):
            raise ValueError(f"La línea {line_number} debe tener `gold_source_ids`.")
        claims = row.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError(f"La línea {line_number} debe tener claims no vacíos.")
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("supported"), bool):
                raise ValueError(
                    f"Cada claim de la línea {line_number} requiere `supported: bool`."
                )
            if not isinstance(claim.get("source_ids", []), list):
                raise ValueError(
                    f"Cada claim de la línea {line_number} requiere `source_ids` como lista."
                )
        rows.append(row)
    if not rows:
        raise ValueError("El archivo de anotaciones no contiene filas.")
    return rows


def evaluate_annotations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula soporte de claims y precisión/recall de citas."""
    claims = [claim for row in rows for claim in row["claims"]]
    supported = sum(1 for claim in claims if claim["supported"])
    gold_sources = sum(len(set(row["gold_source_ids"])) for row in rows)
    cited_sources = 0
    relevant_citations = 0
    citation_recall_hits = 0

    for row in rows:
        gold = set(row["gold_source_ids"])
        for claim in row["claims"]:
            cited = set(claim.get("source_ids", []))
            cited_sources += len(cited)
            relevant = cited & gold
            relevant_citations += len(relevant)
            citation_recall_hits += len(relevant)

    return {
        "answers_evaluated": len(rows),
        "claims_evaluated": len(claims),
        "claims_supported": supported,
        "claim_support_rate": round(supported / len(claims), 3) if claims else 0.0,
        "unsupported_claim_rate": round(1 - supported / len(claims), 3) if claims else 0.0,
        "citation_precision": round(relevant_citations / cited_sources, 3)
        if cited_sources
        else 0.0,
        "citation_recall": round(citation_recall_hits / gold_sources, 3)
        if gold_sources
        else 0.0,
    }


def render_markdown(metrics: dict[str, Any]) -> str:
    """Renderiza un reporte explícito sobre el origen anotado de las etiquetas."""
    return "\n".join([
        "# Faithfulness benchmark",
        "",
        "- Método: anotaciones de claims y fuentes; no se invoca ningún LLM.",
        "- Las etiquetas `supported` deben provenir de revisión humana o de un juez LLM.",
        f"- Respuestas evaluadas: `{metrics['answers_evaluated']}`",
        f"- Claims evaluados: `{metrics['claims_evaluated']}`",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
        f"| Claim support rate | {metrics['claim_support_rate']:.3f} |",
        f"| Unsupported claim rate | {metrics['unsupported_claim_rate']:.3f} |",
        f"| Citation precision | {metrics['citation_precision']:.3f} |",
        f"| Citation recall | {metrics['citation_recall']:.3f} |",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa faithfulness desde anotaciones JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL de anotaciones.")
    parser.add_argument("--output", type=Path, help="Ruta opcional del reporte Markdown.")
    args = parser.parse_args()

    report = render_markdown(evaluate_annotations(load_annotations(args.input)))
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Reporte guardado en: {args.output}")


if __name__ == "__main__":
    main()
