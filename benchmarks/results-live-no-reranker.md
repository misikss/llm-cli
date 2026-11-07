# Retrieval benchmark

- Generado: `2026-11-06T21:31:19-05:00`
- Modo: `live`
- Corpus: `12` documentos / `8` queries
- Embeddings: `gemini-embedding-2`
- Modelo del proyecto: `gemini-2.5-flash`
- Tiempo de embeddings del corpus: `7944.56 ms`
- Tokens del reranker: `0` input / `0` output
- Nota: el LLM reranker fue omitido con `--skip-reranker`.

## Resumen

| Método | Hit@1 | Hit@3 | MRR@3 | Latencia media |
|---|---:|---:|---:|---:|
| BM25 | 0.750 | 0.875 | 0.812 | 0.12 ms |
| Vectorial | 0.875 | 1.000 | 0.938 | 639.76 ms |
| Híbrido RRF | 1.000 | 1.000 | 1.000 | 639.95 ms |

## Detalle por query

| Query | Relevantes | BM25 | Vectorial | Híbrido RRF |
|---|---|---|---|---|
| q01 | memory-001 | memory-001, memory-006, memory-008 | memory-001, memory-007, memory-006 | memory-001, memory-006, memory-007 |
| q02 | memory-002 | memory-005, memory-002, memory-001 | memory-002, memory-006, memory-003 | memory-002, memory-005, memory-006 |
| q03 | memory-003 | memory-003, memory-001, memory-012 | memory-006, memory-003, memory-007 | memory-003, memory-006, memory-001 |
| q04 | memory-004 | memory-001, memory-003 | memory-004, memory-007, memory-006 | memory-004, memory-001, memory-007 |
| q05 | memory-005 | memory-005, memory-001, memory-011 | memory-005, memory-001, memory-008 | memory-005, memory-001, memory-008 |
| q06 | memory-008, memory-001, memory-002 | memory-001, memory-009, memory-006 | memory-001, memory-007, memory-003 | memory-001, memory-007, memory-009 |
| q07 | memory-009 | memory-009, memory-001, memory-005 | memory-009, memory-007, memory-004 | memory-009, memory-004, memory-007 |
| q08 | memory-010 | memory-010, memory-012, memory-004 | memory-010, memory-007, memory-012 | memory-010, memory-012, memory-007 |
