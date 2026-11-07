# Retrieval benchmark

- Generado: `2026-11-06T21:32:20-05:00`
- Modo: `offline`
- Corpus: `12` documentos / `8` queries
- Nota: el modo offline ejecuta solo BM25; vectorial y LLM reranker requieren `--live`.

## Resumen

| Método | Hit@1 | Hit@3 | MRR@3 | Latencia media |
|---|---:|---:|---:|---:|
| BM25 | 0.750 | 0.875 | 0.812 | 0.16 ms |

## Detalle por query

| Query | Relevantes | BM25 |
|---|---|---|
| q01 | memory-001 | memory-001, memory-006, memory-008 |
| q02 | memory-002 | memory-005, memory-002, memory-001 |
| q03 | memory-003 | memory-003, memory-001, memory-012 |
| q04 | memory-004 | memory-001, memory-003 |
| q05 | memory-005 | memory-005, memory-001, memory-011 |
| q06 | memory-008, memory-001, memory-002 | memory-001, memory-009, memory-006 |
| q07 | memory-009 | memory-009, memory-001, memory-005 |
| q08 | memory-010 | memory-010, memory-012, memory-004 |
