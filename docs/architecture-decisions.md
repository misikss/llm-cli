# Registro de decisiones de arquitectura

Este documento reúne las decisiones que cierran la versión actual del
proyecto. Las decisiones describen el sistema implementado, no un roadmap
futuro.

## ADR-001 — Mantener `AssistantService` como caso de uso central

**Decisión:** la CLI y el dashboard delegan en `AssistantService`.

**Motivo:** evita duplicar routing, retrieval, generación, memoria y métricas.
Permite cambiar la interfaz sin alterar el pipeline.

## ADR-002 — Elegir análisis directo o Document RAG según el documento

**Decisión:** los documentos pequeños se envían completos al modelo multimodal.
Los documentos mayores de 2 MB o consultados al menos tres veces por el mismo
usuario se extraen, fragmentan e indexan en ChromaDB.

**Motivo:** el análisis directo ofrece mejor simplicidad para una consulta
puntual; Document RAG evita repetir un contexto grande cuando el archivo se
reutiliza.

**Límite:** imágenes y audio permanecen en el camino multimodal. XLS antiguo
requiere conversión a XLSX o CSV para extracción local.

## ADR-003 — Aislar los chunks documentales

**Decisión:** Document RAG usa la colección `document_chunks`, con filtro por
`user_id`, `conversation_id` y hash del documento.

**Motivo:** los chunks de un archivo no deben mezclarse con recuerdos
episódicos ni cruzarse entre usuarios o conversaciones.

## ADR-004 — Fallback seguro al multimodal directo

**Decisión:** si falla la extracción, el embedding o el retrieval documental,
se registra un fallback y se conserva el envío multimodal original.

**Motivo:** el nuevo camino mejora documentos grandes sin romper el caso base
que ya funcionaba.

## ADR-005 — Contexto del dashboard generado, no editable

**Decisión:** el dashboard mantiene una lista local de usuarios, permite crear
uno nuevo y permite cambiar entre los ya creados. `conversation_id` y
`session_id` son de solo lectura y se generan automáticamente.

**Motivo:** evita que una edición manual mezcle accidentalmente memoria o
historial y hace explícita la separación de contexto.

## ADR-006 — CI mínima reproducible

**Decisión:** GitHub Actions instala `requirements-dev.txt` y ejecuta `pytest`.

**Motivo:** cada push debe comprobar que la capa de servicio, documentos,
retrieval y dashboard siguen funcionando juntos.

## ADR-007 — No introducir LangGraph en esta entrega

**Decisión:** esta versión conserva un pipeline síncrono y modular.

**Motivo:** el objetivo actual es cerrar una implementación entendible y
observable. Un workflow persistente de investigación sería una ampliación
distinta y no es necesario para estas decisiones documentales.
