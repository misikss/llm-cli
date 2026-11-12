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
usuario se extraen, fragmentan e indexan en ChromaDB mediante un retrieval
híbrido: BM25, búsqueda vectorial, RRF y reranking LLM condicional.

**Motivo:** el análisis directo ofrece mejor simplicidad para una consulta
puntual; Document RAG evita repetir un contexto grande cuando el archivo se
reutiliza. BM25 recupera códigos, nombres y cifras exactas; la búsqueda
vectorial recupera formulaciones semánticamente parecidas.

**Límite:** imágenes y audio permanecen en el camino multimodal. XLS antiguo
requiere conversión a XLSX o CSV para extracción local.

## ADR-003 — Usar chunking estructural antes que late chunking

**Decisión:** PDF, DOCX y TXT conservan páginas, párrafos y tablas; XLSX, CSV
y TSV conservan hojas y grupos de filas. El texto sin estructura usa ventanas
de 1.800 caracteres con 250 de solapamiento.

**Motivo:** este enfoque conserva procedencia, funciona con datos tabulares y no
requiere cargar otro modelo local. Late chunking queda fuera de esta versión
porque necesita soporte específico del modelo de embeddings y no aporta una
ventaja demostrada para las tablas del proyecto.

## ADR-004 — Aislar los chunks documentales

**Decisión:** Document RAG usa la colección `document_chunks`, con filtro por
`user_id`, `conversation_id` y hash del documento.

**Motivo:** los chunks de un archivo no deben mezclarse con recuerdos
episódicos ni cruzarse entre usuarios o conversaciones.

## ADR-005 — Fallback seguro al multimodal directo

**Decisión:** si falla la extracción, el embedding o el retrieval documental,
se registra un fallback y se conserva el envío multimodal original.

**Motivo:** el nuevo camino mejora documentos grandes sin romper el caso base
que ya funcionaba.

## ADR-006 — Contexto del dashboard generado, no editable

**Decisión:** el dashboard mantiene una lista local de usuarios, permite crear
uno nuevo y permite cambiar entre los ya creados. `conversation_id` y
`session_id` son de solo lectura y se generan automáticamente.

**Motivo:** evita que una edición manual mezcle accidentalmente memoria o
historial y hace explícita la separación de contexto.

## ADR-007 — CI mínima reproducible

**Decisión:** GitHub Actions instala `requirements-dev.txt` y ejecuta `pytest`.

**Motivo:** cada push debe comprobar que la capa de servicio, documentos,
retrieval y dashboard siguen funcionando juntos.

## ADR-008 — No introducir LangGraph en esta entrega

**Decisión:** esta versión conserva una orquestación síncrona y modular, con un
worker local separado para la indexación documental.

**Motivo:** el objetivo actual es cerrar una implementación entendible y
observable. Un workflow persistente de investigación sería una ampliación
distinta y no es necesario para estas decisiones documentales.

## ADR-009 — Indexar documentos en segundo plano

**Decisión:** todo documento con extracción local se copia a una caché
persistente y se indexa mediante un worker local. La primera consulta no espera
la indexación: usa el camino multimodal directo. Las consultas posteriores
usan Document RAG cuando el estado del índice es `ready`.

**Motivo:** la indexación de muchos chunks es trabajo de preparación, no una
dependencia necesaria para responder la primera pregunta. Separar ambas cosas
reduce el tiempo percibido y permite reutilizar el índice.

**Límite:** si el planificador detecta un documento por encima del presupuesto
seguro de multimodal, la primera consulta puede esperar al índice como
excepción de seguridad. Si esa indexación falla, se devuelve un resultado
controlado y no se intenta enviar el archivo completo. El documento no se aplica automáticamente a preguntas no
relacionadas. Para cambiar el contexto documental se debe adjuntar el archivo
correspondiente; un selector explícito de `document_id` sería una ampliación
posterior.
