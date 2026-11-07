# Arquitectura

## Flujo de una consulta

```mermaid
flowchart TD
    UserQuery["CLI o futura API<br/>Pregunta del usuario"] --> Service["AssistantService<br/>caso de uso"]
    Service --> WorkingMemory["Working Memory<br/>RAM"]
    WorkingMemory --> Router["Semantic Router<br/>determine_route"]
    Router --> Decision{"Ruta"}

    Decision -->|general| Direct["Generación estructurada<br/>Pydantic"]
    Decision -->|web_grounding| Grounding["Google Search<br/>Grounding"]
    Decision -->|weather| Weather["Open-Meteo<br/>Tool Calling"]
    Decision -->|rag_memory| QueryEngineering["rewrite_query"]

    QueryEngineering --> Retrieval["Retrieval híbrido"]
    Retrieval --> BM25["BM25 cacheado<br/>ranking léxico"]
    Retrieval --> Vector["ChromaDB + embeddings<br/>ranking semántico"]
    BM25 --> RRF["Reciprocal Rank Fusion<br/>por ID estable"]
    Vector --> RRF
    RRF --> Reranker["LLM reranker<br/>Gemini Structured Output"]
    Reranker --> Final["Contexto recuperado"]

    Direct --> Answer["Respuesta final"]
    Grounding --> Answer
    Weather --> Answer
    Final --> Answer
    Answer --> Update["Actualizar memoria"]
    Update --> Persist["ChromaDB + JSONL"]
    Service --> Trace["RequestTrace<br/>trace_id + etapas + errores"]
    Trace --> History["history.jsonl / errors.jsonl"]
```

## Decisiones importantes

- BM25 se construye una vez y se invalida cuando crece la colección de ChromaDB.
- BM25, vector search y RRF usan IDs estables; el texto se recupera después para evitar colisiones entre documentos iguales.
- El reranking se ejecuta solo sobre pocos candidatos y usa Gemini como LLM reranker. No se carga un cross-encoder local para mantener bajo el consumo de RAM.
- Las consultas que no necesitan memoria no ejecutan el pipeline RAG.
- `llm_cli.py` no contiene la lógica del caso de uso: solo convierte argumentos/input en `AskRequest` y presenta `AskResult`.
- Los errores se clasifican como configuración, proveedor, cuota, modelo no disponible, retrieval, memoria o tool execution.
- Cada solicitud produce un `trace_id` y registra latencia por etapa sin persistir secretos completos.
- La medición de tokens cubre router, query rewriting, reranker, generación y memory update; embeddings registran cantidad de llamadas y tokens si el SDK los expone.

## Evaluación

El benchmark versionado en `benchmarks/` compara BM25, búsqueda vectorial, híbrido RRF y el híbrido con LLM reranker usando Hit@1, Hit@3, MRR@3 y latencia.

```mermaid
flowchart LR
    Dataset["Dataset versionado<br/>corpus + relevancia esperada"] --> BM25Eval["BM25"]
    Dataset --> VectorEval["Vectorial<br/>Gemini embeddings"]
    Dataset --> HybridEval["Híbrido RRF"]
    Dataset --> RerankEval["Híbrido + LLM reranker"]
    BM25Eval --> Metrics["Hit@1 · Hit@3 · MRR@3 · latencia"]
    VectorEval --> Metrics
    HybridEval --> Metrics
    RerankEval --> Metrics
```
