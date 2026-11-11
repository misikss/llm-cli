# 🤖 LLM CLI — Gemini Modular con Semantic Router, Grounding y Zero-Waste RAG

CLI profesional de ingeniería de IA para interactuar con la API oficial de **Google Gemini (`google-genai`)**, validado con **Pydantic**, con soporte para **imágenes/audio**, **Google Search Grounding en vivo**, **consulta de clima en tiempo real (Tool Calling)**, **Semantic Routing Few-Shot** y **Advanced RAG con Gatekeeper de Costo Cero (Zero-Waste RAG)**.

---

## 🎯 Arquitectura de Enrutamiento Inteligente (Zero-Waste RAG)

A diferencia de los sistemas RAG ingenuos que ejecutan reescrituras de consultas y búsquedas vectoriales a ciegas para cada frase, este sistema implementa un **Semantic Router Few-Shot** como guardián de entrada.

El router clasifica la intención considerando el **contexto activo de la conversación (Working Memory)** y solo dispara el pipeline costoso si realmente se requiere recuperar información histórica de ChromaDB:

```mermaid
flowchart TD
    UserQuery["👤 Pregunta del Usuario"] --> RAMContext["0. Contexto de RAM (Working Memory)<br/>(Costo $0 USD)"]
    
    RAMContext --> Router["1. Semantic Router Few-Shot (determine_route)<br/>Single Source of Truth"]
    
    Router --> RouteDec{"Decisión de Ruta"}
    
    RouteDec -- "needs_memory_rag: true<br/>('rag_memory')" --> RAGPipeline["2. Advanced RAG Pipeline (Gateado)"]
    RouteDec -- "'web_grounding'<br/>(Temas dinámicos / Estado del arte)" --> WebGrounding["Google Search Grounding<br/>(Información viva en la web)"]
    RouteDec -- "'weather'<br/>(Meteorología)" --> WeatherTool["Tool Calling (Open-Meteo API)<br/>(Sin gasto en RAG)"]
    RouteDec -- "'general'<br/>(Lógica / Redacción / Código)" --> DirectLLM["Generación Estructurada Pydantic<br/>(Contexto RAM inmediato, $0 overhead)"]
    
    subgraph RAGPipeline ["Pipeline RAG Completo (Solo cuando se necesita)"]
        QE["rewrite_query()<br/>(Desambiguación + Keywords BM25)"] --> Hybrid["Hybrid Search (BM25 + ChromaDB)"]
        Hybrid --> RRF["Reciprocal Rank Fusion (RRF)"]
        RRF --> LLMReranker["LLM Re-ranking<br/>(Gemini Structured Output)"]
        LLMReranker --> FilterScore{"Score ≥ RERANK_MIN_SCORE (0.45)"}
    end
    
    FilterScore --> LLMFinal["Generación Final Enriquecida"]
    WebGrounding --> LLMFinal
    WeatherTool --> LLMFinal
    DirectLLM --> LLMFinal
    
    LLMFinal --> UpdateMem["3. Actualización de Memoria"]
    UpdateMem --> CheckEphemeral{"¿is_ephemeral?"}
    CheckEphemeral -- "No" --> ChromaPersist[("ChromaDB Persistente")]
    CheckEphemeral -- "Sí (clima / cotizaciones)" --> SkipChroma["Solo actualizar RAM (Anti-contaminación)"]
```

---

### 🚦 Las 4 Rutas Semánticas Universales

El Router no utiliza listas frágiles de palabras clave estáticas, sino comprensión contextual multitemática mediante Few-Shot Learning:

| Ruta | Cuándo se activa | Comportamiento | Overhead RAG |
| :--- | :--- | :--- | :--- |
| **`web_grounding`** | Temas dinámicos, novedades, tendencias, estado del arte o roadmaps técnicos en constante evolución (*"roadmap de AI Engineer"*, *"mejores frameworks web en 2025/2026"*). | Consulta Google Search en tiempo real y extrae fuentes verificadas. | **0 tokens ($0.00)** |
| **`weather`** | Consultas meteorológicas de cualquier ciudad o región (*"clima en Madrid hoy"*). | Llama a la herramienta de Open-Meteo vía Tool Calling. | **0 tokens ($0.00)** |
| **`rag_memory`** | Peticiones explícitas de recuperar notas, acuerdos o recuerdos archivados en sesiones anteriores (*"¿recuerdas qué receta me guardaste la semana pasada?"*). | Dispara el pipeline completo: `rewrite_query` $\rightarrow$ Hybrid Search $\rightarrow$ RRF $\rightarrow$ Re-ranking. | Tokens medidos |
| **`general`** | Conceptos atemporales, matemáticas, lógica, explicaciones teóricas, redacción o código estándar (*"qué es un decorador en Python"*, *"cómo funciona la fotosíntesis"*). | Responde directamente con el contexto inmediato de la Working Memory. | **0 tokens ($0.00)** |

---

### 🔬 Componentes del Pipeline RAG

Cuando una consulta requiere memoria histórica (`rag_memory`), se activan los tres eslabones:

1. **Query Engineering (`rewrite_query`)**:
   - Resuelve pronombres (*"¿cuándo llegó él?"* $\rightarrow$ *"¿cuándo llegó Lionel Messi?"*) y extrae palabras clave para el motor léxico.
2. **Hybrid Search (BM25 + Vectorial) con RRF**:
   - Búsqueda léxica precisa con `rank_bm25` (BM25Okapi), con índice cacheado e IDs estables para no reconstruir el corpus en cada consulta.
   - Búsqueda semántica densa con `gemini-embedding-2` en ChromaDB.
   - Fusión mediante Reciprocal Rank Fusion:
     $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$
3. **LLM Re-ranking (`rerank_documents`)**:
   - Emplea `gemini-2.5-flash-lite` con Structured Output para evaluar la relevancia exacta de cada fragmento.
   - Es un reranker basado en LLM, no un Cross-Encoder local; evita cargar otro modelo en RAM.
   - Parámetros centralizados en `config.py`: `RERANK_MIN_SCORE = 0.45` y `RERANK_TOP_N = 2`.

---

## 📊 Benchmark de Retrieval

El proyecto incluye un dataset sintético versionado en `benchmarks/retrieval_dataset.json` para comparar cuatro estrategias con relevancia esperada:

```mermaid
flowchart LR
    Dataset["Dataset versionado<br/>corpus + queries etiquetadas"] --> BM25["BM25"]
    Dataset --> Vector["Vectorial<br/>Gemini embeddings"]
    Dataset --> Hybrid["Híbrido RRF"]
    Dataset --> Reranker["Híbrido + LLM reranker"]
    BM25 --> Metrics["Hit@1 · Hit@3 · MRR@3<br/>Latencia media"]
    Vector --> Metrics
    Hybrid --> Metrics
    Reranker --> Metrics
```

El modo offline valida BM25 sin consumir API:

```bash
python benchmarks/retrieval_benchmark.py
```

El benchmark completo usa `gemini-embedding-2` para la búsqueda vectorial y Gemini para el reranking. No utiliza tus logs ni tu ChromaDB personal:

```bash
python benchmarks/retrieval_benchmark.py --live --output benchmarks/results.md
```

Si la cuota generativa está agotada, puedes medir embeddings, vectorial y RRF sin invocar el reranker:

```bash
python benchmarks/retrieval_benchmark.py --live --skip-reranker
```

Reporta `Hit@1`, `Hit@3`, `MRR@3`, latencia media y tokens usados por el reranker. El benchmark no aplica query rewriting para aislar la calidad del retrieval.

Reportes generados: [`results-live-no-reranker.md`](benchmarks/results-live-no-reranker.md) y [`results-offline.md`](benchmarks/results-offline.md).

La vista arquitectónica completa está en [`docs/architecture.md`](docs/architecture.md).

---

## 📁 Arquitectura Modular del Proyecto

```
llm-cli/
├── config.py           ← Configuración global (.env), umbrales RERANK_MIN_SCORE y tabla de tarifas
├── models.py           ← Esquemas Pydantic (RouteDecision semántico, LLMResponse, CallMetrics)
├── tools.py            ← Herramientas ejecutables (Clima en tiempo real con Open-Meteo)
├── rag.py              ← Advanced RAG (rewrite_query, BM25, RRF y LLM Reranking)
├── memory.py           ← Memoria híbrida (RAM Working Memory + Episodic Memory en ChromaDB)
├── client.py           ← Lógica con Google GenAI SDK (Semantic Router Few-Shot, Grounding, Failover)
├── service.py          ← Caso de uso AssistantService, independiente de la CLI
├── errors.py           ← Errores tipados, clasificación de cuota y fallos temporales
├── observability.py    ← Trace por solicitud, etapas, latencia y fallbacks
├── display.py          ← Renderizado de UI en consola con Rich y persistencia en JSONL
├── llm_cli.py          ← Punto de entrada principal y bucle interactivo optimizado
├── benchmarks/         ← Dataset y runner para evaluar retrieval
├── docs/architecture.md ← Diagrama y decisiones de arquitectura
├── tests/              ← Tests unitarios del retrieval
├── .env                ← Variables de entorno (GEMINI_API_KEY, GEMINI_MODEL)
├── requirements.txt    ← Dependencias necesarias
└── chroma_db/          ← Base de datos vectorial persistente con embeddings
```

La CLI solo gestiona argumentos, input y presentación. `AssistantService` coordina el routing, el retrieval, la generación y la actualización de memoria, por lo que la misma lógica puede reutilizarse más adelante desde una API.

Cada llamada exitosa registra un `trace_id`, latencia total, etapas (`route`, `query_rewrite`, `retrieval`, `generation`, `memory_update`), tokens, fallbacks y errores sanitizados en `logs/history.jsonl`. Los fallos de solicitudes se registran en `logs/errors.jsonl`.

### Contexto aislado por usuario y conversación

Cada solicitud admite tres identificadores:

- `user_id`: propietario del contexto.
- `conversation_id`: chat lógico cuyos recuerdos se pueden reanudar entre ejecuciones.
- `session_id`: ejecución concreta del proceso; sirve para trazabilidad y separa la Working Memory en RAM.

La memoria episódica de ChromaDB se filtra por `user_id + conversation_id`, mientras que la Working Memory se mantiene separada por los tres valores. Así, una nueva sesión puede continuar la misma conversación sin mezclarla con otro usuario o chat. El CLI usa `local-user` y `default-conversation` por defecto, y genera un `session_id` nuevo por proceso.

```bash
# Reanudar una conversación concreta
python llm_cli.py --user-id victor --conversation-id portfolio-ai -q "¿Qué decidimos sobre el reranker?"

# Forzar una sesión identificable para logs y debugging
python llm_cli.py --user-id victor --conversation-id portfolio-ai --session-id demo-01 -q "Resume el contexto"
```

Los recuerdos creados antes de esta funcionalidad se migran una sola vez de forma idempotente al scope local por defecto (`local-user` / `default-conversation`).

---

## 🚀 Instalación y Uso

```bash
cd llm-cli
pip install -r requirements.txt
```

### Modos de Ejecución

```bash
# Modo interactivo continuo con Working Memory y Semantic Routing:
python llm_cli.py

# Preguntas directas (-q):
# 1. Detecta automáticamente que requiere estado del arte y activa Google Grounding:
python llm_cli.py -q "me armas un roadmap de AI engineer"

# 2. Detecta que es un concepto atemporal y responde directo (0 tokens de overhead RAG):
python llm_cli.py -q "¿Qué es un decorador en Python?"

# 3. Detecta consulta meteorológica y ejecuta la tool de clima:
python llm_cli.py -q "¿Qué temperatura hace en Buenos Aires ahorita?"

# Consultar historial y estadísticas acumuladas:
python llm_cli.py --history
python llm_cli.py --stats
python llm_cli.py --modelos
```

---

## 💰 Medición Precisa del ROI en Tokens

Cada llamada registra en [`logs/history.jsonl`](file:///c:/Users/victor/Downloads/aiengineer/logs/history.jsonl):
- Tokens del semantic router, query rewriting, LLM reranker, generación final y actualización de memoria.
- Cantidad de llamadas de embedding y sus tokens cuando el SDK expone `usage_metadata`.
- `token_usage_scope`: `pipeline` cuando todas las etapas reportan usage, o `pipeline_partial` cuando el proveedor no expone tokens de alguna etapa.
- La latencia se desglosa por etapa para distinguir costo de API, retrieval local y persistencia.
- El overhead RAG sigue siendo **0 tokens** para rutas que no activan memoria episódica, aunque el router y la actualización de memoria pueden realizar llamadas propias.
