"""
config.py — Configuración global, tarifas y constantes del CLI.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Desactivar telemetría anónima de ChromaDB (evita avisos innecesarios en consola)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Lista de API Keys para rotación automática y failover
_raw_keys = os.getenv("GEMINI_API_KEYS", "")
GEMINI_API_KEYS: list[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]

# Si no está GEMINI_API_KEYS, buscamos GEMINI_API_KEY y GEMINI_API_KEY_2
if not GEMINI_API_KEYS:
    k1 = os.getenv("GEMINI_API_KEY")
    k2 = os.getenv("GEMINI_API_KEY_2")
    GEMINI_API_KEYS = [k for k in (k1, k2) if k]

GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Lista de modelos de respaldo en orden de prioridad ante errores 429 (Resource Exhausted / Quota Limit)
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

# Rutas de persistencia
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "history.jsonl"
ERROR_LOG_FILE = LOG_DIR / "errors.jsonl"

# Parámetros del LLM Re-ranking
RERANK_MIN_SCORE: float = 0.45
RERANK_TOP_N: int = 2

# Parámetros de Document RAG. El reranking se activa solo cuando BM25 y la
# búsqueda vectorial discrepan de forma relevante sobre los primeros candidatos.
DOCUMENT_VECTOR_TOP_K: int = 8
DOCUMENT_BM25_TOP_K: int = 8
DOCUMENT_HYBRID_TOP_K: int = 6
DOCUMENT_RERANK_TOP_N: int = 3
DOCUMENT_RERANK_MIN_SCORE: float = 0.45
DOCUMENT_INDEX_WORKERS: int = 2
DOCUMENT_EMBED_BATCH_SIZE: int = 16

# Tarifas reales por modelo (USD por 1 millón de tokens)
# Fuente: https://ai.google.dev/pricing
PRECIOS_MODELO: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {
        "input": 0.075,
        "output": 0.30,
        "output_thinking": 3.50,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.075,
        "output": 0.30,
        "output_thinking": 0.0,
    },
    "gemini-3.1-flash-lite": {
        "input": 0.075,
        "output": 0.30,
        "output_thinking": 0.0,
    },
    "gemini-3.5-flash-lite": {
        "input": 0.075,
        "output": 0.30,
        "output_thinking": 0.0,
    },
    "gemini-3.5-flash": {
        "input": 0.075,
        "output": 0.30,
        "output_thinking": 3.50,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
        "output_thinking": 3.50,
    },
    "gemini-2.0-flash": {
        "input": 0.075,
        "output": 0.30,
        "output_thinking": 0.0,
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.30,
        "output_thinking": 0.0,
    },
    "gemini-1.5-pro": {
        "input": 3.50,
        "output": 10.50,
        "output_thinking": 0.0,
    },
}

TARIFA_DESCONOCIDA: dict[str, float] = {
    "input": 0.0,
    "output": 0.0,
    "output_thinking": 0.0,
}


def get_tarifas(modelo: str) -> dict[str, float]:
    """Obtiene las tarifas por millón de tokens según el modelo activo."""
    if modelo in PRECIOS_MODELO:
        return PRECIOS_MODELO[modelo]
    for clave, tarifas in PRECIOS_MODELO.items():
        if clave in modelo:
            return tarifas
    return TARIFA_DESCONOCIDA

