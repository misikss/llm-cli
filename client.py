"""
client.py — Orquestación de llamadas a Google GenAI con Pydantic, Grounding y Multimodal.
"""

from datetime import datetime
import mimetypes
import logging
from pathlib import Path

from google import genai
from google.genai import types as genai_types

from config import (
    GEMINI_API_KEY,
    GEMINI_API_KEYS,
    GEMINI_MODEL,
    FALLBACK_MODELS,
    get_tarifas,
)
from models import LLMResponse, CallMetrics, RouteDecision
from tools import get_current_weather
from errors import ProviderError, classify_provider_error


logger = logging.getLogger(__name__)


def call_with_fallback(client: genai.Client, contents: list | str, config: genai_types.GenerateContentConfig):
    """
    Estrategia de resiliencia de dos dimensiones:
    Para cada modelo (comenzando por el principal GEMINI_MODEL):
      - Prueba primero con la Key 1.
      - Si la Key 1 agota su cuota (429), prueba el MISMO modelo con la Key 2.
    Solo si TODAS las API keys agotan su cuota para ese modelo, conmuta al siguiente
    modelo de respaldo en FALLBACK_MODELS (repitiendo el ciclo de keys).
    """
    keys = GEMINI_API_KEYS if GEMINI_API_KEYS else [GEMINI_API_KEY]
    candidates = [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]

    last_error: ProviderError | None = None

    for model_name in candidates:
        for key_idx, api_key in enumerate(keys):
            active_client = client if key_idx == 0 and api_key == GEMINI_API_KEY else genai.Client(api_key=api_key)
            try:
                response = active_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return response, model_name
            except Exception as exc:
                provider_error = classify_provider_error(exc, model=model_name)
                logger.warning(
                    "provider_attempt_failed model=%s key_index=%s error_code=%s retryable=%s",
                    model_name,
                    key_idx + 1,
                    provider_error.code,
                    provider_error.retryable,
                )
                if provider_error.retryable:
                    from display import console
                    if provider_error.code == "quota_exceeded":
                        console.print(
                            f"[yellow][!] Cuota en '{model_name}' con Key {key_idx + 1}/{len(keys)}. Probando siguiente clave...[/yellow]"
                        )
                    last_error = provider_error
                    continue
                raise provider_error from exc

    raise last_error or ProviderError(
        "No se pudo obtener respuesta de ningún modelo ni API key."
    )


def prepare_multimodal_content(prompt_text: str, file_path: str | Path | None = None) -> list:
    """Prepara el payload multimodal incluyendo archivos si existen."""
    contents = [prompt_text]
    if not file_path:
        return contents

    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {p}")

    mime_type, _ = mimetypes.guess_type(str(p))
    if not mime_type:
        ext = p.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        elif ext == ".webp":
            mime_type = "image/webp"
        elif ext == ".mp3":
            mime_type = "audio/mp3"
        elif ext == ".wav":
            mime_type = "audio/wav"
        elif ext == ".ogg":
            mime_type = "audio/ogg"
        elif ext == ".pdf":
            mime_type = "application/pdf"
        else:
            mime_type = "application/octet-stream"

    file_bytes = p.read_bytes()
    multimodal_part = genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    contents.append(multimodal_part)
    return contents


def determine_route(
    client: genai.Client,
    question: str,
    active_subject: str = "",
    recent_context: str = ""
) -> RouteDecision:
    """
    Router semántico con Few-Shot Learning y Structured Outputs:
    Clasifica la intención del usuario considerando el contexto activo de la conversación.
    """
    current_date_str = datetime.now().strftime("%Y-%m-%d (%A)")

    router_prompt = (
        f"Fecha y hora actual del sistema: {current_date_str}\n\n"
        f"Eres un clasificador semántico de intenciones para un asistente avanzado.\n"
        f"Determina la ruta óptima para responder a la consulta del usuario, entendiendo la intención de fondo "
        f"y si se beneficia de datos en tiempo real, recuerdos históricos o razonamiento/explicación directa.\n\n"
        f"Rutas disponibles:\n"
        f"1. 'web_grounding': Para temas dinámicos que cambian con el tiempo, novedades, estado del arte, tendencias, "
        f"comparativas actuales, noticias, cotizaciones, o roadmaps/guías de industrias o tecnologías en constante evolución.\n"
        f"2. 'weather': Para meteorología, clima, lluvia, temperatura o pronóstico de cualquier lugar.\n"
        f"3. 'rag_memory': ÚNICAMENTE si el usuario pide buscar o recordar notas, preferencias, acuerdos o temas pasados guardados en su historial.\n"
        f"4. 'general': Para conceptos atemporales, principios fundamentales, matemáticas, lógica, redacción, filosofía, "
        f"explicaciones teóricas o código algorítmico estándar.\n\n"
        f"Ejemplos Few-Shot de clasificación:\n"
        f"- Contexto: '' | Pregunta: '¿Qué es la fotosíntesis y cómo funciona?'\n"
        f"  -> route: 'general', is_ephemeral: false, needs_memory_rag: false, reason: 'Concepto biológico atemporal'\n"
        f"- Contexto: '' | Pregunta: '¿Cuál es el clima actual en Bogotá?'\n"
        f"  -> route: 'weather', is_ephemeral: true, needs_memory_rag: false, reason: 'Consulta meteorológica en vivo'\n"
        f"- Contexto: 'AI Engineer' | Pregunta: 'me armas un roadmap paso a paso'\n"
        f"  -> route: 'web_grounding', is_ephemeral: false, needs_memory_rag: false, reason: 'Roadmap de una disciplina que evoluciona rápidamente y requiere estado del arte'\n"
        f"- Contexto: '' | Pregunta: '¿Cuáles son los mejores frameworks web en 2025/2026?'\n"
        f"  -> route: 'web_grounding', is_ephemeral: false, needs_memory_rag: false, reason: 'Tendencias y estado del arte tecnológico actual'\n"
        f"- Contexto: '' | Pregunta: '¿Recuerdas qué receta o nota te pedí guardar la semana pasada?'\n"
        f"  -> route: 'rag_memory', is_ephemeral: false, needs_memory_rag: true, reason: 'Petición explícita de buscar en el archivo histórico'\n"
        f"- Contexto: '' | Pregunta: 'Escribe un correo formal para solicitar vacaciones a mi jefe'\n"
        f"  -> route: 'general', is_ephemeral: false, needs_memory_rag: false, reason: 'Redacción creativa y formal sin necesidad de datos en vivo'\n"
        f"- Contexto: 'Receta de pollo' | Pregunta: '¿cuánto tiempo lo dejo al horno?'\n"
        f"  -> route: 'general', is_ephemeral: false, needs_memory_rag: false, reason: 'Instrucción culinaria estándar basada en el sujeto activo'\n\n"
        f"Contexto actual de la conversación:\n"
        f"- Sujeto activo: '{active_subject}'\n"
        f"- Contexto inmediato: '{recent_context}'\n\n"
        f"Consulta a clasificar: '{question}'"
    )

    try:
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RouteDecision,
            temperature=0.0,
        )
        res, _ = call_with_fallback(client, router_prompt, config)
        if res.parsed:
            return res.parsed
    except Exception as exc:
        logger.warning(
            "semantic_router_fallback error_type=%s",
            type(exc).__name__,
        )

    # Heurística de fallback en caso de error de red sin hardcodeo temático estricto
    q_low = question.lower()
    if any(k in q_low for k in ("clima", "temperatura", "tiempo en", "pronostico")):
        return RouteDecision(route="weather", is_ephemeral=True, needs_memory_rag=False, reason="Fallback clima", confidence=0.8)
    if any(k in q_low for k in ("recuerdas", "mi historial", "que te dije", "guardaste", "mis notas")):
        return RouteDecision(route="rag_memory", is_ephemeral=False, needs_memory_rag=True, reason="Fallback memoria", confidence=0.8)
    if any(k in q_low for k in ("hoy", "ayer", "actual", "roadmap", "tendencia", "noticia", "ultimo", "novedad")):
        return RouteDecision(route="web_grounding", is_ephemeral=False, needs_memory_rag=False, reason="Fallback actualidad", confidence=0.8)

    return RouteDecision(route="general", is_ephemeral=False, needs_memory_rag=False, reason="Fallback general", confidence=0.7)


def execute_llm_call(
    question: str,
    file_path: str | None = None,
    use_grounding: bool = False,
    use_weather: bool = False,
    auto_route: bool = True,
    context_prefix: str = "",
    rag_tokens: dict | None = None,
    route_decision: RouteDecision | None = None,
    client: genai.Client | None = None,
) -> tuple[LLMResponse, CallMetrics]:
    """
    Ejecuta la llamada a Gemini soportando:
      - Memoria compacta de sesión (Working Memory)
      - Auto-routing inteligente de la intención del usuario
      - Salidas estructuradas con Pydantic (LLMResponse)
      - Google Search Grounding en tiempo real (--grounding / -g)
      - Function calling para clima en tiempo real (--clima / -c)
      - Entradas multimodales (imágenes, audios, documentos)
    """
    client = client or genai.Client(api_key=GEMINI_API_KEY)

    routed_as = "general"

    # Si se pasó una decisión previa del router, la respetamos
    if route_decision:
        routed_as = route_decision.route
        if route_decision.route == "weather":
            use_weather = True
        elif route_decision.route == "web_grounding":
            use_grounding = True
    elif not use_grounding and not use_weather and auto_route:
        decision = determine_route(client, question)
        routed_as = decision.route
        if decision.route == "weather":
            use_weather = True
        elif decision.route == "web_grounding":
            use_grounding = True
    elif use_weather:
        routed_as = "weather (manual)"
    elif use_grounding:
        routed_as = "web_grounding (manual)"

    current_date_str = datetime.now().strftime("%Y-%m-%d (%A)")

    prompt = (
        f"[Fecha actual del sistema: {current_date_str}]\n\n"
        f"Eres un asistente analítico y riguroso. Responde a la pregunta del usuario con información precisa, "
        f"actualizada y situada cronológicamente en la fecha actual ({current_date_str}). "
        f"Evita anacronismos o tratar el año presente como si fuera un año anterior (2024, 2025, etc.):\n\n"
        f"{context_prefix}"
        f"Pregunta del usuario: {question}"
    )
    contents = prepare_multimodal_content(prompt, file_path)

    grounding_sources: list[str] = []

    # Configuración según el modo solicitado:
    # 1. Modo Clima con Function Calling
    if use_weather:
        weather_config = genai_types.GenerateContentConfig(
            tools=[get_current_weather],
            temperature=0.2,
        )
        tool_response, model_used = call_with_fallback(client, contents, weather_config)
        answer_text = tool_response.text or "Consulta de clima completada."
        raw_uso = tool_response.usage_metadata

        validated = LLMResponse(
            answer=answer_text,
            confidence=0.99,
            key_points=["Datos meteorológicos obtenidos en tiempo real vía Open-Meteo API"],
            category="clima",
            sources_suggested=["Open-Meteo Weather API", "Geocoding API"],
        )

    # 2. Modo Grounding Web con Google Search
    elif use_grounding:
        grounding_config = genai_types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            temperature=0.2,
        )
        web_response, model_used = call_with_fallback(client, contents, grounding_config)
        answer_text = web_response.text or ""
        raw_uso = web_response.usage_metadata

        # Extraer enlaces y títulos de grounding reales
        if web_response.candidates:
            c = web_response.candidates[0]
            gm = getattr(c, "grounding_metadata", None)
            if gm:
                chunks = getattr(gm, "grounding_chunks", []) or []
                for chunk in chunks:
                    web_info = getattr(chunk, "web", None)
                    if web_info:
                        title = getattr(web_info, "title", "Fuente Web")
                        uri = getattr(web_info, "uri", "")
                        grounding_sources.append(f"{title}: {uri}")

        validated = LLMResponse(
            answer=answer_text,
            confidence=0.95,
            key_points=["Información verificada en tiempo real mediante Google Search Grounding"],
            category="actualidad",
            sources_suggested=grounding_sources[:5],
        )

    # 3. Modo Estándar con Structured Outputs Nativo (Pydantic)
    else:
        structured_config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LLMResponse,
            temperature=0.3,
            max_output_tokens=4096,
            thinking_config=genai_types.ThinkingConfig(
                thinking_budget=512,
            ),
        )
        response, model_used = call_with_fallback(client, contents, structured_config)
        raw_uso = response.usage_metadata
        validated = response.parsed
        if validated is None:
            raw_text = response.text.strip()
            validated = LLMResponse.model_validate_json(raw_text)

    # Cálculo exacto de tokens y costes (incluyendo overhead de Query Engineering y Reranking)
    tokens_entrada = getattr(raw_uso, "prompt_token_count", 0) or 0
    tokens_salida = getattr(raw_uso, "candidates_token_count", 0) or 0
    tokens_thinking = getattr(raw_uso, "thoughts_token_count", 0) or 0

    r_tok = rag_tokens or {}
    t_qe_in = r_tok.get("query_eng_in", 0)
    t_qe_out = r_tok.get("query_eng_out", 0)
    t_rr_in = r_tok.get("rerank_in", 0)
    t_rr_out = r_tok.get("rerank_out", 0)

    tokens_totales = (
        (getattr(raw_uso, "total_token_count", 0) or (tokens_entrada + tokens_salida + tokens_thinking))
        + t_qe_in + t_qe_out + t_rr_in + t_rr_out
    )

    tarifas = get_tarifas(model_used)
    tarifa_in = tarifas["input"]
    tarifa_out = tarifas["output"]
    tarifa_th = tarifas.get("output_thinking", 0.0)

    costo_entrada = (tokens_entrada / 1_000_000) * tarifa_in
    costo_salida = (tokens_salida / 1_000_000) * tarifa_out
    costo_thinking = (tokens_thinking / 1_000_000) * tarifa_th
    costo_rag_overhead = (
        ((t_qe_in + t_rr_in) / 1_000_000) * tarifa_in +
        ((t_qe_out + t_rr_out) / 1_000_000) * tarifa_out
    )
    costo_total = costo_entrada + costo_salida + costo_thinking + costo_rag_overhead

    metrics = CallMetrics(
        timestamp=datetime.now().isoformat(),
        model=model_used,
        question=question,
        file_attached=str(file_path) if file_path else None,
        routed_as=routed_as,
        grounding_used=use_grounding or use_weather,
        grounding_sources=grounding_sources,
        tokens_entrada=tokens_entrada,
        tokens_salida=tokens_salida,
        tokens_thinking=tokens_thinking,
        tokens_query_eng_in=t_qe_in,
        tokens_query_eng_out=t_qe_out,
        tokens_rerank_in=t_rr_in,
        tokens_rerank_out=t_rr_out,
        tokens_totales=tokens_totales,
        tarifa_input_por_millon=tarifa_in,
        tarifa_output_por_millon=tarifa_out,
        tarifa_thinking_por_millon=tarifa_th,
        costo_entrada_usd=round(costo_entrada, 8),
        costo_salida_usd=round(costo_salida, 8),
        costo_thinking_usd=round(costo_thinking, 8),
        costo_rag_overhead_usd=round(costo_rag_overhead, 8),
        costo_total_usd=round(costo_total, 8),
        response=validated.model_dump(),
    )

    return validated, metrics

