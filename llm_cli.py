# -*- coding: utf-8 -*-
"""
llm_cli.py — Punto de entrada CLI modular con Gemini, Pydantic, Grounding y Clima.
"""

import argparse
import io
import logging
import sys

# Silenciar telemetría interna de ChromaDB para una consola limpia
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# Forzar UTF-8 en stdout para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config import GEMINI_API_KEY, GEMINI_MODEL
from client import execute_llm_call, determine_route
from models import RouteDecision
from memory import HybridMemory
from rag import rewrite_query
from google import genai
from display import (
    console,
    display_response,
    save_to_log,
    show_history,
    show_precios,
    show_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI para preguntas a Gemini con salida estructurada, Multimodal, Grounding y Clima en tiempo real.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python llm_cli.py                                      # Modo interactivo continuo
  python llm_cli.py -q "Que es un decorador en Python?"  # Pregunta directa estructurada
  python llm_cli.py -q "Describe la imagen" -f logo.png  # Multimodal (imagen, audio, pdf)
  python llm_cli.py -q "Ultimas noticias de IA" -g       # Google Search Grounding en tiempo real
  python llm_cli.py -q "Que clima hace en Tokio hoy?" -c # Clima en tiempo real (Tool Calling)
  python llm_cli.py --history                            # Ver historial
  python llm_cli.py --stats                              # Resumen de costos
  python llm_cli.py --modelos                            # Tarifas de modelos
        """,
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Pregunta a enviar directamente (sin modo interactivo)"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Ruta a una imagen, audio o documento (JPG, PNG, MP3, WAV, PDF)"
    )
    parser.add_argument(
        "-g", "--grounding",
        action="store_true",
        help="Activa Google Search Grounding para información web en tiempo real"
    )
    parser.add_argument(
        "-c", "--clima",
        action="store_true",
        help="Activa la herramienta de consulta de clima en tiempo real para cualquier ciudad"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Muestra el historial de llamadas anteriores"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Número de entradas a mostrar en el historial (default: 10)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Muestra el costo total acumulado"
    )
    parser.add_argument(
        "--modelos",
        action="store_true",
        help="Muestra la tabla de precios por modelo"
    )

    args = parser.parse_args()

    # Comandos de solo lectura
    if args.history:
        show_history(args.limit)
        return

    if args.stats:
        show_stats()
        return

    if args.modelos:
        show_precios()
        return

    # Validar API Key
    if not GEMINI_API_KEY:
        console.print("[red]Error: GEMINI_API_KEY no encontrada.[/red]")
        console.print("Configura tu clave en el archivo .env")
        sys.exit(1)

    # Si se pasó -q por argumento, se procesa con enrutamiento inteligente y termina
    if args.question:
        try:
            genai_client = genai.Client(api_key=GEMINI_API_KEY)
            memory = HybridMemory(db_path="chroma_db")

            # 1. Determinar ruta de una sola vez (o respetar flags manuales)
            if args.clima:
                route_decision = RouteDecision(
                    route="weather",
                    is_ephemeral=True,
                    needs_memory_rag=False,
                    reason="Flag manual --clima activado"
                )
            elif args.grounding:
                route_decision = RouteDecision(
                    route="web_grounding",
                    is_ephemeral=False,
                    needs_memory_rag=False,
                    reason="Flag manual --grounding activado"
                )
            else:
                route_decision = determine_route(client=genai_client, question=args.question.strip())

            # 2. Gateo del RAG: Solo si needs_memory_rag es True se corre QueryEng + Hybrid + Rerank
            t_qe_in = t_qe_out = t_rr_in = t_rr_out = 0
            if route_decision.needs_memory_rag:
                rewritten, t_qe_in, t_qe_out = rewrite_query(
                    client=genai_client,
                    question=args.question.strip(),
                )
                context_prefix, t_rr_in, t_rr_out = memory.get_context_prompt(
                    genai_client,
                    query=rewritten.rewritten_query,
                    keywords=rewritten.search_keywords,
                )
            else:
                context_prefix = ""

            rag_tokens = {
                "query_eng_in": t_qe_in,
                "query_eng_out": t_qe_out,
                "rerank_in": t_rr_in,
                "rerank_out": t_rr_out,
            }

            validated, metrics = execute_llm_call(
                question=args.question.strip(),
                file_path=args.file,
                use_grounding=args.grounding,
                use_weather=args.clima,
                route_decision=route_decision,
                context_prefix=context_prefix,
                rag_tokens=rag_tokens,
            )
            save_to_log(metrics)
            display_response(validated, metrics)

            cat = "actualidad_efimera" if route_decision.is_ephemeral else validated.category
            memory.update(genai_client, args.question.strip(), validated.answer, category=cat)
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelado por el usuario.[/dim]")
        except Exception as e:
            console.print(f"\n[red]Error inesperado: {e}[/red]")
            sys.exit(1)
        return

    # Modo interactivo continuo con Memoria Híbrida y Enrutamiento Inteligente (Zero-Waste RAG)
    console.print("\n[bold cyan]LLM CLI -- Asistente con Memoria Híbrida y Semantic Routing (Zero-Waste RAG)[/bold cyan]")
    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d (%A)")
    console.print(f"[dim]Modelo: {GEMINI_MODEL} | Fecha actual: {today_str} | '/reset' para reiniciar RAM, 'salir' para terminar[/dim]\n")

    memory = HybridMemory(db_path="chroma_db")
    genai_client = genai.Client(api_key=GEMINI_API_KEY)

    while True:
        try:
            if memory.active_subject:
                console.print(f"[dim magenta]📍 Tema activo en memoria: '{memory.active_subject}'[/dim magenta]")

            question = console.input("[bold yellow]Tu pregunta: [/bold yellow]").strip()

            if not question:
                continue

            if question.lower() in ("salir", "exit", "quit"):
                console.print("[dim]Hasta luego.[/dim]")
                break

            if question.lower() in ("/reset", "/clear", "/nuevo"):
                memory.reset()
                console.print("[green]✓ Memoria de trabajo (RAM) reiniciada. Los recuerdos históricos se conservan en ChromaDB.[/green]\n")
                continue

            # Opción rápida de archivo adjunto
            file_input = console.input("[dim cyan]Archivo adjunto (opcional, Enter para omitir): [/dim cyan]").strip()
            file_input = file_input.strip('"\'') if file_input else None

            # 1. Contexto de Working Memory en RAM (costo $0)
            wm_context = memory.get_working_memory_context()

            # 2. ÚNICA Clasificación de Intención (Semantic Router con Few-Shot)
            route_decision = determine_route(
                client=genai_client,
                question=question,
                active_subject=memory.active_subject,
                recent_context="; ".join(memory.key_facts),
            )

            # 3. Gateo del RAG: Solo si needs_memory_rag es True se corre QueryEng + Hybrid + Rerank
            t_qe_in = t_qe_out = t_rr_in = t_rr_out = 0
            if route_decision.needs_memory_rag:
                rewritten, t_qe_in, t_qe_out = rewrite_query(
                    client=genai_client,
                    question=question,
                    active_subject=memory.active_subject,
                    recent_context="; ".join(memory.key_facts),
                )
                context_prefix, t_rr_in, t_rr_out = memory.get_context_prompt(
                    genai_client,
                    query=rewritten.rewritten_query,
                    keywords=rewritten.search_keywords,
                )
            else:
                # Usamos el contexto inmediato de la RAM a costo $0
                context_prefix = wm_context

            rag_tokens = {
                "query_eng_in": t_qe_in,
                "query_eng_out": t_qe_out,
                "rerank_in": t_rr_in,
                "rerank_out": t_rr_out,
            }

            # 4. Ejecución del llamado con la ruta ya decidida (sin volver a clasificar)
            validated, metrics = execute_llm_call(
                question=question,
                file_path=file_input,
                route_decision=route_decision,
                context_prefix=context_prefix,
                rag_tokens=rag_tokens,
            )
            save_to_log(metrics)
            display_response(validated, metrics)

            # 5. Actualizamos Working Memory (RAM) y persistimos en ChromaDB si no es efímero
            cat = "actualidad_efimera" if route_decision.is_ephemeral else validated.category
            memory.update(genai_client, question, validated.answer, category=cat)

        except KeyboardInterrupt:
            console.print("\n[dim]Sesión finalizada por el usuario.[/dim]")
            break
        except Exception as e:
            console.print(f"\n[red]Error inesperado: {e}[/red]")


if __name__ == "__main__":
    main()
