# -*- coding: utf-8 -*-
"""Interfaz CLI del asistente; la lógica de aplicación vive en service.py."""

from __future__ import annotations

import argparse
import io
import logging
import sys


# Silenciar telemetría interna de ChromaDB para una consola limpia.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# Forzar UTF-8 en stdout para Windows.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config import GEMINI_API_KEY, GEMINI_MODEL  # noqa: E402
from display import (  # noqa: E402
    console,
    display_response,
    save_error_to_log,
    save_to_log,
    show_history,
    show_precios,
    show_stats,
)
from errors import LLMCLIError  # noqa: E402
from service import AskRequest, AssistantService  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-q", "--question", type=str,
        help="Pregunta a enviar directamente (sin modo interactivo)",
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None,
        help="Ruta a imagen, audio o documento (JPG, PNG, MP3, WAV, PDF, XLSX, CSV)",
    )
    parser.add_argument(
        "-g", "--grounding", action="store_true",
        help="Activa Google Search Grounding para información web en tiempo real",
    )
    parser.add_argument(
        "-c", "--clima", action="store_true",
        help="Activa la herramienta de consulta de clima en tiempo real para cualquier ciudad",
    )
    parser.add_argument("--history", action="store_true", help="Muestra el historial de llamadas anteriores")
    parser.add_argument("--limit", type=int, default=10, help="Número de entradas a mostrar en el historial (default: 10)")
    parser.add_argument("--stats", action="store_true", help="Muestra el costo total acumulado")
    parser.add_argument("--modelos", action="store_true", help="Muestra la tabla de precios por modelo")
    parser.add_argument(
        "--user-id",
        default="local-user",
        help="Identificador del usuario propietario del contexto (default: local-user)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Identificador de ejecución; por defecto se genera uno por proceso",
    )
    parser.add_argument(
        "--conversation-id",
        default="default-conversation",
        help="Identificador del chat/conversación persistente",
    )


def persist_and_display(result) -> None:
    save_to_log(result.metrics)
    display_response(result.response, result.metrics)


def handle_request(service: AssistantService, request: AskRequest) -> None:
    result = service.ask(request)
    persist_and_display(result)


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    add_arguments(parser)
    args = parser.parse_args()

    if args.history:
        show_history(args.limit)
        return
    if args.stats:
        show_stats()
        return
    if args.modelos:
        show_precios()
        return

    if not GEMINI_API_KEY:
        console.print("[red]Error: GEMINI_API_KEY no encontrada.[/red]")
        console.print("Configura tu clave en el archivo .env")
        sys.exit(1)

    try:
        service = AssistantService(
            user_id=args.user_id,
            session_id=args.session_id,
            conversation_id=args.conversation_id,
        )
    except LLMCLIError as exc:
        save_error_to_log(exc)
        console.print(f"[red]{exc.message}[/red]")
        sys.exit(1)

    if args.question:
        try:
            handle_request(
                service,
                AskRequest(
                    question=args.question,
                    file_path=args.file,
                    use_grounding=args.grounding,
                    use_weather=args.clima,
                ),
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelado por el usuario.[/dim]")
        except LLMCLIError as exc:
            save_error_to_log(exc)
            console.print(f"\n[red]{exc.message}[/red]")
            if getattr(exc, "trace_id", None):
                console.print(f"[dim]Trace ID: {exc.trace_id}[/dim]")
            sys.exit(1)
        return

    console.print(
        "\n[bold cyan]LLM CLI -- Asistente con Memoria Híbrida y Semantic Routing (Zero-Waste RAG)[/bold cyan]"
    )
    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d (%A)")
    console.print(
        f"[dim]Modelo: {GEMINI_MODEL} | Fecha actual: {today_str} | '/reset' para reiniciar RAM, 'salir' para terminar[/dim]\n"
    )

    while True:
        try:
            if service.memory.active_subject:
                console.print(
                    f"[dim magenta]📍 Tema activo en memoria: '{service.memory.active_subject}'[/dim magenta]"
                )

            question = console.input("[bold yellow]Tu pregunta: [/bold yellow]").strip()
            if not question:
                continue
            if question.lower() in ("salir", "exit", "quit"):
                console.print("[dim]Hasta luego.[/dim]")
                break
            if question.lower() in ("/reset", "/clear", "/nuevo"):
                service.memory.reset()
                console.print(
                    "[green]✓ Memoria de trabajo (RAM) reiniciada. Los recuerdos históricos se conservan en ChromaDB.[/green]\n"
                )
                continue

            file_input = console.input(
                "[dim cyan]Archivo adjunto (opcional, Enter para omitir): [/dim cyan]"
            ).strip()
            file_input = file_input.strip('"\'') if file_input else None
            handle_request(
                service,
                AskRequest(question=question, file_path=file_input),
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Sesión finalizada por el usuario.[/dim]")
            break
        except LLMCLIError as exc:
            save_error_to_log(exc)
            console.print(f"\n[red]{exc.message}[/red]")
            if getattr(exc, "trace_id", None):
                console.print(f"[dim]Trace ID: {exc.trace_id}[/dim]")
        except Exception as exc:
            save_error_to_log(exc)
            console.print(f"\n[red]Error inesperado: {exc}[/red]")


if __name__ == "__main__":
    main()
