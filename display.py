"""
display.py — Renderizado de consola con Rich y gestión de historial/estadísticas.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import ERROR_LOG_FILE, LOG_DIR, LOG_FILE, PRECIOS_MODELO, GEMINI_MODEL
from errors import LLMCLIError
from models import LLMResponse, CallMetrics

console = Console(force_terminal=True, highlight=False)


def save_to_log(metrics: CallMetrics) -> None:
    """Guarda las métricas en logs/history.jsonl."""
    LOG_DIR.mkdir(exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(metrics.model_dump_json() + "\n")


def save_error_to_log(error: Exception) -> None:
    """Persiste fallos de solicitudes sin secretos ni stack traces completos."""
    LOG_DIR.mkdir(exist_ok=True)
    if isinstance(error, LLMCLIError):
        error_data = error.to_dict()
    else:
        error_data = {
            "type": type(error).__name__,
            "code": "unexpected_error",
            "message": str(error),
            "retryable": False,
        }

    payload = {
        "event_type": "request_error",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "trace_id": getattr(error, "trace_id", None),
        "question": getattr(error, "question", None),
        "user_id": getattr(error, "user_id", None),
        "session_id": getattr(error, "session_id", None),
        "conversation_id": getattr(error, "conversation_id", None),
        "error": error_data,
    }
    with ERROR_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def display_response(validated: LLMResponse, metrics: CallMetrics) -> None:
    """Imprime la respuesta estructurada y las métricas en terminal."""
    console.print()
    console.print(Panel(
        f"[bold white]{validated.answer}[/bold white]",
        title=f"[bold cyan]>> Respuesta de {metrics.model}[/bold cyan]",
        subtitle=f"[dim]Categoria: {validated.category} | Confianza: {validated.confidence:.0%}[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))

    if validated.key_points:
        console.print("\n[bold yellow]Puntos clave:[/bold yellow]")
        for i, point in enumerate(validated.key_points, 1):
            console.print(f"  [cyan]{i}.[/cyan] {point}")

    if validated.sources_suggested:
        console.print("\n[bold blue]Fuentes sugeridas / Grounding:[/bold blue]")
        for src in validated.sources_suggested:
            console.print(f"  - {src}")

    console.print()
    table = Table(title=f"Metricas — {metrics.model}", border_style="dim")
    table.add_column("Metrica", style="bold", min_width=28)
    table.add_column("Valor", justify="right", min_width=18)

    if metrics.file_attached:
        table.add_row("Archivo adjunto", f"[cyan]{Path(metrics.file_attached).name}[/cyan]")
    table.add_row("Enrutado como (Auto-route)", f"[bold magenta]{metrics.routed_as}[/bold magenta]")
    table.add_row("Fuente del routing", metrics.route_source)
    table.add_row("Trace ID", metrics.trace_id)
    table.add_row("User ID", metrics.user_id)
    table.add_row("Session ID", metrics.session_id)
    table.add_row("Conversation ID", metrics.conversation_id)
    table.add_row("Latencia total", f"{metrics.total_latency_ms:.2f} ms")
    if metrics.fallbacks:
        table.add_row("Fallbacks", str(len(metrics.fallbacks)))
    if metrics.errors:
        table.add_row("Errores registrados", str(len(metrics.errors)))
    if metrics.grounding_used:
        table.add_row("Modo Grounding / Tool", "[bold green]Activo (En tiempo real)[/bold green]")

    table.add_row("Tokens entrada (prompt)", f"{metrics.tokens_entrada:,}")
    table.add_row("Tokens salida (respuesta)", f"{metrics.tokens_salida:,}")
    if metrics.tokens_router_in + metrics.tokens_router_out > 0:
        table.add_row(
            "Tokens semantic router",
            f"{metrics.tokens_router_in + metrics.tokens_router_out:,}",
        )
    if metrics.tokens_memory_update_in + metrics.tokens_memory_update_out > 0:
        table.add_row(
            "Tokens memory update",
            f"{metrics.tokens_memory_update_in + metrics.tokens_memory_update_out:,}",
        )
    if metrics.embedding_calls:
        embedding_tokens = metrics.tokens_embedding_in + metrics.tokens_embedding_out
        value = f"{embedding_tokens:,}" if embedding_tokens else "usage no expuesto"
        table.add_row("Embeddings / llamadas", f"{metrics.embedding_calls} / {value}")
    if metrics.tokens_thinking > 0:
        table.add_row("Tokens thinking (razonamiento)", f"{metrics.tokens_thinking:,}")
    if (metrics.tokens_query_eng_in + metrics.tokens_rerank_in) > 0:
        table.add_row("Tokens overhead (QueryEng + Rerank)", f"{metrics.tokens_query_eng_in + metrics.tokens_query_eng_out + metrics.tokens_rerank_in + metrics.tokens_rerank_out:,}")
    table.add_row("Tokens totales", f"[bold]{metrics.tokens_totales:,}[/bold]")
    table.add_row("Cobertura de tokens", metrics.token_usage_scope)

    table.add_section()
    table.add_row("Tarifa entrada (USD/1M)", f"${metrics.tarifa_input_por_millon:.4f}")
    table.add_row("Tarifa salida  (USD/1M)", f"${metrics.tarifa_output_por_millon:.4f}")
    if metrics.tarifa_thinking_por_millon > 0:
        table.add_row("Tarifa thinking (USD/1M)", f"${metrics.tarifa_thinking_por_millon:.4f}")

    table.add_section()
    table.add_row("Costo entrada", f"${metrics.costo_entrada_usd:.8f}")
    table.add_row("Costo salida", f"${metrics.costo_salida_usd:.8f}")
    if metrics.tokens_thinking > 0:
        table.add_row("Costo thinking", f"${metrics.costo_thinking_usd:.8f}")
    if metrics.costo_rag_overhead_usd > 0:
        table.add_row("Costo overhead RAG", f"${metrics.costo_rag_overhead_usd:.8f}")
    table.add_row("Costo total", f"[bold green]${metrics.costo_total_usd:.8f} USD[/bold green]")

    console.print(table)
    console.print(f"[dim]Log guardado en: {LOG_FILE.resolve()}[/dim]\n")


def show_precios() -> None:
    """Muestra la tabla de precios por modelo."""
    table = Table(title="Tabla de precios Gemini (USD / 1M tokens)", border_style="cyan")
    table.add_column("Modelo", style="bold", min_width=24)
    table.add_column("Input (USD/1M)", justify="right", min_width=16)
    table.add_column("Output (USD/1M)", justify="right", min_width=16)
    table.add_column("Thinking (USD/1M)", justify="right", min_width=18)
    table.add_column("Activo", justify="center", min_width=8)

    for nombre, tarifas in PRECIOS_MODELO.items():
        es_activo = nombre in GEMINI_MODEL or GEMINI_MODEL in nombre
        marca = "[bold green]<-- activo[/bold green]" if es_activo else ""
        estilo = "bold green" if es_activo else ""
        thinking = tarifas.get("output_thinking", 0.0)
        table.add_row(
            f"[{estilo}]{nombre}[/{estilo}]" if estilo else nombre,
            f"${tarifas['input']:.4f}",
            f"${tarifas['output']:.4f}",
            f"${thinking:.4f}" if thinking > 0 else "[dim]-[/dim]",
            marca,
        )

    console.print()
    console.print(table)
    console.print("[dim]Fuente: https://ai.google.dev/pricing[/dim]")
    console.print(f"[dim]Modelo activo: {GEMINI_MODEL}[/dim]\n")


def show_history(limit: int = 10) -> None:
    """Muestra las últimas N llamadas del historial."""
    if not LOG_FILE.exists():
        console.print("[yellow]No hay historial todavía.[/yellow]")
        return

    lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    recent = entries[-limit:]

    console.print(f"\n[bold]Ultimas {len(recent)} llamadas:[/bold]\n")
    for entry in recent:
        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        q = entry.get("question", "")[:60] + ("..." if len(entry.get("question", "")) > 60 else "")
        cost = entry.get("costo_total_usd", entry.get("cost_total_usd", 0.0))
        tokens = entry.get("tokens_totales", entry.get("total_tokens", 0))
        modelo = entry.get("model", "desconocido")
        grounding = "[green][Web][/green] " if entry.get("grounding_used") else ""
        trace_id = entry.get("trace_id", "")
        latency = entry.get("total_latency_ms")
        console.print(f"  [dim]{ts}[/dim]  {grounding}[cyan]{q}[/cyan]")
        details = f"[{modelo}] Tokens: {tokens:,} | Costo: [green]${cost:.8f} USD[/green]"
        if latency is not None:
            details += f" | Latencia: {latency:.2f} ms"
        if trace_id:
            details += f" | Trace: {trace_id}"
        console.print(f"           {details}\n")


def show_stats() -> None:
    """Muestra el costo total acumulado y estadísticas."""
    if not LOG_FILE.exists():
        console.print("[yellow]No hay historial todavía.[/yellow]")
        return

    lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    entries = [entry for entry in entries if entry.get("event_type", "llm_call") == "llm_call"]
    total_calls = len(entries)
    total_tokens = 0
    total_cost = 0.0

    for entry in entries:
        total_tokens += entry.get("tokens_totales", entry.get("total_tokens", 0))
        total_cost += entry.get("costo_total_usd", entry.get("cost_total_usd", 0.0))

    table = Table(title="Resumen de uso acumulado", border_style="green")
    table.add_column("Estadistica", style="bold")
    table.add_column("Valor", justify="right")

    table.add_row("Total de llamadas", str(total_calls))
    table.add_row("Tokens totales usados", f"{total_tokens:,}")
    table.add_row("Costo total acumulado", f"[bold green]${total_cost:.4f} USD[/bold green]")
    average_cost = total_cost / total_calls if total_calls else 0.0
    table.add_row("Costo promedio/llamada", f"${average_cost:.6f} USD")

    console.print()
    console.print(table)

