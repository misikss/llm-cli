"""Genera el storyboard GIF del dashboard para el README."""

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "demo.gif"
SIZE = (1200, 720)
BG = "#0a0f0c"
PANEL = "#111914"
PANEL_2 = "#0c130f"
LINE = "#29362d"
TEXT = "#d9e9dc"
MUTED = "#7b9381"
FAINT = "#526557"
LIME = "#d4ff70"
MINT = "#b7f3c5"
ORANGE = "#ffb178"
BLUE = "#8dc7ff"


def font(size: int, mono: bool = False):
    candidates = (
        ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"]
        if mono
        else ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F10 = font(10, True)
F12 = font(12, True)
F14 = font(14)
F16 = font(16)
F18 = font(18)
F26 = font(26)
F36 = font(36)


def label(draw, xy, text, fill=MUTED, size=10):
    draw.text(xy, text.upper(), font=font(size, True), fill=fill)


def panel(draw, box, title=None, accent=False):
    draw.rounded_rectangle(box, radius=5, fill=PANEL, outline=LIME if accent else LINE, width=2 if accent else 1)
    if title:
        label(draw, (box[0] + 18, box[1] + 16), title, LIME if accent else MUTED, 10)


def base_frame(stage, focus):
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, SIZE[0], 58), fill="#0d1510", outline=LINE)
    draw.text((28, 20), "◒  LLM CLI / CONTROL ROOM", font=F12, fill=TEXT)
    draw.ellipse((890, 25, 899, 34), fill=LIME)
    draw.text((908, 21), "LOCAL API / READY", font=F10, fill=MUTED)
    draw.text((1050, 21), "127.0.0.1:8000", font=F10, fill=FAINT)
    draw.text((28, 84), "MULTIMODAL PIPELINE / OBSERVABILITY LAB", font=F10, fill=LIME)
    draw.text((28, 112), "See the pipeline,", font=F36, fill=TEXT)
    draw.text((28, 154), "not just the answer.", font=F36, fill=MINT)
    draw.text((28, 204), "DEMO / LOCAL OBSERVABILITY SURFACE", font=F10, fill=FAINT)
    draw.text((1040, 88), f"{stage:02d} / 05", font=F12, fill=ORANGE)
    draw.text((1040, 110), focus, font=F10, fill=MUTED)
    return image, draw


def add_sidebar(draw, accent_section):
    panel(draw, (28, 250, 350, 680), "01 / INPUT", accent=accent_section == "input")
    draw.text((48, 292), "Pregunta", font=F12, fill=TEXT)
    draw.rounded_rectangle((48, 320, 330, 420), radius=3, fill=PANEL_2, outline=LINE)
    question = "analiza este documento y dime qué debería hacer"
    for index, line in enumerate(textwrap.wrap(question, 31)):
        draw.text((62, 340 + index * 24), line, font=F14, fill=TEXT)
    draw.text((48, 452), "Archivo adjunto", font=F12, fill=TEXT)
    draw.rounded_rectangle((48, 480, 330, 545), radius=3, fill=PANEL_2, outline=ORANGE if accent_section == "input" else LINE)
    draw.text((68, 496), "hoja_ventas.xlsx", font=F12, fill=MINT)
    draw.text((68, 520), "0.01 MB · XLSX", font=F10, fill=FAINT)
    draw.text((48, 579), "Context scope", font=F12, fill=TEXT)
    draw.text((48, 610), "USR   local-user", font=F10, fill=MUTED)
    draw.text((48, 632), "CONV  conv_8f2a", font=F10, fill=MUTED)
    draw.text((48, 654), "SESS  sess_61cd", font=F10, fill=MUTED)


def frame_input():
    image, draw = base_frame(1, "pregunta + archivo")
    add_sidebar(draw, "input")
    panel(draw, (398, 250, 1172, 680), "02 / OUTPUT")
    draw.text((430, 370), "Escribe una consulta, adjunta un archivo", font=F18, fill=TEXT)
    draw.text((430, 406), "y sigue cada decisión del pipeline.", font=F18, fill=MINT)
    return image


def frame_plan():
    image, draw = base_frame(2, "planner")
    add_sidebar(draw, "")
    panel(draw, (398, 250, 1172, 680), "03 / TRACE", accent=True)
    label(draw, (430, 294), "PLANIFICADOR", LIME, 10)
    steps = [("route", "general", MINT), ("document_analysis", "multimodal_direct", ORANGE), ("working_memory", "ready", BLUE)]
    for index, (name, status, color) in enumerate(steps):
        y = 345 + index * 78
        draw.line((450, y + 12, 1110, y + 12), fill=LINE, width=1)
        draw.ellipse((438, y + 3, 462, y + 27), fill=color)
        draw.text((484, y), name, font=F16, fill=TEXT)
        draw.text((900, y + 2), status, font=F12, fill=color)
    draw.text((430, 594), "El documento es pequeño: se mantiene el análisis multimodal directo.", font=F14, fill=MUTED)
    return image


def frame_generation():
    image, draw = base_frame(3, "generation")
    add_sidebar(draw, "")
    panel(draw, (398, 250, 1172, 680), "02 / OUTPUT", accent=True)
    label(draw, (430, 294), "ANSWER", ORANGE, 10)
    draw.text((430, 340), "La venta más alta fue V-003 por S/ 2,392.", font=F18, fill=TEXT)
    draw.text((430, 382), "El panel conserva respuesta, puntos clave y fuentes.", font=F16, fill=MINT)
    draw.line((430, 438, 1135, 438), fill=LINE, width=1)
    label(draw, (430, 468), "KEY POINTS", LIME, 10)
    for index, text in enumerate(["total de ventas: S/ 8,768", "promedio por transacción: S/ 1,096", "trace_id disponible para debugging"]):
        draw.text((440, 506 + index * 30), f"↳  {text}", font=F14, fill=MUTED)
    return image


def frame_trace():
    image, draw = base_frame(4, "stage telemetry")
    add_sidebar(draw, "")
    panel(draw, (398, 250, 1172, 680), "03 / TRACE", accent=True)
    headers = [(430, "Stage"), (670, "Status"), (795, "Latency"), (940, "Tokens in/out")]
    for x, text in headers:
        draw.text((x, 300), text, font=F10, fill=FAINT)
    rows = [("route", "ok", "3.65 s", "725 / 53"), ("document_analysis", "ok", "0.01 s", "0 / 0"), ("generation", "ok", "4.66 s", "125 / 360"), ("memory_update", "ok", "2.90 s", "128 / 81")]
    for index, row in enumerate(rows):
        y = 350 + index * 58
        draw.line((430, y + 30, 1135, y + 30), fill=LINE, width=1)
        draw.text((430, y), row[0], font=F12, fill=TEXT)
        draw.text((670, y), row[1], font=F12, fill=LIME)
        draw.text((795, y), row[2], font=F12, fill=MUTED)
        draw.text((940, y), row[3], font=F12, fill=MINT)
    draw.text((430, 605), "tokens · coste · latencia · fallbacks · errores", font=F14, fill=ORANGE)
    return image


def frame_scope():
    image, draw = base_frame(5, "context isolation")
    add_sidebar(draw, "input")
    panel(draw, (398, 250, 1172, 680), "CONTEXT SCOPE", accent=True)
    label(draw, (430, 294), "USUARIOS LOCALES", LIME, 10)
    draw.text((430, 344), "USR", font=F12, fill=FAINT)
    draw.rounded_rectangle((490, 330, 850, 378), radius=3, fill=PANEL_2, outline=LIME)
    draw.text((510, 346), "local-user   ▾", font=F14, fill=TEXT)
    draw.text((430, 420), "+ Nuevo usuario", font=F16, fill=ORANGE)
    draw.text((430, 492), "Cada identidad conserva su propia conversación,", font=F18, fill=TEXT)
    draw.text((430, 530), "memoria y trazabilidad.", font=F18, fill=MINT)
    draw.text((430, 620), "El archivo se elimina al terminar la solicitud.", font=F12, fill=MUTED)
    return image


def main():
    frames = [frame_input(), frame_plan(), frame_generation(), frame_trace(), frame_scope(), frame_input()]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=5000, loop=0, optimize=False)
    print(f"created {OUTPUT}")


if __name__ == "__main__":
    main()
