"""
MVP: Analizador de comentarios de Facebook
Flujo: URL de Facebook -> Apify (facebook-comments-scraper) -> Gemini -> JSON con dolores, objeciones y deseos.
"""

import logging
import os
from datetime import timedelta
from typing import List

from apify_client import ApifyClientAsync
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field, HttpUrl

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fb-analyzer")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_COMMENTS = int(os.getenv("MAX_COMMENTS", "150"))
APIFY_ACTOR_ID = "apify/facebook-comments-scraper"

app = FastAPI(
    title="Analizador de Comentarios de Facebook",
    description="Extrae comentarios con Apify y los analiza con Gemini.",
    version="1.0.0",
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
apify_client = ApifyClientAsync(APIFY_API_TOKEN) if APIFY_API_TOKEN else None


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    url: HttpUrl


class Insight(BaseModel):
    tema: str = Field(..., description="Resumen corto del dolor/objeción/deseo")
    frecuencia: str = Field(..., description="alta | media | baja")
    ejemplos: List[str] = Field(..., description="Citas textuales de comentarios")


class AnalysisResult(BaseModel):
    dolores: List[Insight]
    objeciones: List[Insight]
    deseos: List[Insight]
    resumen: str


class AnalyzeResponse(BaseModel):
    url: str
    total_comentarios: int
    modelo: str
    analisis: AnalysisResult


# ---------------------------------------------------------------------------
# Apify: extracción de comentarios
# ---------------------------------------------------------------------------
def _field(obj, attr: str, key: str):
    """Lee un campo tanto si Apify devuelve un dict como un objeto."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, attr, None)


async def fetch_comments(url: str) -> List[str]:
    if apify_client is None:
        raise HTTPException(status_code=500, detail="Falta la variable APIFY_API_TOKEN.")

    run_input = {
        "startUrls": [{"url": url}],
        "resultsLimit": MAX_COMMENTS,
        "includeNestedComments": False,
        "viewOption": "RANKED_UNFILTERED",
    }

    try:
        run = await apify_client.actor(APIFY_ACTOR_ID).call(
            run_input=run_input,
            wait_duration=timedelta(minutes=4),
        )
    except Exception as exc:
        logger.exception("Error llamando a Apify")
        raise HTTPException(status_code=502, detail=f"Error al ejecutar Apify: {exc}") from exc

    # Compatible con apify-client 1.x (dict) y 2.x/3.x (objeto Run)
    status = _field(run, "status", "status")
    status = getattr(status, "value", status)
    if not run or str(status) != "SUCCEEDED":
        raise HTTPException(
            status_code=502,
            detail=f"El scraper de Apify no terminó bien (estado: {status or 'desconocido'}).",
        )

    dataset_id = _field(run, "default_dataset_id", "defaultDatasetId")
    page = await apify_client.dataset(dataset_id).list_items()
    items = page.items or []

    comments = [
        item["text"].strip()
        for item in items
        if isinstance(item.get("text"), str) and item["text"].strip()
    ]
    logger.info("Comentarios extraídos: %d", len(comments))
    return comments


# ---------------------------------------------------------------------------
# Gemini: análisis de marketing con salida JSON estructurada
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Eres un estratega de marketing experto en investigación de mercado y copywriting de respuesta directa.
Analizas comentarios reales de clientes en redes sociales para extraer:
- DOLORES: problemas, frustraciones o miedos que expresan.
- OBJECIONES: dudas, desconfianzas o razones por las que no comprarían (precio, tiempo, credibilidad, etc.).
- DESEOS: resultados, aspiraciones o transformaciones que quieren lograr.

Reglas:
- Agrupa ideas repetidas en un solo tema y estima su frecuencia (alta, media, baja).
- Incluye 1 a 3 citas textuales cortas como ejemplos por tema, copiadas tal cual de los comentarios.
- No inventes nada que no esté en los comentarios. Si una categoría no aparece, devuélvela vacía.
- Escribe en el mismo idioma predominante de los comentarios.
- El resumen debe ser de 2 a 4 frases con la conclusión accionable para marketing."""


async def analyze_with_gemini(comments: List[str]) -> AnalysisResult:
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="Falta la variable GEMINI_API_KEY.")

    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(comments))
    user_message = f"Analiza estos {len(comments)} comentarios de Facebook:\n\n{numbered}"

    try:
        response = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=AnalysisResult,
                temperature=0.3,
            ),
        )
    except genai_errors.APIError as exc:
        logger.exception("Error llamando a Gemini")
        raise HTTPException(status_code=502, detail=f"Error en la API de Gemini: {exc}") from exc

    # El SDK ya convierte el JSON en el modelo Pydantic
    if isinstance(response.parsed, AnalysisResult):
        return response.parsed

    # Respaldo: validar el texto crudo por si parsed viene vacío
    try:
        return AnalysisResult.model_validate_json(response.text or "")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini no devolvió un JSON válido: {exc}") from exc


# ---------------------------------------------------------------------------
# Página web: formulario para pegar el enlace y ver los resultados
# ---------------------------------------------------------------------------
HOME_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analizador de comentarios</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600;700&family=Newsreader:ital,opsz@1,6..72&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #EEF1F6; --ink: #1B2A41; --muted: #5B6B82; --line: #D3DAE5; --paper: #FFFFFF;
    --dolor: #C2410C; --objecion: #9A6700; --deseo: #0F766E; --accion: #2B4C7E;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.55 "Instrument Sans", system-ui, -apple-system, sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 40px 20px 80px; }
  h1 { font-size: clamp(28px, 6vw, 40px); line-height: 1.1; margin: 0 0 10px; letter-spacing: -0.02em; }
  .intro { color: var(--muted); margin: 0 0 28px; max-width: 60ch; }
  form { display: flex; gap: 10px; flex-wrap: wrap; }
  input { flex: 1 1 320px; min-width: 0; padding: 14px 16px; font: inherit; color: var(--ink);
    background: var(--paper); border: 1.5px solid var(--line); border-radius: 10px; }
  input:focus { outline: 3px solid rgba(43,76,126,.25); border-color: var(--accion); }
  button { padding: 14px 22px; font: 600 16px "Instrument Sans", sans-serif; color: #fff;
    background: var(--accion); border: 0; border-radius: 10px; cursor: pointer; }
  button:disabled { opacity: .6; cursor: wait; }
  button:focus-visible { outline: 3px solid rgba(43,76,126,.4); outline-offset: 2px; }
  .status { margin: 18px 0 0; color: var(--muted); min-height: 1.5em; }
  .status.error { color: var(--dolor); font-weight: 600; }
  .summary { margin: 36px 0 8px; padding: 22px 24px; background: var(--paper);
    border-left: 5px solid var(--accion); border-radius: 4px 12px 12px 4px; }
  .summary p { margin: 0; font-size: 18px; }
  .meta { color: var(--muted); font-size: 14px; margin-top: 10px; }
  section { margin-top: 36px; }
  h2 { font-size: 22px; margin: 0 0 14px; display: flex; align-items: center; gap: 10px; }
  h2 .dot { width: 12px; height: 12px; border-radius: 50%; background: var(--c); }
  .count { font-weight: 400; color: var(--muted); font-size: 16px; }
  .item { background: var(--paper); border-radius: 12px; padding: 18px 20px; margin-bottom: 12px;
    border: 1px solid var(--line); }
  .item-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
  .item h3 { margin: 0; font-size: 17px; }
  .freq { font-size: 13px; font-weight: 600; color: var(--c); white-space: nowrap; }
  blockquote { margin: 12px 0 0; padding: 0 0 0 14px; border-left: 2px solid var(--c);
    font: italic 17px/1.5 "Newsreader", Georgia, serif; color: #33445C; }
  blockquote + blockquote { margin-top: 8px; }
  .empty { color: var(--muted); font-style: italic; margin: 0; }
  .spinner { display: inline-block; width: 14px; height: 14px; margin-right: 8px; vertical-align: -2px;
    border: 2px solid var(--line); border-top-color: var(--accion); border-radius: 50%;
    animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) { .spinner { animation: none; } }
</style>
</head>
<body>
<main>
  <h1>Lo que tus clientes dicen en los comentarios</h1>
  <p class="intro">Pega el enlace de una publicación pública de Facebook. Leemos sus comentarios y los ordenamos en dolores, objeciones y deseos, con citas reales.</p>

  <form id="form">
    <input id="url" type="url" required placeholder="https://www.facebook.com/..." aria-label="Enlace de la publicación de Facebook">
    <button id="btn" type="submit">Analizar comentarios</button>
  </form>
  <p id="status" class="status" role="status"></p>

  <div id="results"></div>
</main>

<script>
const form = document.getElementById("form");
const btn = document.getElementById("btn");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");

const CATS = [
  ["dolores", "Dolores", "var(--dolor)"],
  ["objeciones", "Objeciones", "var(--objecion)"],
  ["deseos", "Deseos", "var(--deseo)"],
];

function esc(t) {
  const d = document.createElement("div");
  d.textContent = t ?? "";
  return d.innerHTML;
}

function render(data) {
  const a = data.analisis;
  let html = `<div class="summary"><p>${esc(a.resumen)}</p>
    <div class="meta">${data.total_comentarios} comentarios analizados</div></div>`;
  for (const [key, label, color] of CATS) {
    const items = a[key] || [];
    html += `<section style="--c:${color}"><h2><span class="dot"></span>${label}
      <span class="count">(${items.length})</span></h2>`;
    if (!items.length) html += `<p class="empty">No aparecen en estos comentarios.</p>`;
    for (const it of items) {
      html += `<div class="item"><div class="item-head"><h3>${esc(it.tema)}</h3>
        <span class="freq">Frecuencia ${esc(it.frecuencia)}</span></div>
        ${(it.ejemplos || []).map(q => `<blockquote>“${esc(q)}”</blockquote>`).join("")}</div>`;
    }
    html += `</section>`;
  }
  results.innerHTML = html;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  results.innerHTML = "";
  statusEl.className = "status";
  statusEl.innerHTML = '<span class="spinner"></span>Leyendo comentarios y analizándolos. Puede tardar de 1 a 3 minutos.';
  btn.disabled = true;
  btn.textContent = "Analizando…";
  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: document.getElementById("url").value.trim() }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = typeof data.detail === "string" ? data.detail : "Revisa que el enlace sea una publicación pública de Facebook.";
      throw new Error(msg);
    }
    statusEl.textContent = "";
    render(data);
  } catch (err) {
    statusEl.className = "status error";
    statusEl.textContent = "No se pudo analizar: " + err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analizar comentarios";
  }
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "modelo": GEMINI_MODEL}


@app.get("/", response_class=HTMLResponse)
async def home():
    return HOME_HTML


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    url = str(req.url)

    if "facebook.com" not in url and "fb.watch" not in url:
        raise HTTPException(status_code=400, detail="El enlace debe ser de Facebook.")

    comments = await fetch_comments(url)
    if not comments:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron comentarios. Verifica que la publicación sea pública y tenga comentarios.",
        )

    analysis = await analyze_with_gemini(comments)

    return AnalyzeResponse(
        url=url,
        total_comentarios=len(comments),
        modelo=GEMINI_MODEL,
        analisis=analysis,
    )
