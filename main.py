"""
Analizador de comentarios (Facebook + YouTube)

Fuentes:
  - Enlace de Facebook  -> Apify (facebook-comments-scraper)
  - Enlace de YouTube   -> YouTube Data API v3
  - Palabra clave       -> YouTube Data API v3 (busca videos y junta sus comentarios)
Luego todo pasa por Gemini y devuelve JSON con dolores, objeciones y deseos.
"""

import asyncio
import logging
import os
import re
from datetime import timedelta
from typing import List, Optional

import httpx
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
logger = logging.getLogger("comment-analyzer")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
MAX_COMMENTS = int(os.getenv("MAX_COMMENTS", "150"))          # Facebook
YT_COMMENTS_PER_VIDEO = int(os.getenv("YT_COMMENTS_PER_VIDEO", "100"))
MAX_COMMENT_CHARS = 600                                       # recorta comentarios muy largos
APIFY_ACTOR_ID = "apify/facebook-comments-scraper"
YT_API = "https://www.googleapis.com/youtube/v3"

app = FastAPI(
    title="Analizador de Comentarios",
    description="Extrae comentarios de Facebook y YouTube y los analiza con Gemini.",
    version="2.0.0",
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
apify_client = ApifyClientAsync(APIFY_API_TOKEN) if APIFY_API_TOKEN else None


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    url: HttpUrl


class YouTubeSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=150)
    max_videos: int = Field(5, ge=1, le=10)


class Insight(BaseModel):
    tema: str = Field(..., description="Resumen corto del dolor/objeción/deseo")
    frecuencia: str = Field(..., description="alta | media | baja")
    ejemplos: List[str] = Field(..., description="Citas textuales de comentarios")


class AnalysisResult(BaseModel):
    dolores: List[Insight]
    objeciones: List[Insight]
    deseos: List[Insight]
    resumen: str


class VideoInfo(BaseModel):
    titulo: str
    canal: str
    url: str
    comentarios_analizados: int


class AnalyzeResponse(BaseModel):
    fuente: str
    consulta: str
    total_comentarios: int
    modelo: str
    analisis: AnalysisResult
    videos: List[VideoInfo] = []


def _clean(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        if isinstance(t, str) and t.strip():
            out.append(t.strip()[:MAX_COMMENT_CHARS])
    return out


# ---------------------------------------------------------------------------
# Facebook (Apify)
# ---------------------------------------------------------------------------
def _field(obj, attr: str, key: str):
    """Lee un campo tanto si Apify devuelve un dict como un objeto."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, attr, None)


async def fetch_facebook_comments(url: str) -> List[str]:
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
    comments = _clean([item.get("text") for item in items if isinstance(item, dict)])
    logger.info("Facebook: %d comentarios", len(comments))
    return comments


# ---------------------------------------------------------------------------
# YouTube (API oficial)
# ---------------------------------------------------------------------------
YT_ID_PATTERNS = [
    r"(?:v=|/videos/|/embed/|/shorts/|/live/|youtu\.be/)([A-Za-z0-9_-]{11})",
]


def extract_video_id(url: str) -> Optional[str]:
    for pattern in YT_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _yt_error_reason(resp: httpx.Response) -> str:
    try:
        return resp.json()["error"]["errors"][0].get("reason", "")
    except Exception:
        return ""


async def yt_get(client: httpx.AsyncClient, path: str, params: dict) -> Optional[dict]:
    """Llama a la API de YouTube. Devuelve None si los comentarios están desactivados."""
    params = {**params, "key": YOUTUBE_API_KEY}
    try:
        resp = await client.get(f"{YT_API}/{path}", params=params)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo conectar con YouTube: {exc}") from exc

    if resp.status_code == 200:
        return resp.json()

    reason = _yt_error_reason(resp)
    if reason in ("commentsDisabled", "videoNotFound", "forbidden"):
        return None
    if reason in ("quotaExceeded", "dailyLimitExceeded"):
        raise HTTPException(status_code=429, detail="Se acabó la cuota diaria gratis de YouTube. Vuelve a intentarlo mañana.")
    if reason in ("keyInvalid", "accessNotConfigured") or resp.status_code in (400, 403):
        raise HTTPException(
            status_code=502,
            detail=f"YouTube rechazó la clave ({reason or resp.status_code}). Revisa YOUTUBE_API_KEY y que la API esté habilitada.",
        )
    raise HTTPException(status_code=502, detail=f"Error de YouTube ({resp.status_code}): {reason}")


async def fetch_video_comments(client: httpx.AsyncClient, video_id: str) -> List[str]:
    comments: List[str] = []
    page_token = None
    while len(comments) < YT_COMMENTS_PER_VIDEO:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, YT_COMMENTS_PER_VIDEO - len(comments)),
            "order": "relevance",
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token
        data = await yt_get(client, "commentThreads", params)
        if not data:
            break
        for item in data.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"].get("textDisplay", "")
            comments.append(text)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return _clean(comments)


async def get_video_details(client: httpx.AsyncClient, video_ids: List[str]) -> List[dict]:
    data = await yt_get(client, "videos", {"part": "snippet,statistics", "id": ",".join(video_ids)})
    videos = []
    for item in (data or {}).get("items", []):
        videos.append({
            "id": item["id"],
            "titulo": item["snippet"].get("title", ""),
            "canal": item["snippet"].get("channelTitle", ""),
            "comment_count": int(item.get("statistics", {}).get("commentCount", 0) or 0),
        })
    return videos


async def collect_youtube(client: httpx.AsyncClient, videos: List[dict]):
    results = await asyncio.gather(*(fetch_video_comments(client, v["id"]) for v in videos))
    all_comments: List[str] = []
    info: List[VideoInfo] = []
    for video, comments in zip(videos, results):
        if not comments:
            continue
        all_comments.extend(comments)
        info.append(VideoInfo(
            titulo=video["titulo"],
            canal=video["canal"],
            url=f"https://www.youtube.com/watch?v={video['id']}",
            comentarios_analizados=len(comments),
        ))
    return all_comments, info


def _require_youtube_key():
    if not YOUTUBE_API_KEY:
        raise HTTPException(status_code=500, detail="Falta la variable YOUTUBE_API_KEY.")


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
- Ignora spam, saludos vacíos y comentarios que solo halagan al creador sin decir nada del tema.
- No inventes nada que no esté en los comentarios. Si una categoría no aparece, devuélvela vacía.
- Escribe en el mismo idioma predominante de los comentarios.
- El resumen debe ser de 2 a 4 frases con la conclusión accionable para marketing."""


async def analyze_with_gemini(comments: List[str], contexto: str) -> AnalysisResult:
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="Falta la variable GEMINI_API_KEY.")

    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(comments))
    user_message = f"Analiza estos {len(comments)} comentarios de {contexto}:\n\n{numbered}"

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

    if isinstance(response.parsed, AnalysisResult):
        return response.parsed
    try:
        return AnalysisResult.model_validate_json(response.text or "")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini no devolvió un JSON válido: {exc}") from exc


NO_COMMENTS = "No se encontraron comentarios. Verifica que la publicación sea pública y tenga comentarios activados."


# ---------------------------------------------------------------------------
# Página web
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
  .intro { color: var(--muted); margin: 0 0 24px; max-width: 60ch; }
  .tabs { display: flex; gap: 6px; margin-bottom: 14px; flex-wrap: wrap; }
  .tab { padding: 9px 14px; font: 600 14px "Instrument Sans", sans-serif; color: var(--muted);
    background: transparent; border: 1.5px solid var(--line); border-radius: 999px; cursor: pointer; }
  .tab[aria-selected="true"] { color: #fff; background: var(--ink); border-color: var(--ink); }
  .hint { color: var(--muted); font-size: 14px; margin: 0 0 12px; }
  form { display: flex; gap: 10px; flex-wrap: wrap; }
  input, select { padding: 14px 16px; font: inherit; color: var(--ink);
    background: var(--paper); border: 1.5px solid var(--line); border-radius: 10px; }
  input { flex: 1 1 300px; min-width: 0; }
  input:focus, select:focus { outline: 3px solid rgba(43,76,126,.25); border-color: var(--accion); }
  .submit { padding: 14px 22px; font: 600 16px "Instrument Sans", sans-serif; color: #fff;
    background: var(--accion); border: 0; border-radius: 10px; cursor: pointer; }
  .submit:disabled { opacity: .6; cursor: wait; }
  button:focus-visible { outline: 3px solid rgba(43,76,126,.4); outline-offset: 2px; }
  [hidden] { display: none !important; }
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
  .sources a { display: block; background: var(--paper); border: 1px solid var(--line); border-radius: 12px;
    padding: 14px 18px; margin-bottom: 10px; color: var(--ink); text-decoration: none; }
  .sources a:hover { border-color: var(--accion); }
  .sources small { color: var(--muted); display: block; }
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
  <p class="intro">Analizamos comentarios reales y los ordenamos en dolores, objeciones y deseos, con citas textuales.</p>

  <div class="tabs" role="tablist">
    <button class="tab" role="tab" data-mode="yt-search" aria-selected="true">Buscar en YouTube</button>
    <button class="tab" role="tab" data-mode="yt-video" aria-selected="false">Video de YouTube</button>
    <button class="tab" role="tab" data-mode="facebook" aria-selected="false">Publicación de Facebook</button>
  </div>
  <p id="hint" class="hint"></p>

  <form id="form">
    <input id="field" required aria-label="Entrada">
    <select id="nvideos" aria-label="Cantidad de videos">
      <option value="3">3 videos</option>
      <option value="5" selected>5 videos</option>
      <option value="8">8 videos</option>
    </select>
    <button id="btn" class="submit" type="submit">Analizar</button>
  </form>
  <p id="status" class="status" role="status"></p>

  <div id="results"></div>
</main>

<script>
const MODES = {
  "yt-search": {
    endpoint: "/youtube/search", type: "text", placeholder: "Ej: cómo emprender con poco dinero",
    hint: "Escribe un tema. Buscamos los videos en español con más comentarios y los analizamos juntos.",
    wait: "Buscando videos y leyendo sus comentarios. Suele tardar menos de un minuto.",
    body: (v) => ({ query: v, max_videos: Number(document.getElementById("nvideos").value) }),
  },
  "yt-video": {
    endpoint: "/youtube/video", type: "url", placeholder: "https://www.youtube.com/watch?v=...",
    hint: "Pega el enlace de un video (también sirven Shorts).",
    wait: "Leyendo los comentarios del video. Suele tardar unos segundos.",
    body: (v) => ({ url: v }),
  },
  "facebook": {
    endpoint: "/analyze", type: "url", placeholder: "https://www.facebook.com/...",
    hint: "Pega el enlace de una publicación pública de Facebook.",
    wait: "Leyendo comentarios de Facebook. Puede tardar de 1 a 3 minutos.",
    body: (v) => ({ url: v }),
  },
};

let mode = "yt-search";
const form = document.getElementById("form");
const field = document.getElementById("field");
const nvideos = document.getElementById("nvideos");
const hint = document.getElementById("hint");
const btn = document.getElementById("btn");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");

function setMode(m) {
  mode = m;
  const cfg = MODES[m];
  document.querySelectorAll(".tab").forEach(t => t.setAttribute("aria-selected", String(t.dataset.mode === m)));
  field.type = cfg.type;
  field.placeholder = cfg.placeholder;
  field.value = "";
  hint.textContent = cfg.hint;
  nvideos.hidden = m !== "yt-search";
  statusEl.textContent = "";
  statusEl.className = "status";
}
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => setMode(t.dataset.mode)));
setMode(mode);

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
    <div class="meta">${data.total_comentarios} comentarios analizados · ${esc(data.fuente)}</div></div>`;
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
  if (data.videos && data.videos.length) {
    html += `<section class="sources"><h2>Videos analizados</h2>`;
    for (const v of data.videos) {
      html += `<a href="${esc(v.url)}" target="_blank" rel="noopener">${esc(v.titulo)}
        <small>${esc(v.canal)} · ${v.comentarios_analizados} comentarios</small></a>`;
    }
    html += `</section>`;
  }
  results.innerHTML = html;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const cfg = MODES[mode];
  results.innerHTML = "";
  statusEl.className = "status";
  statusEl.innerHTML = `<span class="spinner"></span>${cfg.wait}`;
  btn.disabled = true;
  btn.textContent = "Analizando…";
  try {
    const res = await fetch(cfg.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg.body(field.value.trim())),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = typeof data.detail === "string" ? data.detail : "Revisa lo que escribiste e inténtalo de nuevo.";
      throw new Error(msg);
    }
    statusEl.textContent = "";
    render(data);
  } catch (err) {
    statusEl.className = "status error";
    statusEl.textContent = "No se pudo analizar: " + err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analizar";
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
    return {
        "status": "ok",
        "modelo": GEMINI_MODEL,
        "facebook": bool(APIFY_API_TOKEN),
        "youtube": bool(YOUTUBE_API_KEY),
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    return HOME_HTML


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_facebook(req: AnalyzeRequest):
    """Analiza una publicación de Facebook (ruta original, se mantiene igual)."""
    url = str(req.url)
    if "facebook.com" not in url and "fb.watch" not in url:
        raise HTTPException(status_code=400, detail="El enlace debe ser de Facebook.")

    comments = await fetch_facebook_comments(url)
    if not comments:
        raise HTTPException(status_code=404, detail=NO_COMMENTS)

    analysis = await analyze_with_gemini(comments, "una publicación de Facebook")
    return AnalyzeResponse(
        fuente="Facebook", consulta=url, total_comentarios=len(comments),
        modelo=GEMINI_MODEL, analisis=analysis,
    )


@app.post("/youtube/video", response_model=AnalyzeResponse)
async def analyze_youtube_video(req: AnalyzeRequest):
    _require_youtube_key()
    url = str(req.url)
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="No reconozco ese enlace de YouTube.")

    async with httpx.AsyncClient(timeout=30) as client:
        videos = await get_video_details(client, [video_id])
        if not videos:
            raise HTTPException(status_code=404, detail="Ese video no existe o es privado.")
        comments, info = await collect_youtube(client, videos)

    if not comments:
        raise HTTPException(status_code=404, detail="Este video no tiene comentarios o los tiene desactivados.")

    analysis = await analyze_with_gemini(comments, f'un video de YouTube titulado "{videos[0]["titulo"]}"')
    return AnalyzeResponse(
        fuente="YouTube", consulta=url, total_comentarios=len(comments),
        modelo=GEMINI_MODEL, analisis=analysis, videos=info,
    )


@app.post("/youtube/search", response_model=AnalyzeResponse)
async def analyze_youtube_search(req: YouTubeSearchRequest):
    _require_youtube_key()
    query = req.query.strip()

    async with httpx.AsyncClient(timeout=30) as client:
        search = await yt_get(client, "search", {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 25,
            "relevanceLanguage": "es",
            "order": "relevance",
        })
        ids = [it["id"]["videoId"] for it in (search or {}).get("items", []) if it.get("id", {}).get("videoId")]
        if not ids:
            raise HTTPException(status_code=404, detail="No encontré videos para esa búsqueda. Prueba con otras palabras.")

        details = await get_video_details(client, ids)
        best = sorted((v for v in details if v["comment_count"] > 0),
                      key=lambda v: v["comment_count"], reverse=True)[: req.max_videos]
        if not best:
            raise HTTPException(status_code=404, detail="Los videos encontrados no tienen comentarios disponibles.")

        comments, info = await collect_youtube(client, best)

    if not comments:
        raise HTTPException(status_code=404, detail="No pude leer comentarios de los videos encontrados.")

    analysis = await analyze_with_gemini(comments, f'varios videos de YouTube sobre "{query}"')
    return AnalyzeResponse(
        fuente=f"YouTube · {len(info)} videos", consulta=query, total_comentarios=len(comments),
        modelo=GEMINI_MODEL, analisis=analysis, videos=info,
    )
