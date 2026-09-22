"""
Analizador de comentarios + Agente Analista de Mercado (Facebook, YouTube o texto pegado)

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
from typing import List, Literal, Optional

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
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")
MAX_COMMENTS = int(os.getenv("MAX_COMMENTS", "150"))          # Facebook
YT_COMMENTS_PER_VIDEO = int(os.getenv("YT_COMMENTS_PER_VIDEO", "100"))
MAX_COMMENT_CHARS = 600
MAX_PASTED = int(os.getenv("MAX_PASTED", "400"))                 # tope de comentarios pegados
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
Modo = Literal["rapido", "mercado"]


class AnalyzeRequest(BaseModel):
    url: HttpUrl
    modo: Modo = "rapido"
    nicho: Optional[str] = Field(None, max_length=150)


class YouTubeSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=150)
    max_videos: int = Field(5, ge=1, le=10)
    modo: Modo = "rapido"
    nicho: Optional[str] = Field(None, max_length=150)


class PasteRequest(BaseModel):
    texto: str = Field(..., min_length=20, max_length=400_000)
    modo: Modo = "mercado"
    nicho: Optional[str] = Field(None, max_length=150)


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



# --- Agente Analista de Mercado: estructura del informe (8 pasos) ---------
class FraseRepetida(BaseModel):
    frase: str = Field(..., description="Frase exacta o casi exacta que usa la gente")
    frecuencia: str = Field(..., description="alta | media | baja")
    que_revela: str = Field(..., description="Qué dice esta frase sobre su frustración")


class IntentoFallido(BaseModel):
    que_intentaron: str
    por_que_fallo: str = Field(..., description="Por qué no les funcionó, según los comentarios")
    ejemplo: str = Field(..., description="Cita textual que lo muestra")


class ResultadoDeseado(BaseModel):
    lo_que_dicen: str = Field(..., description="El resultado que piden explícitamente")
    lo_que_realmente_buscan: str = Field(..., description="El resultado profundo, aunque no lo digan")
    evidencia: List[str] = Field(..., description="1 a 3 citas textuales")


class DolorEmocional(BaseModel):
    intensidad: int = Field(..., ge=1, le=10, description="1 = molestia leve, 10 = desesperación")
    emociones: List[str] = Field(..., description="Emociones dominantes: vergüenza, miedo, frustración...")
    costo_de_seguir_igual: str = Field(..., description="Qué les duele si en un mes siguen igual")
    evidencia: List[str] = Field(..., description="1 a 3 citas textuales")


class DisposicionPago(BaseModel):
    nivel: str = Field(..., description="alta | media | baja")
    senales: List[str] = Field(..., description="Señales de que pagarían o de que no pagarían")
    en_que_ya_gastan: str = Field(..., description="Productos, cursos o servicios que mencionan haber pagado; o 'No se menciona'")


class BrechaOportunidad(BaseModel):
    descripcion: str = Field(..., description="Lo que nadie está resolviendo directamente")
    por_que_no_esta_resuelto: str
    evidencia: List[str] = Field(..., description="1 a 3 citas textuales")


class ProblemaUrgente(BaseModel):
    problema_macro_a_evitar: str = Field(..., description="El problema general del nicho (título de curso de 8 módulos)")
    problema_urgente_especifico: str = Field(..., description="Una situación concreta, en palabras del cliente")
    por_que_es_especifico: str = Field(..., description="Por qué no podría ser el título de un curso de 8 módulos")


class PropuestaProducto(BaseModel):
    nombre: str = Field(..., description="Nombre del producto con el mecanismo único integrado")
    mecanismo_unico: str = Field(..., description="El cómo diferente, diseñado a partir de lo que ya falló")
    promesa: str = Field(..., description="Resultado específico y creíble, en una frase")
    formato: str = Field(..., description="Plantilla, checklist, mini curso, taller, guion, etc.")
    que_incluye: List[str]
    precio_sugerido_usd: float
    justificacion_precio: str


class Veredicto(BaseModel):
    recomendacion: str = Field(..., description="CREAR | VALIDAR MÁS | NO CREAR")
    puntuacion: int = Field(..., ge=1, le=10, description="Qué tan buena es la oportunidad")
    justificacion: str
    riesgos: List[str]
    siguiente_paso: str = Field(..., description="La acción concreta para validar en 7 días o menos")


class MarketReport(BaseModel):
    resumen_ejecutivo: str
    frases_repetidas: List[FraseRepetida]
    intentos_fallidos: List[IntentoFallido]
    resultado_deseado: ResultadoDeseado
    dolor_emocional: DolorEmocional
    disposicion_a_pagar: DisposicionPago
    brecha_oportunidad: BrechaOportunidad
    problema_urgente: ProblemaUrgente
    propuesta_producto: PropuestaProducto
    veredicto: Veredicto


class AnalyzeResponse(BaseModel):
    fuente: str
    consulta: str
    total_comentarios: int
    modelo: str
    modo: Modo
    analisis: Optional[AnalysisResult] = None
    mercado: Optional[MarketReport] = None
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


def _yt_error(resp: httpx.Response):
    """Devuelve (reason, message) del error de Google."""
    try:
        err = resp.json()["error"]
        first = (err.get("errors") or [{}])[0]
        return first.get("reason", ""), err.get("message", "")
    except Exception:
        return "", resp.text[:200]


async def yt_get(client: httpx.AsyncClient, path: str, params: dict) -> Optional[dict]:
    """Llama a la API de YouTube. En commentThreads devuelve None si el video no permite leer comentarios."""
    params = {**params, "key": YOUTUBE_API_KEY.strip()}
    try:
        resp = await client.get(f"{YT_API}/{path}", params=params)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo conectar con YouTube: {exc}") from exc

    if resp.status_code == 200:
        return resp.json()

    reason, message = _yt_error(resp)
    logger.warning("YouTube %s -> %s %s: %s", path, resp.status_code, reason, message)

    if path == "commentThreads" and reason in ("commentsDisabled", "videoNotFound", "forbidden"):
        return None
    if reason in ("quotaExceeded", "dailyLimitExceeded"):
        raise HTTPException(status_code=429, detail="Se acabó la cuota diaria gratis de YouTube. Vuelve a intentarlo mañana.")
    if "API key not valid" in message or reason == "keyInvalid":
        raise HTTPException(status_code=502, detail="La clave de YouTube no es válida. Revisa que YOUTUBE_API_KEY esté copiada completa, sin espacios ni comillas.")
    if reason in ("accessNotConfigured", "SERVICE_DISABLED") or "has not been used" in message or "is disabled" in message:
        raise HTTPException(status_code=502, detail="La YouTube Data API v3 no está habilitada en el proyecto de Google de esa clave.")
    if "blocked" in message.lower() or reason in ("forbidden", "API_KEY_SERVICE_BLOCKED"):
        raise HTTPException(status_code=502, detail="Esa clave tiene restricciones que no permiten usar YouTube. Crea una clave nueva sin restricciones o permite YouTube Data API v3.")
    raise HTTPException(status_code=502, detail=f"Error de YouTube ({resp.status_code} {reason}): {message}")


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
# Gemini
# ---------------------------------------------------------------------------
QUICK_PROMPT = """Eres un estratega de marketing experto en investigación de mercado y copywriting de respuesta directa.
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


MARKET_PROMPT = """Eres el AGENTE ANALISTA DE MERCADO. Tu trabajo es decidir, con evidencia, si existe un problema real que vale la pena resolver con un producto, antes de que alguien invierta tiempo en crearlo.
Recibes comentarios reales de redes sociales. Los conviertes en inteligencia de mercado accionable siguiendo 8 pasos, en este orden:

1. FRASES REPETIDAS: detecta las frases exactas que se repiten, las palabras que la gente usa cuando describe su frustración. Cópialas como las escriben (con su ortografía y jerga); no las parafrasees.
2. INTENTOS FALLIDOS: identifica qué ya intentaron y no les funcionó, y por qué. Es la materia prima del mecanismo único.
3. RESULTADO DESEADO: distingue lo que dicen que quieren del resultado que realmente buscan, aunque no lo digan directamente.
4. DOLOR EMOCIONAL: evalúa la intensidad (1 a 10), las emociones dominantes y qué les duele si dentro de un mes siguen igual.
5. DISPOSICIÓN A PAGAR: busca señales reales (ya pagaron cursos, contrataron a alguien, piden recomendaciones de herramientas, dicen "pagaría por…", o al revés: "todo está gratis en YouTube"). No la supongas.
6. BRECHA DE OPORTUNIDAD: detecta lo que nadie está resolviendo directamente, según lo que los comentarios reclaman y no encuentran.
7. PROPUESTA DE PRODUCTO: propone un producto con su mecanismo único integrado en el nombre. El mecanismo debe salir de los pasos 2 y 6: explica por qué esto funcionará donde lo anterior falló.
8. VEREDICTO: recomienda CREAR, VALIDAR MÁS o NO CREAR, con puntuación de 1 a 10, riesgos y un siguiente paso concreto para validar en 7 días o menos.

LA REGLA MÁS IMPORTANTE: el problema urgente específico.
Nunca resuelvas el problema macro del nicho. Siempre resuelve un problema urgente y específico dentro de ese nicho.
- Macro (evitar): "Cómo vender más". Específico (atacar): "Llego al cierre y el cliente me dice 'lo pienso' y desaparece".
- Macro: "Cómo bajar de peso". Específico: "Empiezo la dieta los lunes y la abandono el miércoles".
- Macro: "Cómo salvar mi matrimonio". Específico: "Mi pareja y yo peleamos cada vez que hablamos de dinero".
Prueba: si el problema podría ser el título de un curso de 8 módulos en Hotmart, es macro. Ve más profundo. El problema urgente específico debe describir una situación concreta, en palabras del cliente, y el producto propuesto debe resolver SOLO ese problema.

Reglas de evidencia y honestidad:
- Todo debe apoyarse en los comentarios. Las citas van textuales. No inventes datos, cifras ni testimonios.
- Si la evidencia es débil (pocos comentarios útiles, dolor bajo, sin señales de pago), dilo y baja la puntuación. Un "NO CREAR" o "VALIDAR MÁS" bien justificado vale más que un "CREAR" optimista.
- Si una lista no tiene evidencia, devuélvela vacía en lugar de rellenarla.
- El precio sugerido en USD debe ser coherente con el formato, la intensidad del dolor y las señales de pago. Justifícalo.
- Ignora spam, saludos y halagos vacíos al creador.
- Escribe en el idioma predominante de los comentarios, claro y directo, para un emprendedor que va a tomar una decisión."""


RETRYABLE = {429, 500, 503, 504}


async def _gemini_json(system: str, user_message: str, schema):
    """Llama a Gemini con reintentos. Si el modelo principal está saturado, prueba el de respaldo."""
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="Falta la variable GEMINI_API_KEY.")

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.3,
    )
    models = [GEMINI_MODEL] + ([GEMINI_FALLBACK_MODEL] if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL else [])
    delays = [0, 3, 8]  # segundos de espera antes de cada intento
    last_exc = None

    for model in models:
        for delay in delays:
            if delay:
                await asyncio.sleep(delay)
            try:
                response = await gemini_client.aio.models.generate_content(
                    model=model, contents=user_message, config=config,
                )
            except genai_errors.APIError as exc:
                last_exc = exc
                code = getattr(exc, "code", None)
                logger.warning("Gemini %s falló (%s). Reintentando...", model, code)
                if code in RETRYABLE:
                    continue
                if code == 404:
                    break  # modelo no disponible: pasa al de respaldo
                raise HTTPException(status_code=502, detail=f"Error en la API de Gemini: {exc}") from exc

            if model != GEMINI_MODEL:
                logger.info("Respuesta obtenida con el modelo de respaldo %s", model)
            if isinstance(response.parsed, schema):
                return response.parsed
            try:
                return schema.model_validate_json(response.text or "")
            except Exception as exc:
                last_exc = exc
                logger.warning("JSON inválido de %s, reintentando", model)
                continue

    code = getattr(last_exc, "code", None)
    if code in RETRYABLE:
        raise HTTPException(
            status_code=503,
            detail="Gemini está saturado en este momento (mucha demanda). Espera un par de minutos y vuelve a intentarlo.",
        )
    raise HTTPException(status_code=502, detail=f"Error en la API de Gemini: {last_exc}")


async def run_analysis(comments: List[str], contexto: str, modo: str, nicho: Optional[str]) -> dict:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(comments))
    nicho_txt = f"\nNicho o tema: {nicho.strip()}" if nicho and nicho.strip() else ""
    user_message = f"Fuente: {contexto}.{nicho_txt}\nTotal de comentarios: {len(comments)}\n\nCOMENTARIOS:\n{numbered}"

    if modo == "mercado":
        return {"mercado": await _gemini_json(MARKET_PROMPT, user_message, MarketReport)}
    return {"analisis": await _gemini_json(QUICK_PROMPT, user_message, AnalysisResult)}


def parse_pasted(texto: str) -> List[str]:
    """Convierte el texto pegado en una lista de comentarios (uno por línea)."""
    lines = []
    for raw in texto.splitlines():
        line = re.sub(r"^\s*(\d+[\.\)\-:]|[-•*·])\s*", "", raw).strip()
        if len(line) >= 3:
            lines.append(line)
    return _clean(lines)[:MAX_PASTED]


NO_COMMENTS = "No se encontraron comentarios. Verifica que la publicación sea pública y tenga comentarios activados."


# ---------------------------------------------------------------------------
# Página web
# ---------------------------------------------------------------------------
HOME_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analista de Mercado</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600;700&family=Newsreader:ital,opsz@1,6..72&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #EEF1F6; --ink: #1B2A41; --muted: #5B6B82; --line: #D3DAE5; --paper: #FFFFFF;
    --dolor: #C2410C; --objecion: #9A6700; --deseo: #0F766E; --accion: #2B4C7E;
    --si: #0F766E; --tal: #9A6700; --no: #B42318;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.55 "Instrument Sans", system-ui, -apple-system, sans-serif; }
  main { max-width: 780px; margin: 0 auto; padding: 40px 20px 80px; }
  h1 { font-size: clamp(28px, 6vw, 40px); line-height: 1.1; margin: 0 0 10px; letter-spacing: -0.02em; }
  .intro { color: var(--muted); margin: 0 0 24px; max-width: 62ch; }
  .label { font-size: 13px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 18px 0 8px; }
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .tab { padding: 9px 14px; font: 600 14px "Instrument Sans", sans-serif; color: var(--muted);
    background: transparent; border: 1.5px solid var(--line); border-radius: 999px; cursor: pointer; }
  .tab[aria-pressed="true"] { color: #fff; background: var(--ink); border-color: var(--ink); }
  .hint { color: var(--muted); font-size: 14px; margin: 14px 0 10px; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; }
  input, select, textarea { padding: 14px 16px; font: inherit; color: var(--ink);
    background: var(--paper); border: 1.5px solid var(--line); border-radius: 10px; }
  input { flex: 1 1 300px; min-width: 0; }
  textarea { width: 100%; min-height: 220px; resize: vertical; font-size: 15px; }
  input:focus, select:focus, textarea:focus { outline: 3px solid rgba(43,76,126,.25); border-color: var(--accion); }
  .counter { font-size: 13px; color: var(--muted); margin-top: 6px; }
  .submit { margin-top: 14px; padding: 14px 22px; font: 600 16px "Instrument Sans", sans-serif; color: #fff;
    background: var(--accion); border: 0; border-radius: 10px; cursor: pointer; }
  .submit:disabled { opacity: .6; cursor: wait; }
  button:focus-visible { outline: 3px solid rgba(43,76,126,.4); outline-offset: 2px; }
  [hidden] { display: none !important; }
  .status { margin: 18px 0 0; color: var(--muted); min-height: 1.5em; }
  .status.error { color: var(--no); font-weight: 600; }
  .summary { margin: 36px 0 8px; padding: 22px 24px; background: var(--paper);
    border-left: 5px solid var(--accion); border-radius: 4px 12px 12px 4px; }
  .summary p { margin: 0; font-size: 18px; }
  .meta { color: var(--muted); font-size: 14px; margin-top: 10px; }
  section { margin-top: 36px; }
  h2 { font-size: 22px; margin: 0 0 14px; display: flex; align-items: center; gap: 10px; }
  h2 .dot { width: 12px; height: 12px; border-radius: 50%; background: var(--c); }
  h2 .step { font-size: 13px; font-weight: 700; color: #fff; background: var(--ink); border-radius: 6px; padding: 2px 8px; }
  .count { font-weight: 400; color: var(--muted); font-size: 16px; }
  .item { background: var(--paper); border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; border: 1px solid var(--line); }
  .item-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
  .item h3 { margin: 0; font-size: 17px; }
  .item p { margin: 8px 0 0; }
  .freq { font-size: 13px; font-weight: 600; color: var(--c, var(--accion)); white-space: nowrap; }
  blockquote { margin: 12px 0 0; padding: 0 0 0 14px; border-left: 2px solid var(--c, var(--accion));
    font: italic 17px/1.5 "Newsreader", Georgia, serif; color: #33445C; }
  blockquote + blockquote { margin-top: 8px; }
  .empty { color: var(--muted); font-style: italic; margin: 0; }
  .verdict { margin-top: 28px; padding: 24px; border-radius: 14px; color: #fff; background: var(--v); }
  .verdict .top { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
  .verdict .rec { font-size: 26px; font-weight: 700; letter-spacing: .02em; }
  .verdict .score { font-size: 15px; font-weight: 600; background: rgba(255,255,255,.18); padding: 6px 12px; border-radius: 999px; }
  .verdict p { margin: 12px 0 0; }
  .verdict .next { margin-top: 14px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,.3); }
  .problem .macro { text-decoration: line-through; color: var(--muted); }
  .problem .specific { font: italic 21px/1.4 "Newsreader", Georgia, serif; margin: 10px 0 0; }
  .product { border: 2px solid var(--ink); }
  .product .name { font-size: 24px; font-weight: 700; margin: 0; line-height: 1.2; }
  .product .price { font-size: 30px; font-weight: 700; margin: 14px 0 0; }
  .product ul, .item ul { margin: 8px 0 0; padding-left: 20px; }
  .kv { color: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; margin: 14px 0 2px; }
  .meter { height: 8px; background: var(--line); border-radius: 99px; overflow: hidden; margin-top: 8px; }
  .meter span { display: block; height: 100%; background: var(--dolor); }
  .sources a { display: block; background: var(--paper); border: 1px solid var(--line); border-radius: 12px;
    padding: 14px 18px; margin-bottom: 10px; color: var(--ink); text-decoration: none; }
  .sources small { color: var(--muted); display: block; }
  .spinner { display: inline-block; width: 14px; height: 14px; margin-right: 8px; vertical-align: -2px;
    border: 2px solid var(--line); border-top-color: var(--accion); border-radius: 50%; animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) { .spinner { animation: none; } }
</style>
</head>
<body>
<main>
  <h1>Analista de Mercado</h1>
  <p class="intro">Antes de crear un producto, confirma que hay un problema real que la gente está gritando en internet. Del dolor al producto, con evidencia.</p>

  <div class="label">Tipo de análisis</div>
  <div class="tabs" id="kinds">
    <button class="tab" data-kind="mercado" aria-pressed="true">Analista de mercado · 8 pasos</button>
    <button class="tab" data-kind="rapido" aria-pressed="false">Análisis rápido</button>
  </div>

  <div class="label">Fuente de los comentarios</div>
  <div class="tabs" id="sources">
    <button class="tab" data-src="paste" aria-pressed="true">Pegar comentarios</button>
    <button class="tab" data-src="yt-search" aria-pressed="false">Buscar en YouTube</button>
    <button class="tab" data-src="yt-video" aria-pressed="false">Video de YouTube</button>
    <button class="tab" data-src="facebook" aria-pressed="false">Facebook</button>
  </div>
  <p id="hint" class="hint"></p>

  <form id="form">
    <div id="pasteBox">
      <textarea id="texto" placeholder="Pega aquí 200-300 comentarios, uno por línea (de YouTube, TikTok, Reddit, foros...)"></textarea>
      <div id="counter" class="counter">0 comentarios detectados</div>
    </div>
    <div class="row" id="lineBox" hidden>
      <input id="field" aria-label="Entrada">
      <select id="nvideos" aria-label="Cantidad de videos">
        <option value="3">3 videos</option>
        <option value="5" selected>5 videos</option>
        <option value="8">8 videos</option>
      </select>
    </div>
    <div class="row" style="margin-top:10px">
      <input id="nicho" maxlength="150" placeholder="Nicho o tema (opcional). Ej: docentes que quieren usar IA" aria-label="Nicho">
    </div>
    <button id="btn" class="submit" type="submit">Analizar</button>
  </form>
  <p id="status" class="status" role="status"></p>

  <div id="results"></div>
</main>

<script>
const SOURCES = {
  "paste": { endpoint: "/paste", hint: "Pega los comentarios, uno por línea. Lo ideal son 200-300.",
    wait: "Analizando los comentarios.", body: () => ({ texto: el("texto").value }) },
  "yt-search": { endpoint: "/youtube/search", type: "text", placeholder: "Ej: cómo emprender con poco dinero",
    hint: "Escribe un tema. Buscamos videos en español con más comentarios y los analizamos juntos.",
    wait: "Buscando videos y leyendo sus comentarios.", body: () => ({ query: el("field").value.trim(), max_videos: Number(el("nvideos").value) }) },
  "yt-video": { endpoint: "/youtube/video", type: "url", placeholder: "https://www.youtube.com/watch?v=...",
    hint: "Pega el enlace de un video (también sirven Shorts).",
    wait: "Leyendo los comentarios del video.", body: () => ({ url: el("field").value.trim() }) },
  "facebook": { endpoint: "/analyze", type: "url", placeholder: "https://www.facebook.com/...",
    hint: "Pega el enlace de una publicación pública de Facebook. Puede tardar de 1 a 3 minutos.",
    wait: "Leyendo comentarios de Facebook.", body: () => ({ url: el("field").value.trim() }) },
};
const el = (id) => document.getElementById(id);
let src = "paste", kind = "mercado";

function press(groupId, attr, value) {
  document.querySelectorAll(`#${groupId} .tab`).forEach(t => t.setAttribute("aria-pressed", String(t.dataset[attr] === value)));
}
function setSource(s) {
  src = s; const cfg = SOURCES[s];
  press("sources", "src", s);
  el("pasteBox").hidden = s !== "paste";
  el("lineBox").hidden = s === "paste";
  el("texto").required = s === "paste";
  el("field").required = s !== "paste";
  if (cfg.type) { el("field").type = cfg.type; el("field").placeholder = cfg.placeholder; el("field").value = ""; }
  el("nvideos").hidden = s !== "yt-search";
  el("hint").textContent = cfg.hint;
}
document.querySelectorAll("#sources .tab").forEach(t => t.addEventListener("click", () => setSource(t.dataset.src)));
document.querySelectorAll("#kinds .tab").forEach(t => t.addEventListener("click", () => { kind = t.dataset.kind; press("kinds", "kind", kind); }));
el("texto").addEventListener("input", () => {
  const n = el("texto").value.split("\\n").filter(l => l.trim().length >= 3).length;
  el("counter").textContent = `${n} comentarios detectados` + (n && n < 50 ? " · con menos de 50 el análisis será poco confiable" : "");
});
setSource(src);

function esc(t) { const d = document.createElement("div"); d.textContent = t ?? ""; return d.innerHTML; }
const quotes = (arr, c) => (arr || []).map(q => `<blockquote style="--c:${c}">“${esc(q)}”</blockquote>`).join("");
const list = (arr) => (arr && arr.length) ? `<ul>${arr.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : `<p class="empty">Sin evidencia en los comentarios.</p>`;

function sources(data) {
  if (!data.videos || !data.videos.length) return "";
  return `<section class="sources"><h2>Videos analizados</h2>` + data.videos.map(v =>
    `<a href="${esc(v.url)}" target="_blank" rel="noopener">${esc(v.titulo)}<small>${esc(v.canal)} · ${v.comentarios_analizados} comentarios</small></a>`).join("") + `</section>`;
}

function renderQuick(data) {
  const a = data.analisis;
  const CATS = [["dolores", "Dolores", "var(--dolor)"], ["objeciones", "Objeciones", "var(--objecion)"], ["deseos", "Deseos", "var(--deseo)"]];
  let html = `<div class="summary"><p>${esc(a.resumen)}</p><div class="meta">${data.total_comentarios} comentarios · ${esc(data.fuente)}</div></div>`;
  for (const [key, label, color] of CATS) {
    const items = a[key] || [];
    html += `<section style="--c:${color}"><h2><span class="dot"></span>${label} <span class="count">(${items.length})</span></h2>`;
    if (!items.length) html += `<p class="empty">No aparecen en estos comentarios.</p>`;
    for (const it of items) html += `<div class="item"><div class="item-head"><h3>${esc(it.tema)}</h3><span class="freq">Frecuencia ${esc(it.frecuencia)}</span></div>${quotes(it.ejemplos, color)}</div>`;
    html += `</section>`;
  }
  return html + sources(data);
}

function verdictColor(rec) {
  const r = (rec || "").toUpperCase();
  if (r.startsWith("NO")) return "var(--no)";
  if (r.startsWith("VALID")) return "var(--tal)";
  return "var(--si)";
}

function renderMarket(data) {
  const m = data.mercado, v = m.veredicto, p = m.propuesta_producto, pu = m.problema_urgente;
  const d = m.dolor_emocional, pay = m.disposicion_a_pagar, r = m.resultado_deseado, b = m.brecha_oportunidad;
  let html = `<div class="summary"><p>${esc(m.resumen_ejecutivo)}</p><div class="meta">${data.total_comentarios} comentarios · ${esc(data.fuente)}</div></div>`;

  html += `<div class="verdict" style="--v:${verdictColor(v.recomendacion)}">
    <div class="top"><span class="rec">${esc(v.recomendacion)}</span><span class="score">Oportunidad ${v.puntuacion}/10</span></div>
    <p>${esc(v.justificacion)}</p>
    ${v.riesgos && v.riesgos.length ? `<p><strong>Riesgos:</strong> ${v.riesgos.map(esc).join(" · ")}</p>` : ""}
    <p class="next"><strong>Siguiente paso (7 días):</strong> ${esc(v.siguiente_paso)}</p></div>`;

  html += `<section><h2>El problema urgente específico</h2><div class="item problem">
    <div class="kv">Problema macro (evitar)</div><div class="macro">${esc(pu.problema_macro_a_evitar)}</div>
    <div class="kv">Problema a atacar</div><p class="specific">“${esc(pu.problema_urgente_especifico)}”</p>
    <p>${esc(pu.por_que_es_especifico)}</p></div></section>`;

  html += `<section><h2><span class="step">7</span> Propuesta de producto</h2><div class="item product">
    <p class="name">${esc(p.nombre)}</p>
    <div class="kv">Mecanismo único</div><div>${esc(p.mecanismo_unico)}</div>
    <div class="kv">Promesa</div><div>${esc(p.promesa)}</div>
    <div class="kv">Formato</div><div>${esc(p.formato)}</div>
    <div class="kv">Incluye</div>${list(p.que_incluye)}
    <p class="price">US$ ${Number(p.precio_sugerido_usd).toFixed(p.precio_sugerido_usd % 1 ? 2 : 0)}</p>
    <div>${esc(p.justificacion_precio)}</div></div></section>`;

  html += `<section><h2><span class="step">1</span> Frases que se repiten</h2>`;
  html += (m.frases_repetidas || []).length ? m.frases_repetidas.map(f => `<div class="item"><div class="item-head"><h3>“${esc(f.frase)}”</h3><span class="freq">Frecuencia ${esc(f.frecuencia)}</span></div><p>${esc(f.que_revela)}</p></div>`).join("") : `<p class="empty">Sin frases repetidas claras.</p>`;
  html += `</section>`;

  html += `<section><h2><span class="step">2</span> Lo que ya intentaron y no funcionó</h2>`;
  html += (m.intentos_fallidos || []).length ? m.intentos_fallidos.map(i => `<div class="item"><h3>${esc(i.que_intentaron)}</h3><p>${esc(i.por_que_fallo)}</p>${quotes([i.ejemplo], "var(--objecion)")}</div>`).join("") : `<p class="empty">No mencionan intentos previos.</p>`;
  html += `</section>`;

  html += `<section><h2><span class="step">3</span> El resultado que realmente buscan</h2><div class="item">
    <div class="kv">Lo que dicen</div><div>${esc(r.lo_que_dicen)}</div>
    <div class="kv">Lo que realmente buscan</div><div><strong>${esc(r.lo_que_realmente_buscan)}</strong></div>${quotes(r.evidencia, "var(--deseo)")}</div></section>`;

  html += `<section><h2><span class="step">4</span> Dolor emocional</h2><div class="item">
    <div class="item-head"><h3>Intensidad ${d.intensidad}/10</h3><span class="freq" style="--c:var(--dolor)">${(d.emociones || []).map(esc).join(" · ")}</span></div>
    <div class="meter"><span style="width:${d.intensidad * 10}%"></span></div>
    <div class="kv">Si en un mes siguen igual</div><div>${esc(d.costo_de_seguir_igual)}</div>${quotes(d.evidencia, "var(--dolor)")}</div></section>`;

  html += `<section><h2><span class="step">5</span> Disposición a pagar</h2><div class="item">
    <h3>Nivel ${esc(pay.nivel)}</h3>${list(pay.senales)}
    <div class="kv">En qué ya gastan</div><div>${esc(pay.en_que_ya_gastan)}</div></div></section>`;

  html += `<section><h2><span class="step">6</span> Brecha de oportunidad</h2><div class="item">
    <h3>${esc(b.descripcion)}</h3><p>${esc(b.por_que_no_esta_resuelto)}</p>${quotes(b.evidencia, "var(--accion)")}</div></section>`;

  return html + sources(data);
}

el("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const cfg = SOURCES[src];
  el("results").innerHTML = "";
  el("status").className = "status";
  const extra = kind === "mercado" ? " El análisis de 8 pasos puede tardar un poco más." : "";
  el("status").innerHTML = `<span class="spinner"></span>${cfg.wait}${extra}`;
  el("btn").disabled = true; el("btn").textContent = "Analizando…";
  try {
    const body = { ...cfg.body(), modo: kind, nicho: el("nicho").value.trim() || null };
    const res = await fetch(cfg.endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      let msg = "Revisa lo que escribiste e inténtalo de nuevo.";
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail) && data.detail[0]) msg = "Dato no válido: " + (data.detail[0].msg || "");
      throw new Error(msg);
    }
    el("status").textContent = "";
    el("results").innerHTML = data.mercado ? renderMarket(data) : renderQuick(data);
  } catch (err) {
    el("status").className = "status error";
    el("status").textContent = "No se pudo analizar: " + err.message;
  } finally {
    el("btn").disabled = false; el("btn").textContent = "Analizar";
  }
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def build_response(fuente, consulta, comments, modo, result, videos=None) -> AnalyzeResponse:
    return AnalyzeResponse(
        fuente=fuente, consulta=consulta, total_comentarios=len(comments),
        modelo=GEMINI_MODEL, modo=modo, videos=videos or [], **result,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "modelo": GEMINI_MODEL, "facebook": bool(APIFY_API_TOKEN), "youtube": bool(YOUTUBE_API_KEY)}


@app.get("/", response_class=HTMLResponse)
async def home():
    return HOME_HTML


@app.post("/paste", response_model=AnalyzeResponse)
async def analyze_pasted(req: PasteRequest):
    comments = parse_pasted(req.texto)
    if len(comments) < 10:
        raise HTTPException(status_code=400, detail="Pega al menos 10 comentarios, uno por línea (lo ideal son 200-300).")
    result = await run_analysis(comments, "comentarios pegados de redes sociales", req.modo, req.nicho)
    return build_response("Comentarios pegados", "texto pegado", comments, req.modo, result)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_facebook(req: AnalyzeRequest):
    url = str(req.url)
    if "facebook.com" not in url and "fb.watch" not in url:
        raise HTTPException(status_code=400, detail="El enlace debe ser de Facebook.")
    comments = await fetch_facebook_comments(url)
    if not comments:
        raise HTTPException(status_code=404, detail=NO_COMMENTS)
    result = await run_analysis(comments, "una publicación de Facebook", req.modo, req.nicho)
    return build_response("Facebook", url, comments, req.modo, result)


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
    result = await run_analysis(comments, f'un video de YouTube titulado "{videos[0]["titulo"]}"', req.modo, req.nicho)
    return build_response("YouTube", url, comments, req.modo, result, info)


@app.post("/youtube/search", response_model=AnalyzeResponse)
async def analyze_youtube_search(req: YouTubeSearchRequest):
    _require_youtube_key()
    query = req.query.strip()

    async with httpx.AsyncClient(timeout=30) as client:
        search = await yt_get(client, "search", {
            "part": "snippet", "q": query, "type": "video",
            "maxResults": 25, "relevanceLanguage": "es", "order": "relevance",
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
    result = await run_analysis(comments, f'varios videos de YouTube sobre "{query}"', req.modo, req.nicho or query)
    return build_response(f"YouTube · {len(info)} videos", query, comments, req.modo, result, info)
