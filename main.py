"""
MVP: Analizador de comentarios de Facebook
Flujo: URL de Facebook -> Apify (facebook-comments-scraper) -> Claude -> JSON con dolores, objeciones y deseos.
"""

import logging
import os
from typing import List

from anthropic import AsyncAnthropic, APIError
from apify_client import ApifyClientAsync
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fb-analyzer")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
MAX_COMMENTS = int(os.getenv("MAX_COMMENTS", "150"))
APIFY_ACTOR_ID = "apify/facebook-comments-scraper"

app = FastAPI(
    title="Analizador de Comentarios de Facebook",
    description="Extrae comentarios con Apify y los analiza con Claude.",
    version="1.0.0",
)

anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
apify_client = ApifyClientAsync(APIFY_API_TOKEN) if APIFY_API_TOKEN else None


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    url: HttpUrl


class Insight(BaseModel):
    tema: str = Field(..., description="Resumen corto del dolor/objeción/deseo")
    frecuencia: str = Field(..., description="alta | media | baja")
    ejemplos: List[str] = Field(default_factory=list, description="Citas textuales de comentarios")


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
        run = await apify_client.actor(APIFY_ACTOR_ID).call(run_input=run_input, timeout_secs=240)
    except Exception as exc:
        logger.exception("Error llamando a Apify")
        raise HTTPException(status_code=502, detail=f"Error al ejecutar Apify: {exc}") from exc

    if not run or run.get("status") != "SUCCEEDED":
        status = run.get("status") if run else "desconocido"
        raise HTTPException(status_code=502, detail=f"El scraper de Apify no terminó bien (estado: {status}).")

    dataset = apify_client.dataset(run["defaultDatasetId"])
    items = (await dataset.list_items()).items

    comments = [
        item["text"].strip()
        for item in items
        if isinstance(item.get("text"), str) and item["text"].strip()
    ]
    logger.info("Comentarios extraídos: %d", len(comments))
    return comments


# ---------------------------------------------------------------------------
# Claude: análisis de marketing con salida estructurada (tool use forzado)
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

INSIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "tema": {"type": "string"},
        "frecuencia": {"type": "string", "enum": ["alta", "media", "baja"]},
        "ejemplos": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tema", "frecuencia", "ejemplos"],
}

ANALYSIS_TOOL = {
    "name": "guardar_analisis",
    "description": "Guarda el análisis estructurado de los comentarios.",
    "input_schema": {
        "type": "object",
        "properties": {
            "dolores": {"type": "array", "items": INSIGHT_SCHEMA},
            "objeciones": {"type": "array", "items": INSIGHT_SCHEMA},
            "deseos": {"type": "array", "items": INSIGHT_SCHEMA},
            "resumen": {"type": "string"},
        },
        "required": ["dolores", "objeciones", "deseos", "resumen"],
    },
}


async def analyze_with_claude(comments: List[str]) -> AnalysisResult:
    if anthropic_client is None:
        raise HTTPException(status_code=500, detail="Falta la variable ANTHROPIC_API_KEY.")

    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(comments))
    user_message = f"Analiza estos {len(comments)} comentarios de Facebook:\n\n{numbered}"

    try:
        response = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[ANALYSIS_TOOL],
            tool_choice={"type": "tool", "name": ANALYSIS_TOOL["name"]},
            messages=[{"role": "user", "content": user_message}],
        )
    except APIError as exc:
        logger.exception("Error llamando a Claude")
        raise HTTPException(status_code=502, detail=f"Error en la API de Claude: {exc}") from exc

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise HTTPException(status_code=502, detail="Claude no devolvió un análisis estructurado.")

    try:
        return AnalysisResult.model_validate(tool_block.input)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Respuesta de Claude inválida: {exc}") from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def health():
    return {"status": "ok", "modelo": CLAUDE_MODEL}


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

    analysis = await analyze_with_claude(comments)

    return AnalyzeResponse(
        url=url,
        total_comentarios=len(comments),
        modelo=CLAUDE_MODEL,
        analisis=analysis,
    )
