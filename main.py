"""
MVP: Analizador de comentarios de Facebook
Flujo: URL de Facebook -> Apify (facebook-comments-scraper) -> Gemini -> JSON con dolores, objeciones y deseos.
"""

import logging
import os
from typing import List

from apify_client import ApifyClientAsync
from fastapi import FastAPI, HTTPException
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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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

    if isinstance(response.parsed, AnalysisResult):
        return response.parsed

    try:
        return AnalysisResult.model_validate_json(response.text or "")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini no devolvió un JSON válido: {exc}") from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def health():
    return {"status": "ok", "modelo": GEMINI_MODEL}


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


# ---------------------------------------------------------------------------
# Arranque nativo para Render (¡Esto era lo que faltaba!)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
