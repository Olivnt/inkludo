# realtime_server.py
# - Issues short-lived ephemeral tokens for OpenAI Realtime (WebRTC)
# - Exposes /tool/invoice_search to fetch invoice data from Weaviate
# - Handles CORS (localhost:8501 <-> localhost:5050)
# - Shows clear errors so you can debug quickly
#-  uvicorn realtime_server:app --host 0.0.0.0 --port 5050 --reload 
import os
import json
import traceback
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # read .env in current working dir

# ---------------- FastAPI app + CORS ----------------
app = FastAPI()

ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost",
    "http://127.0.0.1",
    "*"  # keep for local dev (no cookies used)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,  # no cookies -> simpler CORS
    max_age=86400,
)

def _corsify(resp: JSONResponse, request: Request) -> JSONResponse:
    origin = request.headers.get("origin", "http://localhost:8501")
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

@app.get("/health")
def health():
    return {
        "ok": True,
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17"),
        "voice": os.getenv("OPENAI_VOICE", "alloy"),
    }

@app.options("/session")
def options_session(request: Request):
    return _corsify(JSONResponse({"ok": True}), request)

@app.options("/tool/invoice_search")
def options_tool(request: Request):
    return _corsify(JSONResponse({"ok": True}), request)

# ---------------- /session: mint ephemeral key ----------------
@app.post("/session")
def create_session(request: Request):
    key   = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
    voice = os.getenv("OPENAI_VOICE", "alloy")
    base  = os.getenv("OPENAI_BASE", "https://api.openai.com")
    verify_ssl = os.getenv("OPENAI_VERIFY_SSL", "true").lower() != "false"

    if not key:
        return _corsify(JSONResponse({"error": "Missing OPENAI_API_KEY on server"}, status_code=500), request)

    url = f"{base}/v1/realtime/sessions"
    payload = {
        "model": model,
        "voice": voice,
        "modalities": ["audio", "text"],
        # Add knobs like turn_detection after it's working
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30, verify=verify_ssl)
    except requests.RequestException as e:
        resp = JSONResponse(
            {"error": "Request to OpenAI failed", "detail": str(e), "trace": traceback.format_exc()},
            status_code=502,
        )
        return _corsify(resp, request)

    # Log a hint server-side for debugging
    print("OpenAI /realtime/sessions:", r.status_code, r.text[:400])

    if r.status_code >= 300:
        resp = JSONResponse(
            {"error": "OpenAI returned an error", "status": r.status_code, "body": r.text},
            status_code=r.status_code,
        )
        return _corsify(resp, request)

    resp = JSONResponse(r.json())
    return _corsify(resp, request)

# ---------------- /tool/invoice_search ----------------
class InvoiceSearchArgs(BaseModel):
    query: str = Field(..., description="Natural language invoice query (supplier/date/amount/invoice no).")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return (1-20).")

def _get_text_embedder():
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if EMBEDDING_MODEL.startswith("text-embedding") or EMBEDDING_MODEL.startswith("openai/"):
        from langchain_openai import OpenAIEmbeddings
        model = EMBEDDING_MODEL.split("/", 1)[-1]
        return OpenAIEmbeddings(model=model)
    from langchain.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def _search_weaviate(query: str, top_k: int) -> List[Dict[str, Any]]:
    import weaviate
    WEAVIATE_URL = (os.getenv("WEAVIATE_URL", "") or "").strip().rstrip("/")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        return [{"error": "WEAVIATE_URL/WEAVIATE_API_KEY not configured"}]
    if not WEAVIATE_URL.startswith(("http://", "https://")):
        WEAVIATE_URL = "https://" + WEAVIATE_URL

    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY))

    emb = _get_text_embedder()
    vec = emb.embed_query(query)

    fields = ["invoice_no", "issue_date", "supplier", "amount_total", "currency", "text", "pdf_url"]
    res = (
        client.query
        .get("InvoiceChunk", fields)
        .with_near_vector({"vector": vec})
        .with_limit(int(top_k))
        .do()
    )
    items = res.get("data", {}).get("Get", {}).get("InvoiceChunk", []) or []
    return [{
        "invoice_no": d.get("invoice_no"),
        "issue_date": d.get("issue_date"),
        "supplier": d.get("supplier"),
        "amount_total": d.get("amount_total"),
        "currency": d.get("currency"),
        "pdf_url": d.get("pdf_url"),
        "snippet": (d.get("text") or "")[:400],
    } for d in items]

@app.post("/tool/invoice_search")
async def invoice_search(request: Request):
    try:
        data = await request.json()
        args = InvoiceSearchArgs(**(data.get("args") or {}))
    except Exception as e:
        return _corsify(JSONResponse({"error": f"Bad args: {e}"}, status_code=400), request)

    try:
        results = _search_weaviate(args.query, args.top_k)
        return _corsify(JSONResponse({"results": results}), request)
    except Exception as e:
        return _corsify(JSONResponse({"error": f"Search failed: {e.__class__.__name__}: {e}"}, status_code=500), request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
