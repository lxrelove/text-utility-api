from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import os, re, httpx
from collections import Counter

app = FastAPI(
    title="AI Text Utility API",
    description="Summarize, analyze sentiment, extract keywords, and detect language from any text.",
    version="1.0.0",
)

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY_NAME = "X-RapidAPI-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
VALID_API_KEYS = os.environ.get("VALID_API_KEYS", "dev-test-key").split(",")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_API = "https://api-inference.huggingface.co/models"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def verify_key(api_key: str = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key.")
    return api_key

# ── Keyword extraction (no ML needed) ────────────────────────────────────────
STOPWORDS = {"the","a","an","and","or","but","in","on","at","to","for","of","with",
             "is","are","was","were","be","been","being","have","has","had","do",
             "does","did","will","would","could","should","may","might","this",
             "that","these","those","i","you","he","she","it","we","they","what",
             "which","who","when","where","how","not","by","from","as","so","if"}

def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return [w for w, _ in Counter(filtered).most_common(top_n)]

# ── Schemas ───────────────────────────────────────────────────────────────────
class TextIn(BaseModel):
    text: str

class SummaryOut(BaseModel):
    summary: str

class SentimentOut(BaseModel):
    label: str
    score: float

class KeywordsOut(BaseModel):
    keywords: list[str]

class LangOut(BaseModel):
    detected_language: str
    confidence: float

# ── HuggingFace helper ────────────────────────────────────────────────────────
def hf_post(model: str, payload: dict):
    r = httpx.post(f"{HF_API}/{model}", headers=HEADERS, json=payload, timeout=30)
    if r.status_code == 503:
        raise HTTPException(status_code=503, detail="AI model is loading, retry in 20 seconds.")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HuggingFace error: {r.text}")
    return r.json()

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok"}

@app.post("/summarize", response_model=SummaryOut, tags=["Text Tools"],
          summary="Summarize a long text into a short paragraph.")
def summarize(body: TextIn, _: str = Security(verify_key)):
    if len(body.text.split()) < 30:
        raise HTTPException(status_code=400, detail="Text must be at least 30 words.")
    result = hf_post("sshleifer/distilbart-cnn-12-6",
                     {"inputs": body.text, "parameters": {"max_length": 130, "min_length": 30}})
    return {"summary": result[0]["summary_text"]}

@app.post("/sentiment", response_model=SentimentOut, tags=["Text Tools"],
          summary="Detect positive or negative sentiment.")
def sentiment(body: TextIn, _: str = Security(verify_key)):
    result = hf_post("distilbert-base-uncased-finetuned-sst-2-english",
                     {"inputs": body.text[:512]})
    best = max(result[0], key=lambda x: x["score"])
    return {"label": best["label"], "score": round(best["score"], 4)}

@app.post("/keywords", response_model=KeywordsOut, tags=["Text Tools"],
          summary="Extract top keywords from text.")
def keywords(body: TextIn, _: str = Security(verify_key)):
    return {"keywords": extract_keywords(body.text)}

@app.post("/detect-language", response_model=LangOut, tags=["Text Tools"],
          summary="Detect the language of a text.")
def detect_language(body: TextIn, _: str = Security(verify_key)):
    result = hf_post("papluca/xlm-roberta-base-language-detection",
                     {"inputs": body.text[:256]})
    best = max(result[0], key=lambda x: x["score"])
    return {"detected_language": best["label"], "confidence": round(best["score"], 4)}
