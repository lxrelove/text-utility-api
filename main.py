from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from transformers import pipeline
import os, re
from collections import Counter

app = FastAPI(
    title="AI Text Utility API",
    description="Summarize, analyze sentiment, extract keywords, and detect language from any text.",
    version="1.0.0",
)

API_KEY_NAME = "X-RapidAPI-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
VALID_API_KEYS = os.environ.get("VALID_API_KEYS", "dev-test-key").split(",")

def verify_key(api_key: str = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key.")
    return api_key

_summarizer = None
_sentiment  = None
_lang       = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return _summarizer

def get_sentiment():
    global _sentiment
    if _sentiment is None:
        _sentiment = pipeline("sentiment-analysis",
                              model="distilbert-base-uncased-finetuned-sst-2-english")
    return _sentiment

def get_lang():
    global _lang
    if _lang is None:
        _lang = pipeline("text-classification",
                         model="papluca/xlm-roberta-base-language-detection")
    return _lang

# Simple keyword extraction — no extra dependencies needed
STOPWORDS = {"the","a","an","and","or","but","in","on","at","to","for","of","with",
             "is","are","was","were","be","been","being","have","has","had","do",
             "does","did","will","would","could","should","may","might","this",
             "that","these","those","i","you","he","she","it","we","they","what",
             "which","who","when","where","how","not","by","from","as","so","if"}

def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return [w for w, _ in Counter(filtered).most_common(top_n)]

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

@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok"}

@app.post("/summarize", response_model=SummaryOut, tags=["Text Tools"],
          summary="Summarize a long text into a short paragraph.")
def summarize(body: TextIn, _: str = Security(verify_key)):
    if len(body.text.split()) < 30:
        raise HTTPException(status_code=400, detail="Text must be at least 30 words.")
    result = get_summarizer()(body.text, max_length=130, min_length=30, do_sample=False)
    return {"summary": result[0]["summary_text"]}

@app.post("/sentiment", response_model=SentimentOut, tags=["Text Tools"],
          summary="Detect positive or negative sentiment.")
def sentiment(body: TextIn, _: str = Security(verify_key)):
    result = get_sentiment()(body.text[:512])
    return {"label": result[0]["label"], "score": round(result[0]["score"], 4)}

@app.post("/keywords", response_model=KeywordsOut, tags=["Text Tools"],
          summary="Extract top keywords from text.")
def keywords(body: TextIn, _: str = Security(verify_key)):
    return {"keywords": extract_keywords(body.text)}

@app.post("/detect-language", response_model=LangOut, tags=["Text Tools"],
          summary="Detect the language of a text.")
def detect_language(body: TextIn, _: str = Security(verify_key)):
    result = get_lang()(body.text[:256])
    return {"detected_language": result[0]["label"], "confidence": round(result[0]["score"], 4)}
