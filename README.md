# AI Text Utility API

A FastAPI-based REST API that provides four NLP endpoints powered by free HuggingFace models.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Check if the API is running |
| POST | /summarize | Summarize long text (min 30 words) |
| POST | /sentiment | Detect POSITIVE / NEGATIVE sentiment |
| POST | /keywords | Extract keywords from text |
| POST | /detect-language | Detect language of text |

## Request format (all POST endpoints)

```json
{ "text": "Your text goes here..." }
```

## Authentication

Pass your API key in the header:
```
X-RapidAPI-Key: your-key-here
```

## Run locally

```bash
pip install -r requirements.txt
VALID_API_KEYS=dev-test-key uvicorn main:app --reload
```

Then visit http://localhost:8000/docs for the interactive Swagger UI.

## Deploy to Render (free)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New Web Service
3. Connect your repo, set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable: `VALID_API_KEYS=dev-test-key`
5. Deploy → copy the public URL

## List on RapidAPI

1. Go to https://rapidapi.com/provider
2. Add New API → paste your Render URL as the base URL
3. Set Authentication: Header → X-RapidAPI-Key
4. Add endpoints matching the table above
5. Set pricing tiers (Freemium recommended)
6. Submit for review
