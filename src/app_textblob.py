
from textblob import TextBlob
from typing import Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import psutil
import os
from tqdm import tqdm

class TextBlobSentimentService:
    LABELS = ["negative", "neutral", "positive"]

    def predict_one(self, text: str) -> Tuple[str, float]:
        pol = float(TextBlob(str(text)).sentiment.polarity)  # [-1, 1]
        if -0.1 <= pol <= 0.25:
            label = "neutral"
        elif pol > 0.25:
            label = "positive"
        else:
            label = "negative"
        return label, pol


app = FastAPI(title="Sentiment API (TextBlob)", version="1.0.0")

# --- CORS (Frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = TextBlobSentimentService()

# Latency-Header
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start)*1000:.2f}"
    return response

# --- Timing Middleware ---
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
    return response

# --- Schemas ---
class PredictRequest(BaseModel):
    text: Optional[str] = Field(None, description="Single text")
    texts: Optional[List[str]] = Field(None, description="Batch of texts")
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "not great not terrible"},
                {"texts": ["this is bad", "meh", "pretty good"]}
            ]
        }
    }

class Prediction(BaseModel):
    label: str
    polarity: Optional[float] = None

class PredictResponse(BaseModel):
    results: List[Prediction]
    # Optional example for Swagger
    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {"label": "positive", "polarity": 0.65},
                    {"label": "neutral", "polarity": 0.00},
                    {"label": "negative", "polarity": -0.60}
                ]
            }
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "engine": "textblob", "labels": service.LABELS}

@app.get("/metrics")
def metrics():
    p = psutil.Process(os.getpid())
    mem = p.memory_info().rss  # Bytes
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.0),
        "mem_rss_bytes": mem,
        "mem_rss_mb": round(mem / (1024*1024), 2),
        "num_threads": p.num_threads(),
    }


# --- Endpoints ---
@app.get("/Resources")
def metrics():
    """Resources Unilized(System + this process)."""
    p = psutil.Process(os.getpid())
    mem = p.memory_info().rss  # Bytes
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.0),
        "mem_rss_bytes": mem,
        "num_threads": p.num_threads(),
    }

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict sentiment",
    tags=["inference"],
)
def predict(req: PredictRequest):
    if not req.text and not req.texts:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'texts'.")

    inputs = [req.text] if req.text is not None else list(map(str, req.texts or []))
    if not inputs:
        raise HTTPException(status_code=400, detail="'texts' must contain at least one item.")

    results = []
    for t in tqdm(inputs,desc="Predicting sentiment"):
        label, pol = service.predict_one(t)
        results.append(Prediction(label=label, polarity=pol))
    return PredictResponse(results=results)
