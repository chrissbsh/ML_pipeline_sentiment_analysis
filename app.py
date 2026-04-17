from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = FastAPI(title="Sentiment Analysis API for Trading")

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

GEOPOLITICS_KEYWORDS = ["war", "tariffs", "sanctions", "conflict", "election"]
regex_pattern = re.compile(r'\b(' + '|'.join(GEOPOLITICS_KEYWORDS) + r')\b', re.IGNORECASE)

class TweetRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "L'API de trading est en ligne"}

@app.post("/analyze")
async def analyze_sentiment(request: TweetRequest):
    text = request.text

    is_geopolitical = bool(regex_pattern.search(text))

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores = probs[0].tolist()

    return {
        "text": text,
        "is_geopolitical": is_geopolitical,
        "sentiment": {
            "positive": round(scores[0], 4),
            "negative": round(scores[1], 4),
            "neutral": round(scores[2], 4)
        }
    }