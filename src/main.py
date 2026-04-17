from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# 1. Initialisation de FastAPI
app = FastAPI(title="Sentiment Analysis API for Trading")

# 2. Chargement du modèle (Une seule fois au démarrage)
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True)

# 3. Notre filtre géopolitique (Regex)
GEOPOLITICS_KEYWORDS = ["war", "tariffs", "sanctions", "conflict", "election"]
regex_pattern = re.compile(r'\b(' + '|'.join(GEOPOLITICS_KEYWORDS) + r')\b', re.IGNORECASE)

# 4. Modèle de données pour les requêtes (Pydantic)
class TweetRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "L'API de trading est en ligne"}

@app.post("/analyze")
async def analyze_sentiment(request: TweetRequest):
    text = request.text
    
    # Vérification du filtre géopolitique
    is_geopolitical = bool(regex_pattern.search(text))
    
    # Inférence avec FinBERT
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad(): # On désactive le calcul des gradients pour aller plus vite
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # FinBERT labels: 0 -> Positive, 1 -> Negative, 2 -> Neutral
    # Attention: L'ordre peut varier selon le modèle, ici c'est souvent [Pos, Neg, Neu]
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

if __name__ == "__main__":
    print("to launch: uvicorn main:app --reload")