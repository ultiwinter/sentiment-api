import pandas as pd
import requests
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sentiment import print_accuracy

PREDICT_API_URL = "http://localhost:8000/predict"
DATA_PATH = (Path(__file__).parent.parent / "data" / "data.csv").resolve()
OUT_PATH = DATA_PATH.parent / "with_predictions_textblob_api.csv"

df = pd.read_csv(DATA_PATH)

if "feedback" not in df.columns:
    raise KeyError(f"'feedback' column not found. Available: {list(df.columns)}")

texts = df["feedback"].astype(str).tolist()

resp = requests.post(PREDICT_API_URL, json={"texts": texts, "return_polarity": True})
resp.raise_for_status()

results = resp.json()["results"]  # list of {label, polarity?}
df["predicted_sentiment"] = [r["label"] for r in results]
df["predicted_polarity"] = [r.get("polarity") for r in results]

df.to_csv(OUT_PATH, index=False)
print_accuracy(df)
print(f"Saved: {OUT_PATH}")

RESOURCES_API_URL = "http://localhost:8000/metrics"
resp = requests.get(RESOURCES_API_URL)
resp.raise_for_status()
metrics = resp.json()   
print("Resources:", metrics)