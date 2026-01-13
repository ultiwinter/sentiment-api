import pandas as pd
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from tqdm import tqdm



# sentiment analysis (TextBlob)
def get_sentiment_label(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > -0.09 and polarity < 0.09:
        return "neutral"
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    
def print_accuracy(df):
    if "predicted_sentiment" not in df.columns:
        raise ValueError("Predicted sentiment column not found. Please run get_sentiment_prediction() first.")
    if "sentiment (text)" in df.columns:
        y_true = df["sentiment (text)"]
        y_pred = df["predicted_sentiment"]
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))
    else:
        print("No ground truth sentiment labels found for accuracy calculation.")
    
class SentimentAnalyzer:
    def __init__(self, model="textblob", data_path=None):
        if data_path is not None:
            self.data_path = Path(data_path)
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file {self.data_path} does not exist.")  
        else:
            self.data_path = None 
            raise FileNotFoundError("Data Path is empty.")   

        if model not in ["textblob", "transformers"]:
            raise ValueError("Model must be either 'textblob' or 'transformers'")   
        
        self.model = model
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path) if self.data_path else None

    def _get_sentiment_single(self, text):
        if self.model == "textblob":
            return get_sentiment_label(text)
        # elif self.model == "transformers":
        #     sentiment_pipeline = pipeline(
        #         "sentiment-analysis",
        #         model="distilbert-base-uncased"
        #     )
        #     try:
        #         result = sentiment_pipeline(str(text))[0]
        #         return result["label"].lower()  # labels: 'negative', 'neutral', 'positive'
        #     except Exception as e:
        #         print(f"Error processing text: {text}, error: {e}")
        #         return "neutral"
            
    def get_sentiment_prediction(self):
        preds = []
        for text in tqdm(self.df["feedback"], desc="Analyzing sentiments"):
            result = self._get_sentiment_single(text)
            
            preds.append(result.lower())  # labels: 'negative', 'neutral', 'positive'

        self.df["predicted_sentiment"] = preds
        print("Sentiment analysis completed.")
        return preds
    
    def print_accuracy(self):
        if "predicted_sentiment" not in self.df.columns:
            raise ValueError("Predicted sentiment column not found. Please run get_sentiment_prediction() first.")
        if "sentiment (text)" in self.df.columns:
            y_true = self.df["sentiment (text)"]
            y_pred = self.df["predicted_sentiment"]
            print("Accuracy:", accuracy_score(y_true, y_pred))
            print(classification_report(y_true, y_pred))
        else:
            print("No ground truth sentiment labels found for accuracy calculation.")

    def get_df(self):
        if self.df is None:
            raise ValueError("DataFrame is empty. Please run get_sentiment_prediction() first.")
        return self.df



if __name__ == "__main__":
    print("Starting sentiment analysis...")

    data_path = Path(__file__).parent.parent / "data" / "data.csv"

    sentiment_analyzer = SentimentAnalyzer(model="textblob", data_path=data_path)
    preds = sentiment_analyzer.get_sentiment_prediction()

    sentiment_analyzer.print_accuracy()
    df = sentiment_analyzer.get_df()

    df.to_csv("data/feedback_with_predictions.csv", index=False)
