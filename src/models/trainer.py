import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("news_classification")

    def prepare_data(self):
        df = pd.read_parquet('data/processed/processed_news_dataset.parquet')
        X = self.vectorizer.fit_transform(df['processed_text'])
        y = df['category']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        with mlflow.start_run():
            # Train model
            model = MultinomialNB()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Log metrics
            metrics = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_metrics({
                "accuracy": metrics['accuracy'],
                "weighted_avg_f1": metrics['weighted avg']['f1-score']
            })

            # Log model
            mlflow.sklearn.log_model(model, "model")
            mlflow.sklearn.log_model(self.vectorizer, "vectorizer")

        return model, metrics