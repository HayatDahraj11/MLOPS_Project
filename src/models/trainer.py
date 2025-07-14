import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pickle
import os

class ModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        # Make MLflow optional to avoid blocking training
        self.use_mlflow = True
        try:
            mlflow.set_tracking_uri("http://localhost:5001")
            mlflow.set_experiment("news_classification")
        except Exception as e:
            print(f"MLflow not available: {e}")
            self.use_mlflow = False

    def load_data(self):
        """Load the preprocessed and split data"""
        train_df = pd.read_parquet('data/processed/train.parquet')
        val_df = pd.read_parquet('data/processed/validation.parquet')
        test_df = pd.read_parquet('data/processed/test.parquet')
        
        return train_df, val_df, test_df

    def find_text_column(self, df):
        """Find the text column in the dataframe"""
        # Based on your data structure, prioritize headline first
        possible_columns = ['headline', 'short_description', 'processed_text', 'text', 'cleaned_text', 'content']
        
        for col in possible_columns:
            if col in df.columns:
                print(f"Using text column: {col}")
                return col
        
        # If none found, show available columns
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"No text column found. Available columns: {df.columns.tolist()}")

    def prepare_features(self, train_df, val_df, test_df):
        """Prepare TF-IDF features"""
        # Find the text column
        text_col = self.find_text_column(train_df)
        
        # Fit vectorizer on training data
        X_train = self.vectorizer.fit_transform(train_df[text_col])
        X_val = self.vectorizer.transform(val_df[text_col])
        X_test = self.vectorizer.transform(test_df[text_col])
        
        y_train = train_df['category']
        y_val = val_df['category']
        y_test = test_df['category']
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_single_model(self, model, model_name, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train a single model and return metrics"""
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val, y_pred_val)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        val_report = classification_report(y_val, y_pred_val, output_dict=True)
        test_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        metrics = {
            'model_name': model_name,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'val_f1_weighted': val_report['weighted avg']['f1-score'],
            'test_f1_weighted': test_report['weighted avg']['f1-score']
        }
        
        # Log to MLflow if available
        if self.use_mlflow:
            try:
                with mlflow.start_run(run_name=model_name):
                    mlflow.log_metrics({
                        "val_accuracy": val_accuracy,
                        "test_accuracy": test_accuracy,
                        "val_f1_weighted": val_report['weighted avg']['f1-score'],
                        "test_f1_weighted": test_report['weighted avg']['f1-score']
                    })
                    mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                print(f"MLflow logging failed: {e}")
        
        # Save model locally as backup
        os.makedirs('models', exist_ok=True)
        with open(f'models/{model_name.lower().replace(" ", "_")}.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        return model, metrics

    def train_all_models(self):
        """Train all models and return results"""
        # Load data
        train_df, val_df, test_df = self.load_data()
        print(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Show column info for debugging
        print(f"Train columns: {train_df.columns.tolist()}")
        
        # Prepare features
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_features(train_df, val_df, test_df)
        print(f"Features prepared: {X_train.shape[1]} features")
        
        # Define models to train
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        results = []
        trained_models = {}
        
        # Train each model
        for model_name, model in models.items():
            try:
                trained_model, metrics = self.train_single_model(
                    model, model_name, X_train, X_val, X_test, y_train, y_val, y_test
                )
                results.append(metrics)
                trained_models[model_name] = trained_model
                print(f"✅ {model_name} - Val Acc: {metrics['val_accuracy']:.4f}, Test Acc: {metrics['test_accuracy']:.4f}")
            except Exception as e:
                print(f"❌ {model_name} training failed: {e}")
        
        # Save vectorizer
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Find best model
        if results:
            best_model = max(results, key=lambda x: x['val_accuracy'])
            print(f"\n🏆 Best model: {best_model['model_name']} with {best_model['val_accuracy']:.4f} validation accuracy")
        
        return trained_models, results

    def train_model(self):
        """Main training function for Airflow compatibility"""
        trained_models, results = self.train_all_models()
        
        # Return best model metrics for Airflow logging
        if results:
            best_result = max(results, key=lambda x: x['val_accuracy'])
            return trained_models[best_result['model_name']], best_result
        else:
            raise Exception("No models were trained successfully")