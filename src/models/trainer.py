import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
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
import gc

class ModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        # Make MLflow optional to avoid blocking training
        self.use_mlflow = True
        self.mlflow_client = None
        
        try:
            # FIX 1: Use Docker service name instead of localhost
            mlflow.set_tracking_uri("http://mlflow:5001")
            mlflow.set_experiment("news_classification")
            self.mlflow_client = MlflowClient()
            print("✅ MLflow connection established")
        except Exception as e:
            print(f"MLflow not available: {e}")
            self.use_mlflow = False

    def load_data(self):
        """Load the preprocessed and split data"""
        # FIX 2: Use absolute paths for Docker environment
        base_path = '/opt/airflow/data/processed'
        
        try:
            train_df = pd.read_parquet(f'{base_path}/train.parquet')
            val_df = pd.read_parquet(f'{base_path}/validation.parquet')
            test_df = pd.read_parquet(f'{base_path}/test.parquet')
        except FileNotFoundError:
            # Fallback to relative paths
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

    def register_model_to_mlflow(self, model, vectorizer, model_name, metrics):
        """Register model and vectorizer to MLflow Model Registry"""
        if not self.use_mlflow:
            return
            
        try:
            # Register the main model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            
            # Register model
            try:
                model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name="news_classifier"
                )
                print(f"✅ Registered news_classifier version {model_version.version}")
                
                # Transition to Production if it's the best model
                self.mlflow_client.transition_model_version_stage(
                    name="news_classifier",
                    version=model_version.version,
                    stage="Production"
                )
                print(f"✅ Transitioned news_classifier to Production")
                
            except Exception as e:
                print(f"Model registration failed: {e}")
            
            # Register vectorizer separately
            try:
                mlflow.sklearn.log_model(vectorizer, "vectorizer")
                vectorizer_uri = f"runs:/{mlflow.active_run().info.run_id}/vectorizer"
                
                vectorizer_version = mlflow.register_model(
                    model_uri=vectorizer_uri,
                    name="news_classifier_vectorizer"
                )
                print(f"✅ Registered news_classifier_vectorizer version {vectorizer_version.version}")
                
                # Transition to Production
                self.mlflow_client.transition_model_version_stage(
                    name="news_classifier_vectorizer",
                    version=vectorizer_version.version,
                    stage="Production"
                )
                print(f"✅ Transitioned news_classifier_vectorizer to Production")
                
            except Exception as e:
                print(f"Vectorizer registration failed: {e}")
                
        except Exception as e:
            print(f"MLflow model registration failed: {e}")

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
        
        val_report = classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)
        test_report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        
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
                    
                    # FIX 3: Register best models to Model Registry
                    if val_accuracy > 0.5:  # Only register reasonably good models
                        self.register_model_to_mlflow(model, self.vectorizer, model_name, metrics)
                    
            except Exception as e:
                print(f"MLflow logging failed: {e}")
        
        # Save model locally as backup with absolute path
        models_dir = '/opt/airflow/models'
        try:
            os.makedirs(models_dir, exist_ok=True)
        except:
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)
            
        model_path = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # FIX 4: Force garbage collection after each model
        gc.collect()
        
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
        
        # FIX 5: Optimize model configurations for memory efficiency
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                solver='liblinear'  # More memory efficient
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=50,      # Reduced from 100
                max_depth=15,         # Limit tree depth
                max_features='sqrt',  # Use fewer features per tree
                n_jobs=1,            # Single thread to avoid memory spikes
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(50,),  # Reduced from (100, 50)
                max_iter=300,              # Reduced from 500
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        results = []
        trained_models = {}
        
        # Train each model
        for model_name, model in models.items():
            try:
                print(f"\n--- Training {model_name} ---")
                trained_model, metrics = self.train_single_model(
                    model, model_name, X_train, X_val, X_test, y_train, y_val, y_test
                )
                results.append(metrics)
                trained_models[model_name] = trained_model
                print(f"✅ {model_name} - Val Acc: {metrics['val_accuracy']:.4f}, Test Acc: {metrics['test_accuracy']:.4f}")
                
                # FIX 6: Clear variables after each model to free memory
                del trained_model
                gc.collect()
                
            except Exception as e:
                print(f"❌ {model_name} training failed: {e}")
                # Continue with other models even if one fails
                continue
        
        # Save vectorizer with absolute path
        vectorizer_dir = '/opt/airflow/models'
        try:
            os.makedirs(vectorizer_dir, exist_ok=True)
        except:
            vectorizer_dir = 'models'
            os.makedirs(vectorizer_dir, exist_ok=True)
            
        vectorizer_path = os.path.join(vectorizer_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✅ Vectorizer saved to {vectorizer_path}")
        
        # Find best model and register it
        if results:
            best_model = max(results, key=lambda x: x['val_accuracy'])
            print(f"\n🏆 Best model: {best_model['model_name']} with {best_model['val_accuracy']:.4f} validation accuracy")
            
            # FIX 7: Ensure best model is registered to MLflow
            if self.use_mlflow and best_model['val_accuracy'] > 0.4:
                try:
                    best_model_obj = trained_models.get(best_model['model_name'])
                    if best_model_obj:
                        with mlflow.start_run(run_name=f"best_{best_model['model_name']}_production"):
                            mlflow.sklearn.log_model(best_model_obj, "model")
                            mlflow.sklearn.log_model(self.vectorizer, "vectorizer")
                            self.register_model_to_mlflow(best_model_obj, self.vectorizer, best_model['model_name'], best_model)
                except Exception as e:
                    print(f"Failed to register best model: {e}")
        
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