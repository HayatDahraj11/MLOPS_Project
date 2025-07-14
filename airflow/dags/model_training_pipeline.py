from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, '/opt/airflow')

from src.models.trainer import ModelTrainer

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='NLP model training pipeline - trains 4 different models',
    schedule_interval=timedelta(days=1),
    catchup=False
)

def train_all_models():
    """Train all 4 models and return results"""
    trainer = ModelTrainer()
    trained_models, results = trainer.train_all_models()
    
    print("=" * 50)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 50)
    
    for result in results:
        print(f"Model: {result['model_name']}")
        print(f"  Validation Accuracy: {result['val_accuracy']:.4f}")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Validation F1: {result['val_f1_weighted']:.4f}")
        print(f"  Test F1: {result['test_f1_weighted']:.4f}")
        print("-" * 30)
    
    # Find and log best model
    if results:
        best_model = max(results, key=lambda x: x['val_accuracy'])
        print(f"🏆 BEST MODEL: {best_model['model_name']}")
        print(f"🎯 Best Validation Accuracy: {best_model['val_accuracy']:.4f}")
        return f"Training completed! Best model: {best_model['model_name']} with {best_model['val_accuracy']:.4f} accuracy"
    else:
        raise Exception("No models were trained successfully")

def validate_data():
    """Validate that all required data files exist"""
    required_files = [
        'data/processed/train.parquet',
        'data/processed/validation.parquet', 
        'data/processed/test.parquet'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    print("✅ All required data files found")
    return "Data validation passed"

# Define tasks
start = DummyOperator(task_id='start', dag=dag)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

train_models_task = PythonOperator(
    task_id='train_all_models',
    python_callable=train_all_models,
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

# Task dependencies
start >> validate_data_task >> train_models_task >> end