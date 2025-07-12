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
    description='NLP model training pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False
)

def train_model():
    trainer = ModelTrainer()
    model, metrics = trainer.train_model()
    print(f"Training completed with accuracy: {metrics['accuracy']}")

# Define tasks
start = DummyOperator(task_id='start', dag=dag)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

# Task dependencies
start >> train_model_task >> end