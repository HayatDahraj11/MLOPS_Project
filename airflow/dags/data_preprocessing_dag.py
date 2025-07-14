from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import sys
import os
import pandas as pd

# Add project root to Python path
sys.path.insert(0, '/opt/airflow')

from src.data.processor import TextProcessor

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_preprocessing_pipeline',
    default_args=default_args,
    description='Data preprocessing pipeline for news classification',
    schedule_interval='@daily',
    catchup=False
)

def preprocess_data():
    """Preprocess the raw dataset"""
    processor = TextProcessor()
    df = pd.read_json('data/raw/News_Category_Dataset_v3.json', lines=True)
    df = df.dropna(subset=['headline', 'category'])
    processed_df = processor.process_dataset(df)
    os.makedirs('data/processed', exist_ok=True)
    processed_df.to_parquet('data/processed/news_dataset.parquet')
    print(f"Dataset processed and saved with {len(processed_df)} records")

start = DummyOperator(task_id='start', dag=dag)

process_data = PythonOperator(
    task_id='process_data',
    python_callable=preprocess_data,
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

start >> process_data >> end