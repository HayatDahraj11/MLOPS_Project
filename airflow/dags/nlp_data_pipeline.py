from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, '/opt/airflow')

from src.data.data_loader import download_dataset, prepare_dataset
from src.data.processor import TextProcessor
import pandas as pd

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
    'nlp_data_pipeline',
    default_args=default_args,
    description='NLP data processing pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Define tasks
start = DummyOperator(task_id='start', dag=dag)

download_data = PythonOperator(
    task_id='download_data',
    python_callable=download_dataset,
    dag=dag
)

prepare_data = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_dataset,
    dag=dag
)

def process_data():
    processor = TextProcessor()
    df = pd.read_parquet('data/processed/news_dataset.parquet')
    processed_df = processor.process_dataset(df)
    processed_df.to_parquet('data/processed/processed_news_dataset.parquet')

process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

# Task dependencies
start >> download_data >> prepare_data >> process_data_task >> end